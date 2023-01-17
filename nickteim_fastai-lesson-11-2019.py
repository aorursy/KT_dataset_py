

%load_ext autoreload

%autoreload 2



%matplotlib inline
x_train,y_train,x_valid,y_valid = get_data()



x_train,x_valid = normalize_to(x_train,x_valid)

train_ds,valid_ds = Dataset(x_train, y_train),Dataset(x_valid, y_valid)



nh,bs = 50,512

c = y_train.max().item()+1

loss_func = F.cross_entropy



data = DataBunch(*get_dls(train_ds, valid_ds, bs), c)
mnist_view = view_tfm(1,28,28)

cbfs = [Recorder,

        partial(AvgStatsCallback,accuracy),

        CudaCallback,

        partial(BatchTransformXCallback, mnist_view)]
nfs = [8,16,32,64,64]
class ConvLayer(nn.Module):

    def __init__(self, ni, nf, ks=3, stride=2, sub=0., **kwargs):

        super().__init__()

        self.conv = nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride, bias=True)

        self.relu = GeneralRelu(sub=sub, **kwargs)

    

    def forward(self, x): return self.relu(self.conv(x))

    

    @property

    def bias(self): return -self.relu.sub

    @bias.setter

    def bias(self,v): self.relu.sub = -v

    @property

    def weight(self): return self.conv.weight
learn,run = get_learn_run(nfs, data, 0.6, ConvLayer, cbs=cbfs) #all above code is the as before tog get something to compare to 

run.fit(2, learn)#we see here we get a acc. on about 95%
learn,run = get_learn_run(nfs, data, 0.6, ConvLayer, cbs=cbfs)
def get_batch(dl, run):

    run.xb,run.yb = next(iter(dl))

    for cb in run.cbs: cb.set_runner(run)

    run('begin_batch')

    return run.xb,run.yb
xb,yb = get_batch(data.train_dl, run) #one minibatch


#export

def find_modules(m, cond):

    if cond(m): return [m]

    return sum([find_modules(o,cond) for o in m.children()], []) #note we are using recursing since moduels is like a tree and the mudiels has to finde the modules 



def is_lin_layer(l):

    lin_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.ReLU)

    return isinstance(l, lin_layers)


mods = find_modules(learn.model, lambda o: isinstance(o,ConvLayer)) #find all the modules of type ConvLayer (find:modules does that)


mods #so here is our list of all the conv layers 


def append_stat(hook, mod, inp, outp):

    d = outp.data

    hook.mean,hook.std = d.mean().item(),d.std().item() #... hooks grap the mean and std of a partigular module 
mdl = learn.model.cuda()
with Hooks(mods, append_stat) as hooks: #so now we can create a hook ... that gonne grap the mean and std 

    mdl(xb)

    for hook in hooks: print(hook.mean,hook.std) #print all of the means and stds 

        #and we can se that the means and stds are not 0 and 1. so the mean is to high and the std is to low 
def lsuv_module(m, xb):

    h = Hook(m, append_stat) #so first of all we hook it 



    while mdl(xb) is not None and abs(h.mean)  > 1e-3: m.bias -= h.mean #so to get a better mean we create a loop 

        #calls the model ( mdl(xb)) and passing in the minibatch we have(xb) and check is the absolut value of the mean is close to 0( abs(h.mean)> 1e-3)

        #and if it is not then we subrat the the mean from the bias. so it just keep looping through intil we get about a zero mean 

    while mdl(xb) is not None and abs(h.std-1) > 1e-3: m.weight.data /= h.std #we do the same thing for the std where  we check if std-1 is near zero (abs(h.std-1) > 1e-3)

        #and if it is not then will divide the weights with the std till we get there 



    h.remove()

    return h.mean,h.std
for m in mods: print(lsuv_module(m, xb)) #so we see we get something very close to what we want though the mean isnt 0 but it is close and the std is perfect 


%time run.fit(2, learn)#and we now see the scc. is about 97 % so its better 