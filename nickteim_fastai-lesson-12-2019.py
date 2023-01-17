%load_ext autoreload
%autoreload 2

%matplotlib inline
path = datasets.untar_data(datasets.URLs.IMAGENETTE_160)#same  data as before
tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]
bs = 64

il = ImageList.from_files(path, tfms=tfms)
sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='val'))
ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())
data = ll.to_databunch(bs, c_in=3, c_out=10, num_workers=4)

img1 = PIL.Image.open(ll.train.x.items[0])
img1
img2 = PIL.Image.open(ll.train.x.items[4000])
img2
mixed_up = ll.train.x[0] * 0.3 + ll.train.x[4000] * 0.7 #so we are mixing two images together where we take 30 procent of the one image and 70 of the other image
plt.imshow(mixed_up.permute(1,2,0)); 

# PyTorch has a log-gamma but not a gamma, so we'll create one
Γ = lambda x: x.lgamma().exp()


facts = [math.factorial(i) for i in range(7)]
plt.plot(range(7), facts, 'ro')
plt.plot(torch.linspace(0,6), Γ(torch.linspace(0,6)+1))
plt.legend(['factorial','Γ']);

torch.linspace(0,0.9,10)

_,axs = plt.subplots(1,2, figsize=(12,4))
x = torch.linspace(0,1, 100)
for α,ax in zip([0.1,0.8], axs):
    α = tensor(α)
#     y = (x.pow(α-1) * (1-x).pow(α-1)) / (gamma_func(α ** 2) / gamma_func(α)) #so this is in programming 'sprog'
    y = (x**(α-1) * (1-x)**(α-1)) / (Γ(α)**2 / Γ(2*α)) #and this is like if we did it in math 
    ax.plot(x,y)
    ax.set_title(f"α={α:.1}")
    #so we get to pick alfa and if it is high it is very likely we get a equarel mix (procentage of each image, combined in one)
    #and if it is low unlikely, so more likely that one or the other will be more dominant.
#export
class NoneReduce():
    def __init__(self, loss_func): 
        self.loss_func,self.old_red = loss_func,None
        
    def __enter__(self):
        if hasattr(self.loss_func, 'reduction'):
            self.old_red = getattr(self.loss_func, 'reduction')
            setattr(self.loss_func, 'reduction', 'none')
            return self.loss_func
        else: return partial(self.loss_func, reduction='none')
        
    def __exit__(self, type, value, traceback):
        if self.old_red is not None: setattr(self.loss_func, 'reduction', self.old_red)
#export
from torch.distributions.beta import Beta

def unsqueeze(input, dims):
    for dim in listify(dims): input = torch.unsqueeze(input, dim)
    return input

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss #so takes the sum of all the losses in a minibtach and take the mean to it
#note we 
#so mixup is gonna change our loss function. So we need to know what loss function to change 
class MixUp(Callback): 
    _order = 90 #Runs after normalization and cuda
    def __init__(self, α:float=0.4): self.distrib = Beta(tensor([α]), tensor([α]))
    
    def begin_fit(self): self.old_loss_func,self.run.loss_func = self.run.loss_func,self.loss_func #when you start fitting we find out what the old loss function on the learner was and tore it away
    
    def begin_batch(self):
        if not self.in_train: return #Only mixup things during training
        λ = self.distrib.sample((self.yb.size(0),)).squeeze().to(self.xb.device)
        λ = torch.stack([λ, 1-λ], 1)
        self.λ = unsqueeze(λ.max(1)[0], (1,2,3))
        shuffle = torch.randperm(self.yb.size(0)).to(self.xb.device) #grap all the images and shuffle them 
        xb1,self.yb1 = self.xb[shuffle],self.yb[shuffle] #randomly picked one 
        self.run.xb = lin_comb(self.xb, xb1, self.λ)#linear combination (lin_comb) of our actual images(xb) and some randomly picked images(xb1) in that imibatch
        
    def after_fit(self): self.run.loss_func = self.old_loss_func
    
    def loss_func(self, pred, yb):
        if not self.in_train: return self.old_loss_func(pred, yb) #if it is in validation there is no mixup involed 
        with NoneReduce(self.old_loss_func) as loss_func: #and when we are training we will calulate the loss on to different set of images ...
            loss1 = loss_func(pred, yb) # ... one is just the regular set from training data ...
            loss2 = loss_func(pred, self.yb1)#... and the other one is randomly picked 
        loss = lin_comb(loss1, loss2, self.λ) #and our loss is a linear combination of our the loss of our normaly batched loss and our randomly batched loss
        return reduce_loss(loss, getattr(self.old_loss_func, 'reduction', 'mean'))
nfs = [32,64,128,256,512]

def get_learner(nfs, data, lr, layer, loss_func=F.cross_entropy,
                cb_funcs=None, opt_func=optim.SGD, **kwargs):
    model = get_cnn_model(data, nfs, layer, **kwargs)
    init_cnn(model)
    return Learner(model, data, loss_func, lr=lr, cb_funcs=cb_funcs, opt_func=opt_func)
cbfs = [partial(AvgStatsCallback,accuracy),
        CudaCallback, 
        ProgressCallback,
        partial(BatchTransformXCallback, norm_imagenette),
        MixUp]

learn = get_learner(nfs, data, 0.4, conv_layer, cb_funcs=cbfs)

learn.fit(1)
#if there is a lot of nose because of wrong or no labels this can be realy good 
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ε:float=0.1, reduction='mean'):
        super().__init__()
        self.ε,self.reduction = ε,reduction
    
    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return lin_comb(loss/c, nll, self.ε)
cbfs = [partial(AvgStatsCallback,accuracy),
        CudaCallback,
        ProgressCallback,
        partial(BatchTransformXCallback, norm_imagenette)]
learn = get_learner(nfs, data, 0.4, conv_layer, cb_funcs=cbfs, loss_func=LabelSmoothingCrossEntropy())
learn.fit(1)
assert learn.loss_func.reduction == 'mean'

%load_ext autoreload
%autoreload 2

%matplotlib inline

# export 
import apex.fp16_utils as fp16
bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
def bn_to_float(model):
    if isinstance(model, bn_types): model.float()#batch norm goes to float 
    for child in model.children():  bn_to_float(child)
    return model
def model_to_half(model):
    model = model.half() #turns model in half 
    return bn_to_float(model)

model = nn.Sequential(nn.Linear(10,30), nn.BatchNorm1d(30), nn.Linear(30,2)).cuda()
model = model_to_half(model)
def check_weights(model):
    for i,t in enumerate([torch.float16, torch.float32, torch.float16]):
        assert model[i].weight.dtype == t
        assert model[i].bias.dtype   == t

check_weights(model)
model = nn.Sequential(nn.Linear(10,30), nn.BatchNorm1d(30), nn.Linear(30,2)).cuda()
model = fp16.convert_network(model, torch.float16)
check_weights(model)
from torch.nn.utils import parameters_to_vector

def get_master(model, flat_master=False):
    model_params = [param for param in model.parameters() if param.requires_grad]
    if flat_master:
        master_param = parameters_to_vector([param.data.float() for param in model_params])
        master_param = torch.nn.Parameter(master_param, requires_grad=True)
        if master_param.grad is None: master_param.grad = master_param.new(*master_param.size())
        return model_params, [master_param]
    else:
        master_params = [param.clone().float().detach() for param in model_params]
        for param in master_params: param.requires_grad_(True)
        return model_params, master_params
model_p,master_p = get_master(model)
model_p1,master_p1 = fp16.prep_param_lists(model)
def same_lists(ps1, ps2):
    assert len(ps1) == len(ps2)
    for (p1,p2) in zip(ps1,ps2): 
        assert p1.requires_grad == p2.requires_grad
        assert torch.allclose(p1.data.float(), p2.data.float())
same_lists(model_p,model_p1)
same_lists(model_p,master_p)
same_lists(master_p,master_p1)
same_lists(model_p1,master_p1)
model1 = nn.Sequential(nn.Linear(10,30), nn.Linear(30,2)).cuda()
model1 = fp16.convert_network(model1, torch.float16)
model_p,master_p = get_master(model1, flat_master=True)
model_p1,master_p1 = fp16.prep_param_lists(model1, flat_master=True)
same_lists(model_p,model_p1)
same_lists(master_p,master_p1)
assert len(master_p[0]) == 10*30 + 30 + 30*2 + 2
assert len(master_p1[0]) == 10*30 + 30 + 30*2 + 2
def get_master(opt, flat_master=False):
    model_params = [[param for param in pg if param.requires_grad] for pg in opt.param_groups]
    if flat_master:
        master_params = []
        for pg in model_params:
            mp = parameters_to_vector([param.data.float() for param in pg])
            mp = torch.nn.Parameter(mp, requires_grad=True)
            if mp.grad is None: mp.grad = mp.new(*mp.size())
            master_params.append(mp)
    else:
        master_params = [[param.clone().float().detach() for param in pg] for pg in model_params]
        for pg in master_params:
            for param in pg: param.requires_grad_(True)
    return model_params, master_params
def to_master_grads(model_params, master_params, flat_master:bool=False)->None:
    if flat_master:
        if master_params[0].grad is None: master_params[0].grad = master_params[0].data.new(*master_params[0].data.size())
        master_params[0].grad.data.copy_(parameters_to_vector([p.grad.data.float() for p in model_params]))
    else:
        for model, master in zip(model_params, master_params):
            if model.grad is not None:
                if master.grad is None: master.grad = master.data.new(*master.data.size())
                master.grad.data.copy_(model.grad.data)
            else: master.grad = None

x = torch.randn(20,10).half().cuda()
z = model(x)
loss = F.cross_entropy(z, torch.randint(0, 2, (20,)).cuda())
loss.backward()
to_master_grads(model_p, master_p)
def check_grads(m1, m2):
    for p1,p2 in zip(m1,m2): 
        if p1.grad is None: assert p2.grad is None
        else: assert torch.allclose(p1.grad.data, p2.grad.data)
check_grads(model_p, master_p)
fp16.model_grads_to_master_grads(model_p, master_p)
check_grads(model_p, master_p)

from torch._utils import _unflatten_dense_tensors

def to_model_params(model_params, master_params, flat_master:bool=False)->None:
    if flat_master:
        for model, master in zip(model_params, _unflatten_dense_tensors(master_params[0].data, model_params)):
            model.data.copy_(master)
    else:
        for model, master in zip(model_params, master_params): model.data.copy_(master.data)
def get_master(opt, flat_master=False):
    model_pgs = [[param for param in pg if param.requires_grad] for pg in opt.param_groups]
    if flat_master:
        master_pgs = []
        for pg in model_pgs:
            mp = parameters_to_vector([param.data.float() for param in pg])
            mp = torch.nn.Parameter(mp, requires_grad=True)
            if mp.grad is None: mp.grad = mp.new(*mp.size())
            master_pgs.append([mp])
    else:
        master_pgs = [[param.clone().float().detach() for param in pg] for pg in model_pgs]
        for pg in master_pgs:
            for param in pg: param.requires_grad_(True)
    return model_pgs, master_pgs
# export 
def to_master_grads(model_pgs, master_pgs, flat_master:bool=False)->None:
    for (model_params,master_params) in zip(model_pgs,master_pgs):
        fp16.model_grads_to_master_grads(model_params, master_params, flat_master=flat_master)

# export 
def to_model_params(model_pgs, master_pgs, flat_master:bool=False)->None:
    for (model_params,master_params) in zip(model_pgs,master_pgs):
        fp16.master_params_to_model_params(model_params, master_params, flat_master=flat_master)
class MixedPrecision(Callback):
    _order = 99
    def __init__(self, loss_scale=512, flat_master=False): #multiply by loss_scale and then divide by it to get the right scaleing 
        assert torch.backends.cudnn.enabled, "Mixed precision training requires cudnn."
        self.loss_scale,self.flat_master = loss_scale,flat_master

    def begin_fit(self):
        self.run.model = fp16.convert_network(self.model, dtype=torch.float16)
        self.model_pgs, self.master_pgs = get_master(self.opt, self.flat_master)
        #Changes the optimizer so that the optimization step is done in FP32.
        self.run.opt.param_groups = self.master_pgs #Put those param groups inside our runner.
        
    def after_fit(self): self.model.float()

    def begin_batch(self): self.run.xb = self.run.xb.half() #Put the inputs to half precision
    def after_pred(self):  self.run.pred = self.run.pred.float() #Compute the loss in FP32
    def after_loss(self):  self.run.loss *= self.loss_scale #Loss scaling to avoid gradient underflow

    def after_backward(self):
        #Copy the gradients to master and unscale
        to_master_grads(self.model_pgs, self.master_pgs, self.flat_master)
        for master_params in self.master_pgs:
            for param in master_params:
                if param.grad is not None: param.grad.div_(self.loss_scale)

    def after_step(self):
        #Zero the gradients of the model since the optimizer is disconnected.
        self.model.zero_grad()
        #Update the params from master to model.
        to_model_params(self.model_pgs, self.master_pgs, self.flat_master)
path = datasets.untar_data(datasets.URLs.IMAGENETTE_160)

tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]
bs = 64

il = ImageList.from_files(path, tfms=tfms)
sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='val'))
ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())
data = ll.to_databunch(bs, c_in=3, c_out=10, num_workers=4)
nfs = [32,64,128,256,512]

def get_learner(nfs, data, lr, layer, loss_func=F.cross_entropy,
                cb_funcs=None, opt_func=adam_opt(), **kwargs):
    model = get_cnn_model(data, nfs, layer, **kwargs)
    init_cnn(model)
    return Learner(model, data, loss_func, lr=lr, cb_funcs=cb_funcs, opt_func=opt_func)

cbfs = [partial(AvgStatsCallback,accuracy),
        ProgressCallback,
        CudaCallback,
        partial(BatchTransformXCallback, norm_imagenette)]

learn = get_learner(nfs, data, 1e-2, conv_layer, cb_funcs=cbfs)

learn.fit(1)
cbfs = [partial(AvgStatsCallback,accuracy),
        CudaCallback,
        ProgressCallback,
        partial(BatchTransformXCallback, norm_imagenette),
        MixedPrecision]

learn = get_learner(nfs, data, 1e-2, conv_layer, cb_funcs=cbfs)
learn.fit(1)
test_eq(next(learn.model.parameters()).type(), 'torch.cuda.FloatTensor')
# export 
def test_overflow(x):
    s = float(x.float().sum())
    return (s == float('inf') or s == float('-inf') or s != s)
x = torch.randn(512,1024).cuda()
test_overflow(x)
x[123,145] = float('inf')
test_overflow(x)
%timeit test_overflow(x)

%timeit torch.isnan(x).any().item()

# export 
def grad_overflow(param_groups):
    for group in param_groups:
        for p in group:
            if p.grad is not None:
                s = float(p.grad.data.float().sum())
                if s == float('inf') or s == float('-inf') or s != s: return True
    return False
# export 
class MixedPrecision(Callback):
    _order = 99
    def __init__(self, loss_scale=512, flat_master=False, dynamic=True, max_loss_scale=2.**24, div_factor=2.,
                 scale_wait=500):
        assert torch.backends.cudnn.enabled, "Mixed precision training requires cudnn."
        self.flat_master,self.dynamic,self.max_loss_scale = flat_master,dynamic,max_loss_scale
        self.div_factor,self.scale_wait = div_factor,scale_wait
        self.loss_scale = max_loss_scale if dynamic else loss_scale

    def begin_fit(self):
        self.run.model = fp16.convert_network(self.model, dtype=torch.float16)
        self.model_pgs, self.master_pgs = get_master(self.opt, self.flat_master)
        #Changes the optimizer so that the optimization step is done in FP32.
        self.run.opt.param_groups = self.master_pgs #Put those param groups inside our runner.
        if self.dynamic: self.count = 0

    def begin_batch(self): self.run.xb = self.run.xb.half() #Put the inputs to half precision
    def after_pred(self):  self.run.pred = self.run.pred.float() #Compute the loss in FP32
    def after_loss(self):  
        if self.in_train: self.run.loss *= self.loss_scale #Loss scaling to avoid gradient underflow

    def after_backward(self):
        #First, check for an overflow
        if self.dynamic and grad_overflow(self.model_pgs):
            #Divide the loss scale by div_factor, zero the grad (after_step will be skipped)
            self.loss_scale /= self.div_factor
            self.model.zero_grad()
            return True #skip step and zero_grad
        #Copy the gradients to master and unscale
        to_master_grads(self.model_pgs, self.master_pgs, self.flat_master)
        for master_params in self.master_pgs:
            for param in master_params:
                if param.grad is not None: param.grad.div_(self.loss_scale)
        #Check if it's been long enough without overflow
        if self.dynamic:
            self.count += 1
            if self.count == self.scale_wait:
                self.count = 0
                self.loss_scale *= self.div_factor

    def after_step(self):
        #Zero the gradients of the model since the optimizer is disconnected.
        self.model.zero_grad()
        #Update the params from master to model.
        to_model_params(self.model_pgs, self.master_pgs, self.flat_master)
cbfs = [partial(AvgStatsCallback,accuracy),
        CudaCallback,
        ProgressCallback,
        partial(BatchTransformXCallback, norm_imagenette),
        MixedPrecision]
learn = get_learner(nfs, data, 1e-2, conv_layer, cb_funcs=cbfs)
learn.fit(1)
learn.cbs[-1].loss_scale
%load_ext autoreload
%autoreload 2

%matplotlib inline

path = datasets.untar_data(datasets.URLs.IMAGENETTE_160)

size = 128
tfms = [make_rgb, RandomResizedCrop(size, scale=(0.35,1)), np_to_float, PilRandomFlip()] #a minimum scale of 0.35 to work "virker til" good. And we are not gonna use an other argutation then fliping 

bs = 64

il = ImageList.from_files(path, tfms=tfms)
sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='val'))
ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())

ll.valid.x.tfms = [make_rgb, CenterCrop(size), np_to_float]

data = ll.to_databunch(bs, c_in=3, c_out=10, num_workers=8)

#export
def noop(x): return x

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)
#export
act_fn = nn.ReLU(inplace=True) #and for our activation function we are just gonna use Relu for now 

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

def conv_layer(ni, nf, ks=3, stride=1, zero_bn=False, act=True):
    bn = nn.BatchNorm2d(nf)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.) #we initizise the weight to sometimes have weight of 1 and sometimes to have weights of 1 
    layers = [conv(ni, nf, ks, stride=stride), bn] #the layers start with a conv with some stide followed by a batch norm(bn)
    if act: layers.append(act_fn) #and optionaly we can add an activation function (act_fn)
    return nn.Sequential(*layers) #a conv layer is sequential containing a bunch of layers 
#export
class ResBlock(nn.Module):
    def __init__(self, expansion, ni, nh, stride=1): #so expansion is 1 for resnet 18 or 34 and and 4 if it is bigger 
        super().__init__()
        nf,ni = nh*expansion,ni*expansion
        layers  = [conv_layer(ni, nh, 3, stride=stride),
                   conv_layer(nh, nf, 3, zero_bn=True, act=False) #if expansion is 1 we just we just add one exstra conv layer but...
        ] if expansion == 1 else [ #...if expansion is bigger then 1 we add two exxtra conv layers 
                   conv_layer(ni, nh, 1),
                   conv_layer(nh, nh, 3, stride=stride),
                   conv_layer(nh, nf, 1, zero_bn=True, act=False)
        ]
        self.convs = nn.Sequential(*layers)
        self.idconv = noop if ni==nf else conv_layer(ni, nf, 1, act=False) #Conv(1x1) if the number of inputs is different to the numbers of filters we add a conv layer
        self.pool = noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True) #AvgPool if the stride is other then one we add an Averge pooling 

    def forward(self, x): return act_fn(self.convs(x) + self.idconv(self.pool(x)))
#export
class XResNet(nn.Sequential):
    @classmethod
    def create(cls, expansion, layers, c_in=3, c_out=1000):
        nfs = [c_in, (c_in+1)*8, 64, 64] #start wift setting up how many filters there is gonna be for the first 3 layer. and the first 3 layers will start with 3 channels(c_in)
        #and the output to the first layer will be the c_in plus 1 times 8, and the reason for this is that this is 32 to the secound layer which is the same as the aticle "the bag of thrich uses".
        #the reason we multiply by 8 is because invidia grafic card like everything to be a multitude of 8 
        stem = [conv_layer(nfs[i], nfs[i+1], stride=2 if i==0 else 1) #the stem is the very start of a of a CNN an it is just the 3 conv layers (nfs[i], nfs[i+1], stride=2)
            for i in range(3)]

        nfs = [64//expansion,64,128,256,512]
        res_layers = [cls._make_layer(expansion, nfs[i], nfs[i+1], #now we are gonna create a bunch of resbloks. #Gonna create a resnet blok for every res layer 
                                      n_blocks=l, stride=1 if i==0 else 2)
                  for i,l in enumerate(layers)]
        res = cls(
            *stem,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *res_layers,
            nn.AdaptiveAvgPool2d(1), Flatten(),
            nn.Linear(nfs[-1]*expansion, c_out),
        )
        init_cnn(res)
        return res

    @staticmethod
    def _make_layer(expansion, ni, nf, n_blocks, stride):
        return nn.Sequential(
            *[ResBlock(expansion, ni if i==0 else nf, nf, stride if i==0 else 1) #createing the resblok 
              for i in range(n_blocks)])

#create all of our resnet 
def xresnet18 (**kwargs): return XResNet.create(1, [2, 2,  2, 2], **kwargs) #exsampel here is the [2, 2,  2, 2] how many bloks we want in each layer and 
#expansion is 1 
def xresnet34 (**kwargs): return XResNet.create(1, [3, 4,  6, 3], **kwargs)
def xresnet50 (**kwargs): return XResNet.create(4, [3, 4,  6, 3], **kwargs)
def xresnet101(**kwargs): return XResNet.create(4, [3, 4, 23, 3], **kwargs)
def xresnet152(**kwargs): return XResNet.create(4, [3, 8, 36, 3], **kwargs)
cbfs = [partial(AvgStatsCallback,accuracy), ProgressCallback, CudaCallback,
        partial(BatchTransformXCallback, norm_imagenette),
#         partial(MixUp, alpha=0.2)
       ]
loss_func = LabelSmoothingCrossEntropy()
arch = partial(xresnet18, c_out=10)
opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)
#export
def get_batch(dl, learn):
    learn.xb,learn.yb = next(iter(dl))
    learn.do_begin_fit(0)
    learn('begin_batch')
    learn('after_fit')
    return learn.xb,learn.yb
# export
def model_summary(model, data, find_all=False, print_mod=False):
    xb,yb = get_batch(data.valid_dl, learn)
    mods = find_modules(model, is_lin_layer) if find_all else model.children()
    f = lambda hook,mod,inp,out: print(f"====\n{mod}\n" if print_mod else "", out.shape)
    with Hooks(mods, f) as hooks: learn.model(xb)
learn = Learner(arch(), data, loss_func, lr=1, cb_funcs=cbfs, opt_func=opt_func)
learn.model = learn.model.cuda()
model_summary(learn.model, data, print_mod=False)
arch = partial(xresnet34, c_out=10)

learn = Learner(arch(), data, loss_func, lr=1, cb_funcs=cbfs, opt_func=opt_func)

learn.fit(1, cbs=[LR_Find(), Recorder()])
learn.recorder.plot(3)
#export
def create_phases(phases):
    phases = listify(phases)
    return phases + [1-sum(phases)]
print(create_phases(0.3))
print(create_phases([0.3,0.2]))

lr = 1e-2
pct_start = 0.5
phases = create_phases(pct_start)
sched_lr  = combine_scheds(phases, cos_1cycle_anneal(lr/10., lr, lr/1e5))
sched_mom = combine_scheds(phases, cos_1cycle_anneal(0.95, 0.85, 0.95))
cbsched = [
    ParamScheduler('lr', sched_lr),
    ParamScheduler('mom', sched_mom)]
learn = Learner(arch(), data, loss_func, lr=lr, cb_funcs=cbfs, opt_func=opt_func)
learn.fit(5, cbs=cbsched)
def cnn_learner(arch, data, loss_func, opt_func, c_in=None, c_out=None,
                lr=1e-2, cuda=True, norm=None, progress=True, mixup=0, xtra_cb=None, **kwargs):
    cbfs = [partial(AvgStatsCallback,accuracy)]+listify(xtra_cb)
    if progress: cbfs.append(ProgressCallback)
    if cuda:     cbfs.append(CudaCallback)
    if norm:     cbfs.append(partial(BatchTransformXCallback, norm))
    if mixup:    cbfs.append(partial(MixUp, mixup))
    arch_args = {}
    if not c_in : c_in  = data.c_in
    if not c_out: c_out = data.c_out
    if c_in:  arch_args['c_in' ]=c_in
    if c_out: arch_args['c_out']=c_out
    return Learner(arch(**arch_args), data, loss_func, opt_func=opt_func, lr=lr, cb_funcs=cbfs, **kwargs)
learn = cnn_learner(xresnet34, data, loss_func, opt_func, norm=norm_imagenette)
learn.fit(5, cbsched)
%load_ext autoreload
%autoreload 2

%matplotlib inline
path = datasets.untar_data(datasets.URLs.IMAGEWOOF_160) 

size = 128
bs = 64

#data block API
tfms = [make_rgb, RandomResizedCrop(size, scale=(0.35,1)), np_to_float, PilRandomFlip()]
val_tfms = [make_rgb, CenterCrop(size), np_to_float]
il = ImageList.from_files(path, tfms=tfms)
sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='val'))
ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())
ll.valid.x.tfms = val_tfms
data = ll.to_databunch(bs, c_in=3, c_out=10, num_workers=8)
len(il)
loss_func = LabelSmoothingCrossEntropy()
opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)
learn = cnn_learner(xresnet18, data, loss_func, opt_func, norm=norm_imagenette)

def sched_1cycle(lr, pct_start=0.3, mom_start=0.95, mom_mid=0.85, mom_end=0.95): #schedular one cycle 
    phases = create_phases(pct_start) #creae our ohases 
    sched_lr  = combine_scheds(phases, cos_1cycle_anneal(lr/10., lr, lr/1e5)) #create learning rate 
    sched_mom = combine_scheds(phases, cos_1cycle_anneal(mom_start, mom_mid, mom_end)) #cetate momentum 
    return [ParamScheduler('lr', sched_lr),
            ParamScheduler('mom', sched_mom)]
lr = 3e-3
pct_start = 0.5
cbsched = sched_1cycle(lr, pct_start)

learn.fit(40, cbsched) #we are gonna use this model to do transfer learning on the data below so we need to save the model 
st = learn.model.state_dict() #so when we save a model we grap the state dictornary 

type(st) #so it created a dictornary 

', '.join(st.keys()) #where the keys is just the layers 
st['10.bias'] #so we can look up fx 10.bias from above and it just retrns the weights 
mdl_path = path/'models' #create somewhere to...
mdl_path.mkdir(exist_ok=True)#... save our model 
torch.save(st, mdl_path/'iw5') #and torch.save will save that dictornary 
pets = datasets.untar_data(datasets.URLs.PETS)
pets.ls()
pets_path = pets/'images'
il = ImageList.from_files(pets_path, tfms=tfms)
il
#so since there isnt a validation set we are gonna create a random splitter
def random_splitter(fn, p_valid): return random.random() < p_valid
random.seed(42)
sd = SplitData.split_by_func(il, partial(random_splitter, p_valid=0.1)) #pass the random splitter to split_by_func

sd#and we are done the validationset is created 
n = il.items[0].name; n# so lets grap one file name 
re.findall(r'^(.*)_\d+.jpg$', n)[0] #and let just get the dogs name 

def pet_labeler(fn): return re.findall(r'^(.*)_\d+.jpg$', fn.name)[0] #find all the dogs and cats names 
proc = CategoryProcessor()

ll = label_by_func(sd, pet_labeler, proc_y=proc)#and label them all

', '.join(proc.vocab) #so this is how it looks like 
ll.valid.x.tfms = val_tfms
c_out = len(proc.vocab)
data = ll.to_databunch(bs, c_in=3, c_out=c_out, num_workers=8)
learn = cnn_learner(xresnet18, data, loss_func, opt_func, norm=norm_imagenette)

learn.fit(5, cbsched) #and now we can train without the transfer and it dose not look good so lets try with transfer learning 
learn = cnn_learner(xresnet18, data, loss_func, opt_func, c_out=10, norm=norm_imagenette) #data is the pets databuch but lets tell it to have 10 channels (c_out) output so
#so 10 activations at the end becouse the data from the model from the transfer model only have 10 dogs labels so we are gonna use that 

st = torch.load(mdl_path/'iw5')#grap our state dictornary we saved above 

m = learn.model
m.load_state_dict(st) #and we load it into our model 
cut = next(i for i,o in enumerate(m.children()) if isinstance(o,nn.AdaptiveAvgPool2d))#so we look through all the children of the model and we try to find the adaptive averge pooling layer 
m_cut = m[:cut] #so lets create a new model that take all up to the adaptive averge pooling layer. so this is the body of the model 

xb,yb = get_batch(data.valid_dl, learn)
pred = m_cut(xb)#so now we give the body a new head 
pred.shape #so we need how many outputs there are from the m_cut as the input to the new model, so we print it and it is 512 

ni = pred.shape[1]
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=1):
        super().__init__()
        self.output_size = sz
        self.ap = nn.AdaptiveAvgPool2d(sz) #do averge pool
        self.mp = nn.AdaptiveMaxPool2d(sz)#do max pool
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1) #and catilate them together 

nh = 40

m_new = nn.Sequential( #so our need model..
    m_cut, AdaptiveConcatPool2d(), Flatten(), #.. contain the whole body with the adaptive pooling, and flatten and linear 
    nn.Linear(ni*2, data.c_out)) #our linear layer needs twice as many inout since we got both averge and max pooling 
learn.model = m_new #lets replace the old model with the new model we just created 
learn.fit(5, cbsched) #and look at that it is much better then before 
#now we just refactor the code above into a function 
def adapt_model(learn, data):
    cut = next(i for i,o in enumerate(learn.model.children())
               if isinstance(o,nn.AdaptiveAvgPool2d))
    m_cut = learn.model[:cut]
    xb,yb = get_batch(data.valid_dl, learn)
    pred = m_cut(xb)
    ni = pred.shape[1]
    m_new = nn.Sequential(
        m_cut, AdaptiveConcatPool2d(), Flatten(),
        nn.Linear(ni*2, data.c_out))
    learn.model = m_new

learn = cnn_learner(xresnet18, data, loss_func, opt_func, c_out=10, norm=norm_imagenette)
learn.model.load_state_dict(torch.load(mdl_path/'iw5'))

adapt_model(learn, data)
#so lets just train the head so we take all the paramters in the body (m_cut) and freeze them like so 
for p in learn.model[0].parameters(): p.requires_grad_(False)
#so now are training just the head and we get 54 which is ifne 
learn.fit(3, sched_1cycle(1e-2, 0.5))
#so we unfreeze the whole body so we get the whole model for training 
for p in learn.model[0].parameters(): p.requires_grad_(True)
#so we train again but now we hit a issue since we get the same acc. as the head trainig. note when something wried is happing it is almost surtant it is bacthnorm. and it is true here to 
#so what happened is that our frozen part of our model #so the inside the head model had a different avg and std then the body so everything just tried to catch up when we unfroze it 
learn.fit(5, cbsched, reset_opt=True)
learn = cnn_learner(xresnet18, data, loss_func, opt_func, c_out=10, norm=norm_imagenette)
learn.model.load_state_dict(torch.load(mdl_path/'iw5'))
adapt_model(learn, data)

def apply_mod(m, f): #apply any function you pass to it recursively to all the children of the module #note pytorch has its own called model.apply(see below when used)
    f(m)
    for l in m.children(): apply_mod(l, f)

def set_grad(m, b): #set gradient 
    if isinstance(m, (nn.Linear,nn.BatchNorm2d)): return #if it s a linear layer or batch norm layer at the middel return, so dont change the gradient otherwise
    if hasattr(m, 'weight'):#if it got weights ...
        for p in m.parameters(): p.requires_grad_(b) #...set required gradient to whatever you asked for and we set it to false below
apply_mod(learn.model, partial(set_grad, b=False)) #so we freeze just the non-batchnorm layers and the last layer 
learn.fit(3, sched_1cycle(1e-2, 0.5))
apply_mod(learn.model, partial(set_grad, b=True)) #unfreeze
learn.fit(5, cbsched, reset_opt=True) #and now we see a mush better result again 
#same thing as the function apply_mod
learn.model.apply(partial(set_grad, b=False));
learn = cnn_learner(xresnet18, data, loss_func, opt_func, c_out=10, norm=norm_imagenette)
learn.model.load_state_dict(torch.load(mdl_path/'iw5'))
adapt_model(learn, data)
def bn_splitter(m):
    def _bn_splitter(l, g1, g2):
        if isinstance(l, nn.BatchNorm2d): g2 += l.parameters() #which will recursely look for for batchnorm layers and put htem intor the seound group(g2)
        elif hasattr(l, 'weight'): g1 += l.parameters() #or anything else with a weight goes into the first group 
        for ll in l.children(): _bn_splitter(ll, g1, g2) #and do it recursely 
        
    g1,g2 = [],[] #create two emty arrays for the two groups of paramters 
    _bn_splitter(m[0], g1, g2) #and its gonna pass the body to _bn_splitter...
    
    g2 += m[1:].parameters() #and also everyting in the secound group will add everything after the head 
    return g1,g2
#set veribles for the 2 groups we split our paramters up into
a,b = bn_splitter(learn.model)
test_eq(len(a)+len(b), len(list(m.parameters()))) #check that the total lenght of a+b is the same as all the paramters in the total model 
Learner.ALL_CBS 
#export
from types import SimpleNamespace
cb_types = SimpleNamespace(**{o:o for o in Learner.ALL_CBS})

cb_types.after_backward
class DebugCallback(Callback): #for debuging
    _order = 999
    def __init__(self, cb_name, f=None): self.cb_name,self.f = cb_name,f
    def __call__(self, cb_name): #overwrite dounder call itself 
        if cb_name==self.cb_name:
            if self.f: self.f(self.run)
            else:      set_trace()
def sched_1cycle(lrs, pct_start=0.3, mom_start=0.95, mom_mid=0.85, mom_end=0.95):
    phases = create_phases(pct_start)
    sched_lr  = [combine_scheds(phases, cos_1cycle_anneal(lr/10., lr, lr/1e5))
                 for lr in lrs]
    sched_mom = combine_scheds(phases, cos_1cycle_anneal(mom_start, mom_mid, mom_end))
    return [ParamScheduler('lr', sched_lr),
            ParamScheduler('mom', sched_mom)]
disc_lr_sched = sched_1cycle([0,3e-2], 0.5) #so no learning rate for the body (0) but a learning rate of 3e-2 for the head and the batch norm
learn = cnn_learner(xresnet18, data, loss_func, opt_func,
                    c_out=10, norm=norm_imagenette, splitter=bn_splitter)

learn.model.load_state_dict(torch.load(mdl_path/'iw5'))
adapt_model(learn, data)
def _print_det(o): 
    print (len(o.opt.param_groups), o.opt.hypers)
    raise CancelTrainException()

learn.fit(1, disc_lr_sched + [DebugCallback(cb_types.after_batch, _print_det)])
learn.fit(3, disc_lr_sched)
disc_lr_sched = sched_1cycle([1e-3,1e-2], 0.3)

learn.fit(5, disc_lr_sched)