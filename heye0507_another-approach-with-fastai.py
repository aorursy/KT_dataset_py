import numpy as np

import pandas as pd

from fastai.callbacks import *

from fastai.callback import *

from fastai.basic_train import *

from fastai.basic_data import *

from fastai.basics import *

from fastai.metrics import *
path = Path('../input')

path.ls()
df_train = pd.read_csv(path/'train.csv')
df_train.head()
y = df_train['label'].values

X = df_train.drop(columns='label').values
from sklearn.model_selection import train_test_split
x_train,x_valid,y_train,y_valid = train_test_split(X,y)

x_train.shape, x_valid.shape,y_train.max(),y_train.min()
x_train,y_train,x_valid,y_valid = map(tensor,(x_train,y_train,x_valid,y_valid))
def normalize_data(train,valid):

    train = train.float()

    valid = valid.float()

    m = train.mean()

    s = train.std()

    return (train - m)/s, (valid - m)/s
x_train, x_valid = normalize_data(x_train,x_valid)

x_train.shape,x_valid.shape
plt.imshow(x_train[0].view(28,28),cmap='gray')

plt.title(str(y_train[0]))
train_ds = TensorDataset(x_train,y_train)

valid_ds = TensorDataset(x_valid,y_valid)
train_dl = DataLoader(

    dataset=train_ds,

    batch_size=64,

    shuffle=True,

    num_workers=2

)

valid_dl = DataLoader(

    dataset=valid_ds,

    batch_size=128,

    num_workers=2

)
data = DataBunch(train_dl,valid_dl)

data.c = len(y_train.unique())
plt.imshow(data.train_ds[0][0].view(28,28),cmap='gray')

plt.title(str(data.train_ds[0][1]))
def flatten(x):

    return x.view(x.shape[0],-1)



class Lambda(nn.Module):

    def __init__(self,func): 

        super().__init__()

        self.func = func

        

    def forward(self,xb): return self.func(xb)



class SubRelu(nn.Module):

    def __init__(self,sub=0.4):

        super().__init__()

        self.sub = sub

    

    def forward(self,xb):

        xb = F.relu(xb)

        xb.sub_(self.sub)

        return xb



def subConv2d(ni,nf,ks=3,stride=2):

    return nn.Sequential(nn.Conv2d(ni,nf,ks,padding=ks//2,stride=stride),SubRelu())



def get_subRelu_model():

    model = nn.Sequential(

        subConv2d(1,8),

        subConv2d(8,16),

        subConv2d(16,32),

        subConv2d(32,32),

        nn.AdaptiveAvgPool2d(1),

        Lambda(flatten),

        nn.Linear(32,10),

    )

    return model
model = get_subRelu_model()

model
def init_model(model):

    for layer in model:

        if isinstance(layer,nn.Sequential):

            nn.init.kaiming_normal_(layer[0].weight)

            layer[0].bias.detach().zero_()
model[0][0].bias
init_model(model)

model[0][0].bias #check the model is initialized
class BatchTransFormXCallback(Callback):

    _order = 2 

    def __init__(self,tfm):

        #super().__init__(learn)

        self.tfm = tfm

    

    def on_batch_begin(self,**kwargs):

        xb = self.tfm(kwargs['last_input'])

        return {'last_input': xb.float()}
# wrap learner creation step, for easy re-use during build up this notebook

def get_learner(model):

    opt_func = optim.SGD

    loss_func = nn.CrossEntropyLoss()

    return Learner(data,model.cuda(),opt_func=opt_func,loss_func=loss_func,metrics=accuracy)
learn = get_learner(model)
learn.callbacks.append(BatchTransFormXCallback(lambda x: x.view(-1,1,28,28)))
learn.callback_fns.append(ActivationStats) #fastai build in hook to grab layer activation stats
learn.split(lambda m: m[4])
#double check the model is split

learn.layer_groups
learn.lr_find()

learn.recorder.plot()
#apply cos sched and discriminative lrs

learn.fit_one_cycle(8,slice(1e-1,4e-1),pct_start=0.3)
means = learn.activation_stats.stats[0]

for i in range(4):

    plt.plot(means[i][:800])

plt.legend(range(4))
std = learn.activation_stats.stats[1]

for i in range(4):

    plt.plot(std[i][:800])

plt.legend(range(4))
learn.recorder.plot_lr()
learn.recorder.plot_losses()
class BatchNorm_layer(nn.Module):

    def __init__(self,nf,mom=0.1,eps=1e-6):

        super().__init__()

        self.nf = nf

        self.mom = mom

        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(nf,1,1))

        self.beta = nn.Parameter(torch.zeros(nf,1,1))

        self.register_buffer('vars',torch.ones(1,nf,1,1))

        self.register_buffer('means',torch.zeros(1,nf,1,1))

    

    def batch_norm(self,xb):

        m = xb.mean(dim=(0,2,3),keepdim=True)

        #var = xb.var(dim=(0,2,3),keepdim=True)

        var = xb.detach().cpu().numpy() #kaggle torch.var only takes int for dim, not tuple

        var = var.var((0,2,3),keepdims=True)

        var = torch.from_numpy(var).cuda()

        self.means.lerp_(m,self.mom)

        self.vars.lerp_(var,self.mom)

        return m,var

    

    def forward(self,xb):

        if self.training:

            with torch.no_grad(): m,v = self.batch_norm(xb)

        else:

            m,v = self.means,self.vars

        xb = (xb - m) / (v+self.eps).sqrt()

        return self.gamma * xb + self.beta
class RunningBatchNorm(nn.Module):

    def __init__(self, nf, mom=0.1, eps=1e-5):

        super().__init__()

        self.mom, self.eps = mom, eps

        self.mults = nn.Parameter(torch.ones (nf,1,1))

        self.adds  = nn.Parameter(torch.zeros(nf,1,1))

        self.register_buffer('sums', torch.zeros(1,nf,1,1))

        self.register_buffer('sqrs', torch.zeros(1,nf,1,1))

        self.register_buffer('count', tensor(0.))

        self.register_buffer('factor', tensor(0.))

        self.register_buffer('offset', tensor(0.))

        self.batch = 0

        

    def update_stats(self, x):

        bs,nc,*_ = x.shape

        self.sums.detach_()

        self.sqrs.detach_()

        dims = (0,2,3)

        s    = x    .sum(dims, keepdim=True)

        ss   = (x*x).sum(dims, keepdim=True)

        c    = s.new_tensor(x.numel()/nc)

        mom1 = s.new_tensor(1 - (1-self.mom)/math.sqrt(bs-1))

        self.sums .lerp_(s , mom1)

        self.sqrs .lerp_(ss, mom1)

        self.count.lerp_(c , mom1)

        self.batch += bs

        means = self.sums/self.count

        varns = (self.sqrs/self.count).sub_(means*means)

        if bool(self.batch < 20): varns.clamp_min_(0.01)

        self.factor = self.mults / (varns+self.eps).sqrt()

        self.offset = self.adds - means*self.factor

        

    def forward(self, x):

        if self.training: self.update_stats(x)

        return x*self.factor + self.offset
def Conv2d_BN(ni,nf,ks=3,stride=2,BN=True):

    if BN:

        return nn.Sequential(nn.Conv2d(ni,nf,ks,padding=ks//2,stride=stride),SubRelu(),BatchNorm_layer(nf))

    else:

        return nn.Sequential(nn.Conv2d(ni,nf,ks,padding=ks//2,stride=stride),SubRelu(),RunningBatchNorm(nf))



def get_batchNorm_model(BN=True):

    model = nn.Sequential(

        #Lambda(lambda x: x.view(-1,1,28,28).float()),

        Conv2d_BN(1,8,BN=BN),

        Conv2d_BN(8,16,BN=BN),

        Conv2d_BN(16,32,BN=BN),

        Conv2d_BN(32,32,BN=BN),

        nn.AdaptiveAvgPool2d(1),

        Lambda(flatten),

        nn.Linear(32,10),

    )

    return model
bn_model = get_batchNorm_model()

opt_func = optim.SGD

loss_func = nn.CrossEntropyLoss

learn = Learner(data,bn_model.cuda(),opt_func=opt_func,loss_func=loss_func(),metrics=accuracy)

cb = BatchTransFormXCallback(tfm=lambda x: x.view(-1,1,28,28))

learn.callbacks.append(cb)

learn.callback_fns.append(ActivationStats)

init_model(learn.model)
learn.split(lambda m: m[4])
learn.fit_one_cycle(12,slice(1e-1,2.),pct_start=0.3)
learn.recorder.plot_lr()
learn.recorder.plot_losses()
means = learn.activation_stats.stats[0]

for i in range(4):

    plt.plot(means[i][:800])

plt.legend(range(4))
std = learn.activation_stats.stats[1]

for i in range(4):

    plt.plot(std[i][:800])

plt.legend(range(4))
bn_model = get_batchNorm_model(BN=False)

opt_func = optim.SGD

loss_func = nn.CrossEntropyLoss

learn = Learner(data,bn_model.cuda(),opt_func=opt_func,loss_func=loss_func(),metrics=accuracy)

cb = BatchTransFormXCallback(tfm=lambda x: x.view(-1,1,28,28))

learn.callbacks.append(cb)

learn.callback_fns.append(ActivationStats)

init_model(learn.model)
learn.split(lambda m: m[4])
learn.fit_one_cycle(12,slice(1e-1,2.),pct_start=0.3)
df_test = pd.read_csv(path/'test.csv')

df_test = df_test.values
test_train = torch.from_numpy(X) #just to get m and std

test = torch.from_numpy(df_test)

test.shape
_,test = normalize_data(test_train,test)
dummy_y = torch.ones(test.shape[0])

dummy_y.shape
test_ds = TensorDataset(test,dummy_y)
test_dl = DataLoader(

    dataset = test_ds,

    batch_size = 64,

    num_workers = 2

)
learn.model.eval()
def get_preds(test_dl,model):

    preds = []

    model.cpu()

    for dl in test_dl:

        pred_batch = torch.argmax(model(dl[0].view(-1,1,28,28)),dim=1)

        preds += pred_batch.detach().tolist()

    return preds
preds = get_preds(test_dl,learn.model)

len(preds)
final = pd.Series(preds,name='Label')

submission = pd.concat([pd.Series(range(1,28001),name='ImageId'),final],axis=1)

submission.to_csv('fastai-pytorch-0.99.csv',index=False)