import pandas as pd

import numpy as np

import torch



from torchvision import models, transforms

from torch import nn

from torch.nn import functional as F

from matplotlib import pyplot as plt



from PIL import Image



from pathlib import Path

Path.ls = lambda x: list(x.iterdir())

import re

from functools import partial



from typing import *

import math

from IPython.core.debugger import set_trace

from sklearn.metrics import roc_auc_score

import time

from fastprogress.fastprogress import format_time

def listify(o):

    if o is None: return []

    if isinstance(o, list): return o

    if isinstance(o, str): return [o]

    if isinstance(o, Iterable): return list(o)

    return [o]



def compose(x, funcs, *args, order_key='_order', **kwargs):

    key = lambda o: getattr(o, order_key,0)

    for f in sorted(listify(funcs), key=key): x = f(x,**kwargs)

    return x
def annealer(f):

    def _inner(start,end): return partial(f,start,end)

    return _inner



@annealer

def sched_lin(start,end,pos): return start+pos*(end-start)



@annealer

def sched_cos(start,end,pos): return start + (1+math.cos(math.pi*(1-pos))) * (end-start) / 2

@annealer

def sched_no(start,end,pos): return start

@annealer

def sched_exp(start, end, pos): return start * (end/start) ** pos



def cos_1cycle_anneal(start, high, end):

    return [sched_cos(start,high), sched_cos(high,end)]



def combine_scheds(pcts, scheds):

    assert sum(pcts)==1.

    pcts = torch.tensor([0]+listify(pcts))

    assert torch.all(pcts >= 0)

    pcts = torch.cumsum(pcts, 0)

    def _inner(pos):

        idx = (pos >= pcts).nonzero().max()

        actual_pos = (pos-pcts[idx])/ (pcts[idx+1] - pcts[idx])

        return scheds[idx](actual_pos)

    return _inner


class CancelBatchException(Exception): pass

class CancelTrainException(Exception): pass

class CancelEpochException(Exception): pass





_camel_re1 = re.compile('(.)([A-Z][a-z]+)')

_camel_re2 = re.compile('([a-z0-9])([A-Z])')



def camel2snake(name):

    s1 = re.sub(_camel_re1, r'\1_\2',name)

    return re.sub(_camel_re2,r'\1_\2',s1).lower()



class Callback():

    _order = 0

    def set_runner(self,run):self.run=run

    def __getattr__(self,k): return getattr(self.run, k)

    @property

    def name(self):

        name = re.sub(r'Callback$','',self.__class__.__name__)

        return camel2snake(name or 'callback')



    def __call__(self,cb_name):

        f = getattr(self, cb_name, None)

        if f and f(): return True

        return False
###########

# AvgStats

###########

class AvgStats():

    def __init__(self,metrics,in_train): self.metrics,self.in_train = listify(metrics),in_train



    def reset(self):

        self.tot_loss, self.count = 0., 0

        self.tot_mets = [0.] * len(self.metrics)



    @property

    def all_stats(self): return [self.tot_loss.item()] + self.tot_mets

    @property

    def avg_stats(self): return [o/self.count for o in self.all_stats]



    def __repr__(self):

        if not self.count: return ""

        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"



    def accumulate(self,run):

        bn = run.xb.shape[0]

        self.tot_loss += run.loss *bn

        self.count += bn

        for i,m in enumerate(self.metrics):

            self.tot_mets[i]+=m(run.yb.cpu(),run.pred.detach().cpu())*bn



########################

#BatchTransformXCallback

########################

class BatchTransformXCallback(Callback):

    def __init__(self,tfm): self.tfm = tfm

    

    def begin_batch(self): self.run.xb = self.tfm(self.xb)

#######################

#CudaCallback

#######################

class CudaCallback(Callback):

    def begin_fit(self): self.model.cuda()

    

    def begin_batch(self): self.run.xb, self.run.yb = self.xb.cuda(), self.yb.cuda()

##################

#AvgStatsCallback

##################

class AvgStatsCallback(Callback):

    def __init__(self, metrics):

        self.train_stats, self.valid_stats = AvgStats(metrics, True), AvgStats(metrics, False)

        

    def begin_fit(self):

        met_names = ['loss'] + [m.__name__ for m in self.train_stats.metrics]

        names = ['epoch'] + [f'train_{n}' for n in met_names] + [

            f'valid_{n}' for n in met_names] + ['time']

        self.logger(names)

    

    def begin_epoch(self):

        self.train_stats.reset()

        self.valid_stats.reset()

        self.start_time = time.time()

    

    def after_loss(self):

        set_trace()

        stats = self.train_stats if self.in_train else self.valid_stats

        with torch.no_grad(): stats.accumulate(self.run)

            

    def after_epoch(self):

        stats = [str(self.epoch)]

        for o in [self.train_stats, self.valid_stats]:

            stats += [f'{v:6f}' for v in o.avg_stats]

        stats += [format_time(time.time() - self.start_time)]

        self.logger(stats)



##################

#LogStatsCallback

##################

class LogStatsCallback(Callback):

    def __init__(self, metrics):

        self.train_stats, self.valid_stats = AccumulateStats(metrics, True), AccumulateStats(metrics, False)

        

    def begin_fit(self):

        met_names = ['loss'] + [m.__name__ for m in self.train_stats.metrics]

        names = ['epoch'] + [f'train_{n}' for n in met_names] + [

            f'valid_{n}' for n in met_names] + ['time']

        self.logger(names)

    

    def begin_epoch(self):

        self.train_stats.reset()

        self.valid_stats.reset()

        self.start_time = time.time()

    

    def after_loss(self):

        stats = self.train_stats if self.in_train else self.valid_stats

        with torch.no_grad(): stats.accumulate(self.run)

            

    def after_epoch(self):

        stats = [str(self.epoch)]

        for o in [self.train_stats, self.valid_stats]:

            stats += [f'{v:6f}' for v in o.avg_stats]

        stats += [format_time(time.time() - self.start_time)]

        self.logger(stats)

        

#############

#Recorder

#############



class Recorder(Callback):

    def __init__(self,pnames):

        self.pnames = listify(pnames)

        self.hps = []



    def begin_fit(self):

        for pname in self.pnames:

            self.hps.append([])

        self.losses = []



    def after_batch(self):

        if not self.in_train: return

        for hp,pname in zip(self.hps,self.pnames):

            hp.append(self.opt.hypers[-1][pname])



        self.losses.append(self.loss.detach().cpu())



    def plot_lr  (self): plt.plot(self.hps[self.pnames.index('lr')])

    def plot_mom (self): plt.plot(self.hps[self.pnames.index('mom')])

    def plot_loss(self): plt.plot(self.losses)



    def plot(self,skip_last=0,pgid=-1):

        lrs = self.hps[self.pnames.index('lr')]

        losses = [o.item() for o in self.losses]

        n = len(self.losses) - skip_last

        plt.xscale('log')

        plt.plot(lrs[:n],losses[:n])





######################

#TrainEvalCallback

######################



class TrainEvalCallback(Callback):

    def begin_fit(self):

        self.run.n_epochs=0.

        self.run.n_iter =0.

    

    def begin_epoch(self):

        self.run.n_epochs = self.epoch

        self.model.train()

        self.run.in_train = True

        

    def begin_validate(self):

        self.model.eval()

        self.run.in_train=False

    

    def after_batch(self):

        if not self.in_train: return

        self.run.n_epochs += 1./self.iters

        self.run.n_iter += 1



######################

#LR_Find

######################



class LR_Find(Callback):

    _order = 1

    def __init__(self,max_iter=100,min_lr=1e-6,max_lr=10):

        self.max_iter,self.min_lr,self.max_lr = max_iter,min_lr,max_lr

        self.best_loss = 1e9

        

    def begin_batch(self):

        if not self.in_train: return

        pos = self.n_iter/self.max_iter

        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos

        for pg in self.opt.hypers: pg['lr'] = lr

    

    def after_step(self):

        if self.n_iter >= self.max_iter or self.loss > self.best_loss*10:

            raise CancelTrainException()

        if self.loss < self.best_loss: self.best_loss = self.loss



##################

#ParamScheduler

################## 



class ParamScheduler(Callback):

    _order=1

    def __init__(self, pname, sched_func):

        self.pname,self.sched_func = pname, sched_func

        

    def set_param(self):

        for h in self.opt.hypers:

            h[self.pname] = self.sched_func(self.n_epochs/self.epochs)

    

    def begin_batch(self):

        if self.in_train: self.set_param()



##################

#ProgressCallback

##################

from fastprogress.fastprogress import master_bar, progress_bar,ProgressBar



class ProgressCallback(Callback):

    _order=-1

    def begin_fit(self):

        self.mbar = master_bar(range(self.epochs))

        self.mbar.on_iter_begin()

        self.run.logger = partial(self.mbar.write, table=True)

        

    def after_fit(self): self.mbar.on_iter_end()

    def after_batch(self): self.pb.update(self.iter)

    def begin_epoch   (self): self.set_pb()

    def begin_validate(self): self.set_pb()

    

    def set_pb(self):

        self.pb = progress_bar(self.dl, parent=self.mbar)

        self.mbar.update(self.epoch)
def sgd_step(p, lr, **kwargs):

    p.data.add_(p.grad.data, alpha=-lr)

    return p



def weight_decay(p, lr, wd, **kwargs):

    p.data.mul_(1-lr*wd)

    return p

weight_decay._defaults = dict(wd=0.)



def maybe_update(os, dest, f):

    for o in os:

        for k,v in f(o).items():

            if k not in dest: dest[k] = v

def get_defaults(d): return getattr(d, '_defaults', {})



class Optimizer():

    def __init__(self, params, steppers, **defaults):

        self.steppers = listify(steppers)

        maybe_update(self.steppers,defaults, get_defaults)

        self.param_groups = list(params)

        if not isinstance(self.param_groups[0], list): self.param_groups = [self.param_groups]

        self.hypers = [{**defaults} for p in self.param_groups]

       

    def grad_params(self):

        return [(p,hyper) for pg,hyper in zip(self.param_groups,self.hypers) 

               for p in pg if p.grad is not None]

    

    def zero_grad(self):

        for p,hyper in self.grad_params():

            p.grad.detach_()

            p.grad.zero_()

            

    def step(self):

        for p,hyper in self.grad_params(): compose(p, self.steppers, **hyper)



sgd_opt = partial(Optimizer, steppers=[weight_decay,sgd_step])



class StatefulOptimizer(Optimizer):

    def __init__(self, params, steppers, stats=None, **defaults):

        self.stats = listify(stats)

        maybe_update(self.stats, defaults, get_defaults)

        super().__init__(params, steppers, **defaults)

        self.state = {}

    

    def step(self):

        for p,hyper in self.grad_params():

            if p not in self.state:

                #Create a state for p and call all the statistics to initialize it.

                self.state[p] = {}

                maybe_update(self.stats, self.state[p], lambda o: o.init_state(p))

            state = self.state[p]

            for stat in self.stats: state = stat.update(p, state, **hyper)

            compose(p, self.steppers, **state, **hyper)

            self.state[p] = state



class Stat():

    _defaults = {}

    def init_state(self,p): raise NotImplementedError

    def update(self,p,state,**kwargs): raise NotImplementedError



class AverageGrad(Stat):

    _defaults = dict(mom=0.9)

    

    def __init__(self, dampening:bool=False): self.dampening=dampening

    def init_state(self,p): return {'grad_avg': torch.zeros_like(p.grad.data)}

    def update(self,p,state,mom,**kwargs):

        state['mom_damp'] = 1-mom if self.dampening else 1.

        state['grad_avg'].mul_(mom).add_(state['mom_damp'],p.grad.data)

        return state



def momentum_step(p, lr, grad_avg, **kwargs):

    p.data.add_(grad_avg,alpha=-lr)

    return p



sgd_mom_opt = partial(StatefulOptimizer, steppers=[momentum_step, weight_decay], 

                     stats=AverageGrad(), wd=0.01)





class AverageSqrGrad(Stat):

    _defaults = dict(sqr_mom=0.99)

    

    def __init__(self, dampening:bool=True): self.dampening=dampening

    def init_state(self,p): return {'sqr_avg': torch.zeros_like(p.grad.data)}

    def update(self, p, state, sqr_mom, **kwargs):

        state['sqr_damp'] = 1-sqr_mom if self.dampening else 1.

        state['sqr_avg'].mul_(sqr_mom).addcmul_(state['sqr_damp'], p.grad.data, p.grad.data)

        return state



class StepCount(Stat):

    def init_state(self, p): return {'step': 0}

    def update(self, p, state, **kwargs):

        state['step'] += 1

        return state



def debias(mom, damp,step): return damp * (1-mom**step) / (1-mom)



def adam_step(p,lr,mom,mom_damp,step,sqr_mom, sqr_damp, grad_avg, sqr_avg, eps, **kwargs):

    debias1 = debias(mom,     mom_damp, step)

    debias2 = debias(sqr_mom, sqr_damp, step)

    p.data.addcdiv_(-lr/debias1, grad_avg, (sqr_avg/debias2).sqrt()+eps)

    return p

adam_step._defaults = dict(eps=1e-5)



def adam_opt(xtra_step=None, **kwargs):

    return partial(StatefulOptimizer, steppers=[adam_step, weight_decay]+listify(xtra_step),

                  stats=[AverageGrad(dampening=True), AverageSqrGrad(), StepCount()],**kwargs)
def param_getter(m): return m.parameters()



#Exceptions



class Learner():

    def __init__(self, model, data, loss_func, opt_func, lr=1e-2, splitter=param_getter,

                cbs=None, cb_funcs=None):

        self.model, self.data, self.loss_func, self.opt_func, self.lr, self.splitter = model,data,loss_func,opt_func, lr, splitter

        self.in_train, self.logger, self.opt = False,print,None

        

        self.cbs = []

        self.add_cb(TrainEvalCallback())

        self.add_cbs(cbs)

        self.add_cbs(cbf() for cbf in listify(cb_funcs))

        

    def add_cbs(self,cbs):

        for cb in listify(cbs): self.add_cb(cb)

            

    def add_cb(self,cb):

        cb.set_runner(self)

        setattr(self,cb.name,cb)

        self.cbs.append(cb)

        

    def remove_cbs(self,cbs):

        for cb in listify(cbs): self.cbs.remove(cb)

            

    def one_batch(self,i,xb,yb):

        try: 

            self.iter = i

            self.xb,self.yb = xb,yb;                        self('begin_batch')

            self.pred = self.model(self.xb);                self('after_pred')

            self.loss = self.loss_func(self.pred,self.yb);  self('after_loss')

            if not self.in_train: return

            self.loss.backward();                           self('after_backward')

            self.opt.step();                                self('after_step')

            self.opt.zero_grad()

        except CancelBatchException:                        self('after_cancel_batch')

        finally:                                            self('after_batch')

            

    def all_batches(self):

        self.iters = len(self.dl)

        try:

            for i, (xb,yb) in enumerate(self.dl): self.one_batch(i,xb,yb)

        except CancelEpochException: self('after_cancel_epoch')

    

    def do_begin_fit(self, epochs):

        self.epochs, self.loss = epochs, torch.tensor(0.)

        self('begin_fit')

    

    def do_begin_epoch(self,epoch):

        self.epoch, self.dl = epoch, self.data.train_dl

        return self('begin_epoch')

    

    def fit(self,epochs,cbs=None,reset_opt=False):

        self.add_cbs(cbs)

        

        if reset_opt or not self.opt: self.opt = self.opt_func(self.splitter(self.model), lr=self.lr)

            

        try:

            self.do_begin_fit(epochs)

            for epoch in range(epochs):

                if not self.do_begin_epoch(epoch): self.all_batches()

                    

                with torch.no_grad():

                    self.dl = self.data.valid_dl

                    if not self('begin_validate'): self.all_batches()

                self('after_epoch')

        except CancelTrainException: self('after_cancel_train')

        finally:

            self('after_fit')

            self.remove_cbs(cbs)

            

    ALL_CBS = {'begin_batch','after_pred','after_loss','after_backward','after_step',

              'after_cancel_batch', 'after_batch','after_cancel_epoch','begin_fit','begin_epoch',

              'begin_validate','after_epoch','after_cancel_train','after_fit'}

    

    def __call__(self,cb_name):

        res= False

        assert cb_name in self.ALL_CBS

        for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name) and res

        return res
path = Path('/kaggle/input/siim-isic-melanoma-classification/')

path.ls()
df = pd.read_csv(path/'train.csv')

df.head()
from sklearn.model_selection import StratifiedKFold



kf = StratifiedKFold(n_splits=5)



df['valid'] = 0



for train_idx, valid_idx in kf.split(df, df['target']):

    df.loc[valid_idx, 'valid'] = 1

    break



df['valid'] = df['valid'].astype('int')
path_train = Path('/kaggle/input/siic-isic-224x224-images')

path_train.ls()
class Dataset:

    def __init__(self,path,df,folder='train',extension='.png',is_test=False,size=224):

        self.path = Path(path)

        self.df = df

        self.folder = folder

        self.extension = extension

        self.is_test = is_test

        self.size = size if isinstance(size,tuple) else (size,size)

        self.tfms = transforms.Compose([transforms.Resize(self.size),

            transforms.ToTensor(),transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])

        

    def __len__(self): return len(self.df)

    def __getitem__(self,idx):

        row = self.df.loc[idx,:]

        file_name = row['image_name']+self.extension

        

        img = Image.open(self.path/self.folder/file_name)

        img = self.tfms(img)

        

        if not self.is_test:

            target = torch.tensor(row['target'],dtype=torch.long)

            return img,target

        return img
class DataBunch():

    def __init__(self, train_dl, valid_dl):

        self.train_dl, self.valid_dl = train_dl, valid_dl

        

    @property

    def train_ds(self): return self.train_dl.dataset

    @property

    def valid_ds(self): return self.valid_dl.dataset
class Model(nn.Module):

    def __init__(self, pretrained=False):

        super().__init__()

        self.model = models.densenet121(pretrained=pretrained).features

        self.out = nn.Linear(1024,1)

        

    def forward(self, x):

        x = F.adaptive_avg_pool2d(self.model(x),1)

        x = self.out(x.view(x.shape[0],-1))

        return x        
train_df = df.loc[df['valid']==0, :].reset_index(drop=True)

valid_df = df.loc[df['valid']==1, :].reset_index(drop=True)
# train_ds = Dataset(path_train, train_df,size=128)

# valid_ds = Dataset(path_train, valid_df,size=128)



# train_dl = torch.utils.data.DataLoader(

#     train_ds, 

#     batch_size=256,

#     shuffle=True,

#     num_workers=4

# )

# valid_dl = torch.utils.data.DataLoader(

#     valid_ds,

#     batch_size=256,

#     shuffle=False,

#     num_workers=4

# )
# data = DataBunch(train_dl, valid_dl)
def loss_func(outputs, targets,**kwargs):

    loss = nn.BCEWithLogitsLoss(**kwargs)(outputs.view(-1), targets.float())

    return loss
sched_lr = combine_scheds([0.3,0.7],cos_1cycle_anneal(4e-5,4e-4,4e-6))

sched_mom = combine_scheds([0.3,0.7],cos_1cycle_anneal(0.94,0.85,0.94))
def lin_comb (v1,v2,beta): return beta*v1 + (1-beta)*v2
class NoneReduce():

    def __init__(self, loss_func):

        self.loss_func, self.old_red = loss_func, None

    

    def __enter__(self):

        if hasattr(self.loss_func, 'reduction'):

            self.old_red = getattr(self.loss_func, 'reduction')

            setattr(self.loss_func,'reduction', 'none')

            return self.loss_func

        else: return partial(self.loss_func, reduction='none')

    

    def __exit__(self, type, value, traceback):

        if self.old_red is not None: setattr(self.loss_func, 'reduction', self.old_red)
from torch.distributions.beta import Beta



def unsqueeze(input, dims):

    for dim in listify(dims): input = torch.unsqueeze(input,dim)

    return input



def reduce_loss(loss, reduction='mean'):

    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss



class Mixup(Callback):

    _order = 90 #Runs after normalization and cuda

    def __init__(self, alpha:float=0.4): self.distrib = Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    

    def begin_fit(self): self.old_loss_func, self.run.loss_func = self.run.loss_func, self.loss_func

        

    def begin_batch(self):

        if not self.in_train: return #Only mixup things during training

        L = self.distrib.sample((self.yb.size(0),)).squeeze().to(self.xb.device)

        L = torch.stack([L, 1-L], 1)

        self.L = unsqueeze(L.max(1)[0], (1,2,3))

        shuffle = torch.randperm(self.yb.size(0)).to(self.xb.device)

        xb1, self.yb1 = self.xb[shuffle], self.yb[shuffle]

        self.run.xb = lin_comb(self.xb, xb1, self.L)

    

    def after_fit(self): self.run.loss_func = self.old_loss_func

        

    def loss_func(self, pred, yb):

        if not self.in_train: return self.old_loss_func(pred,yb)

        with NoneReduce(self.old_loss_func) as loss_func:

            loss1 = loss_func(pred,yb)

            loss2 = loss_func(pred, self.yb1)

        loss = lin_comb(loss1, loss2, self.L)

        return reduce_loss(loss, getattr(self.old_loss_func, 'reduction', 'mean'))
cbfs=[partial(Recorder,['lr','mom']),CudaCallback,ProgressCallback]
def get_learner(data, loss_func, opt_func, lr=1e-3, cbfs=None, pretrained=False):

    model = Model(pretrained)

    learner = Learner(model, data, loss_func, opt_func=opt_func,lr=1e-3,cb_funcs=cbfs)

    return learner
# learner = get_learner(data, loss_func, adam_opt(), cbfs=cbfs,pretrained=True)

# learner.fit(1,cbs=[LR_Find(),Mixup()])

# learner.recorder.plot()
learner = None

import gc

gc.collect()
##################

#AccumulateStats

##################

class AccumulateStats():

    def __init__(self, metrics, in_train): self.metrics, self.in_train = listify(metrics), in_train

    

    def reset(self):

        self.loss_list = []

        self.met_list = [[]]*len(self.metrics)

        self.preds_list = []

        self.yb_list = []

    

    def accumulate(self, run):

        self.loss_list.append(run.loss)

        self.yb_list.append(run.yb.cpu())

        for i,m in enumerate(self.metrics):

            self.met_list[i].append(run.pred.detach().cpu())



    def get_stats(self):

        loss = torch.tensor(self.loss_list).mean()

        metrics=[]

        for i,m in enumerate(self.metrics):

            metrics.append(m(torch.cat(self.yb_list),torch.cat(self.met_list[i])))

        

        return [loss]+metrics

    @property

    def avg_stats(self): return self.get_stats()
train_ds = Dataset(path_train, train_df,size=224)

valid_ds = Dataset(path_train, valid_df,size=224)



train_dl = torch.utils.data.DataLoader(

    train_ds, 

    batch_size=96,

    shuffle=True

)

valid_dl = torch.utils.data.DataLoader(

    valid_ds,

    batch_size=128,

    shuffle=False

)
data = DataBunch(train_dl,valid_dl)
learner = get_learner(data, loss_func, adam_opt(),cbfs=cbfs,pretrained=True)
learner.fit(5, cbs=[ParamScheduler('lr',sched_lr),ParamScheduler('mom',sched_mom),LogStatsCallback(roc_auc_score),Mixup()])
# learner.fit(1,cbs=[LR_Find()])

# learner.recorder.plot()
test_df = pd.read_csv(path/'test.csv')

test_df.head()
test_ds = Dataset(path_train, test_df, folder='test',is_test=True)
test_dl = torch.utils.data.DataLoader(

    test_ds,

    batch_size=128,

    shuffle=False

)
from tqdm import tqdm
def predict(dataloader, model, device):

    

    predictions = []

    model.eval()

    model.to(device)

    with torch.no_grad():

        for x in tqdm(dataloader, total=len(dataloader)):

            x = x.to(device)

            preds = model(x)

            predictions.append(preds.detach().cpu())



    return torch.cat(predictions)
predictions =predict(test_dl,learner.model, 'cuda')
submissions = pd.read_csv(path/'sample_submission.csv')

submissions.head()
submissions['target'] = predictions.numpy()

submissions.to_csv('submission.csv',index=False)





submissions['target'] = torch.sigmoid(predictions).numpy()

submissions.to_csv('submission_sigmoid.csv',index=False)