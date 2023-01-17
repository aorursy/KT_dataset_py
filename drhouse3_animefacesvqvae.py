# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        break

    break

    

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch.nn as nn

from torch.nn import init

import functools

import albumentations as A



from torchvision import transforms#.RandomResizedCrop(size

from torch.utils.data import Dataset 

from PIL import Image

import matplotlib.pyplot as plt

import torch

import torch.nn.functional as F

import cv2

class DataSet(Dataset):

    def __init__(self,dataset):

        self.dataset = dataset

        self.faces = transforms.Compose([transforms.Resize((64,64)),

                               transforms.RandomAffine(5),

                               transforms.ToTensor()])

    def __len__(self):

        return len(self.dataset)

    def get_img(self,path):       

        tiles = cv2.imread(path)[:,:,::-1]

        return tiles

    def trans(self,img):

        

        transforms = A.Compose([A.Resize(64,64),A.Normalize(mean=[0.485, 0.456, 0.406],

                             std=[0.229, 0.224, 0.225])])

        return transforms(image=img)['image']

    def __getitem__(self,item):

        path=self.dataset.loc[item]['path']

        img= self.get_img(path)

        img = self.trans(img)

        img=img.transpose(2,0,1)

        return  {'X':torch.tensor(img,dtype=torch.float)},{

            'y':torch.tensor(img,dtype=torch.float)}
dataset=pd.DataFrame()

dataset['path']=os.listdir('/kaggle/input/ffhq-face-data-set/thumbnails128x128/')[:21552]

dataset['path']='/kaggle/input/ffhq-face-data-set/thumbnails128x128/'+dataset['path']

df = pd.DataFrame()

df['path']=os.listdir('/kaggle/input/anime-faces/data/')

df['path']='/kaggle/input/anime-faces/data/'+df['path']

dataset=pd.concat([df,dataset]).reset_index(drop=True)
ds= DataSet(dataset)

plt.imshow(ds[22321][0]['X'].permute(1,2,0))
import math

from fastprogress import master_bar, progress_bar

from functools import partial

from fastprogress.fastprogress import format_time

import re

from typing import *

def param_getter(m): return m.parameters()

def listify(o):

    if o is None : return []

    if isinstance(o,list): return o

    if isinstance(o,str): return [o]

    if isinstance(o,Iterable): return list(o)

    return [o]

class DataBunch():

    def __init__(self, train_dl, valid_dl, c=None):

        self.train_dl,self.valid_dl,self.c = train_dl,valid_dl,c

        

    @property

    def train_ds(self): return self.train_dl.dataset

        

    @property

    def valid_ds(self): return self.valid_dl.dataset

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')

_camel_re2 = re.compile('([a-z0-9])([A-Z])')

def camel2snake(name):

    s1 = re.sub(_camel_re1, r'\1_\2', name)

    return re.sub(_camel_re2, r'\1_\2', s1).lower()



class Callback():

    _order=0

    def set_runner(self, run): self.run=run

    def __getattr__(self, k): return getattr(self.run, k)

    

    @property

    def name(self):

        name = re.sub(r'Callback$', '', self.__class__.__name__)

        return camel2snake(name or 'callback')

    

    def __call__(self, cb_name):

        f = getattr(self, cb_name, None)

        if f and f(): return True

        return False

class AvgStatsCallback(Callback):

    def __init__(self, metrics):

        self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)

        

    def begin_epoch(self):

        self.train_stats.reset()

        self.valid_stats.reset()

        

    def after_loss(self):

        #if not self.in_train:

            stats =  self.valid_stats if not self.in_train else self.train_stats

            with torch.no_grad(): stats.accumulate(self.run)

    

    def after_epoch(self):

        print(self.train_stats)

        print(self.valid_stats)

        

class Recorder(Callback):

    def begin_fit(self):

        self.lrs = [[] for _ in self.opt.param_groups]

        self.losses = []



    def after_batch(self):

        if not self.in_train: return

        for pg,lr in zip(self.opt.param_groups,self.lrs): lr.append(pg['lr'])

        self.losses.append(self.loss.detach().cpu())        



    def plot_lr  (self, pgid=-1): plt.plot(self.lrs[pgid])

    def plot_loss(self, skip_last=0): plt.plot(self.losses[:len(self.losses)-skip_last])

        

    def plot(self, skip_last=0, pgid=-1):

        losses = [o.item() for o in self.losses]

        lrs    = self.lrs[pgid]

        n = len(losses)-skip_last

        plt.xscale('log')

        plt.plot(lrs[:n], losses[:n])



class ParamScheduler(Callback):

    _order=1

    def __init__(self, pname, sched_funcs): self.pname,self.sched_funcs = pname,sched_funcs

        

    def begin_fit(self):

        if not isinstance(self.sched_funcs, (list,tuple)):

            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)



    def set_param(self):

        assert len(self.opt.param_groups)==len(self.sched_funcs)

        for pg,f in zip(self.opt.param_groups,self.sched_funcs):

            pg[self.pname] = f(self.n_epochs/self.epochs)

            

    def begin_batch(self): 

        if self.in_train: self.set_param()





class LR_Find(Callback):

    _order=1

    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):

        self.max_iter,self.min_lr,self.max_lr = max_iter,min_lr,max_lr

        self.best_loss = 1e9

        

    def begin_batch(self): 

        if not self.in_train: return

        pos = self.n_iter/self.max_iter

        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos

        for pg in self.opt.param_groups: pg['lr'] = lr

            

    def after_step(self):

        if self.n_iter>=self.max_iter or self.loss>self.best_loss*10:

            raise CancelTrainException()

        if self.loss < self.best_loss: self.best_loss = self.loss

class TrainEvalCallback(Callback):

    def begin_fit(self):

        self.run.n_epochs=0.

        self.run.n_iter=0

    

    def after_batch(self):

        if not self.in_train: return

        self.run.n_epochs += 1./self.iters

        self.run.n_iter   += 1

        

    def begin_epoch(self):

        self.run.n_epochs=self.epoch

        self.model.train()

        self.run.in_train=True



    def begin_validate(self):

        self.model.eval()

        self.run.in_train=False



class CancelTrainException(Exception): pass

class CancelEpochException(Exception): pass

class CancelBatchException(Exception): pass



class Learner():

    def __init__(self, model, data, loss_func, optimizer, lr=1e-2, splitter=param_getter,

                 cbs=None, cb_funcs=None):

        self.model,self.data,self.loss_func,self.lr,self.splitter = model,data,loss_func,lr,splitter

        self.in_train,self.logger,self.opt = False,print,optimizer

        

        # NB: Things marked "NEW" are covered in lesson 12

        # NEW: avoid need for set_runner

        self.cbs = []

        self.add_cb(TrainEvalCallback())

        self.add_cbs(cbs)

        self.add_cbs(cbf() for cbf in listify(cb_funcs))



    def add_cbs(self, cbs):

        for cb in listify(cbs): self.add_cb(cb)

            

    def add_cb(self, cb):

        cb.set_runner(self)

        setattr(self, cb.name, cb)

        self.cbs.append(cb)



    def remove_cbs(self, cbs):

        for cb in listify(cbs): self.cbs.remove(cb)

            

    def one_batch(self, i, xb, yb):

        try:

            self.iter = i

            self.xb,self.yb = xb,yb;                        self('begin_batch')

            self.pred = self.model(self.xb);                self('after_pred')

            self.loss = self.loss_func(self.pred, self.yb); self('after_loss')

            if not self.in_train: return

            self.loss.backward();                           self('after_backward')

            self.opt.step();                                self('after_step')

            self.opt.zero_grad()

        except CancelBatchException:                        self('after_cancel_batch')

        finally:                                            self('after_batch')



    def all_batches(self):

        self.iters = len(self.dl)

        try:

            for i,(xb,yb) in enumerate(self.dl): self.one_batch(i, xb, yb)

        except CancelEpochException: self('after_cancel_epoch')



    def do_begin_fit(self, epochs):

        self.epochs,self.loss = epochs,torch.tensor(0.)

        self('begin_fit')



    def do_begin_epoch(self, epoch):

        self.epoch,self.dl = epoch,self.data.train_dl

        return self('begin_epoch')



    def fit(self, epochs, cbs=None, reset_opt=False):

        # NEW: pass callbacks to fit() and have them removed when done

        self.add_cbs(cbs)

        # NEW: create optimizer on fit(), optionally replacing existing

        #if reset_opt or not self.opt: self.opt = self.opt_func(self.splitter(self.model), lr=self.lr)

            

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



    ALL_CBS = {'begin_batch', 'after_pred', 'after_loss', 'after_backward', 'after_step',

        'after_cancel_batch', 'after_batch', 'after_cancel_epoch', 'begin_fit',

        'begin_epoch', 'begin_validate', 'after_epoch',

        'after_cancel_train', 'after_fit'}

    

    def __call__(self, cb_name):

        res = False

        assert cb_name in self.ALL_CBS

        for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name) and res

        return res

def annealer(f):

    def _inner(start,end): return partial(f,start,end)

    return _inner

@annealer

def sched_lin(start,end,pos): return start + pos*(end-start)

@annealer

def sched_cos(start,end,pos):return start+(1+math.cos(math.pi*(1-pos)))*(end-start)/2

@annealer

def sched_no(start,end,pos): return start

@annealer

def sched_exp(start,end,pos): return start*(end/start)**pos



def cos_1cycle_anneal(start,high,end):

    return [sched_cos(start,high),sched_cos(high,end)]

def combine_scheds(pcts, scheds):

    assert sum(pcts) == 1.

    pcts = torch.tensor([0] + listify(pcts))

    assert torch.all(pcts >= 0)

    pcts = torch.cumsum(pcts, 0)

    def _inner(pos):

        idx = (pos >= pcts).nonzero().max()

        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])

        return scheds[idx](actual_pos)

    return _inner

class CudaCallback(Callback):

    def begin_fit(self): self.model.cuda()

    def begin_batch(self): 

        if type(self.run.xb) is dict:

            for key in self.run.xb.keys():

                self.run.xb[key]=self.run.xb[key].cuda()

        if type(self.run.yb) is dict:

            for key in self.run.yb.keys():

                if type(self.run.yb[key]) is not list:

                    self.run.yb[key]=self.run.yb[key].cuda()

class AvgStats():

    def __init__(self,metrics,in_train): self.metrics,self.in_train = listify(metrics),in_train

    def  reset(self):

        self.tot_loss,self.count =0.,0.

        self.tot_mets = [0.]*len(self.metrics)

    @property

    def all_stats(self):return [self.tot_loss.item()]+ self.tot_mets

    @property

    def avg_stats(self): return [o/self.count for o in self.all_stats]



    def __repr__(self):

        if not self.count: return ''

        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self,run):

        bn = run.xb[list(run.xb.keys())[0]].shape[0]

        self.tot_loss+=run.loss*bn

        self.count+=bn

        if not self.in_train:

            for i,m in enumerate(self.metrics):

                self.tot_mets[i]+= m(run.pred,run.yb)*bn

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

    def after_loss(self): self.pb.comment='loss= ' +str(self.loss.item())[:5]+' lr '+str(self.opt.param_groups[0]['lr'])[:9]



    def set_pb(self):

        self.pb = progress_bar(self.dl, parent=self.mbar)

        self.mbar.update(self.epoch)

class EarlyStopingCallback(Callback):

    _order=-1

    def __init__(self,iters=1,path='bert.pth'):

        self.iters=iters

    

        self.bad_metrics =iters 

        self.path=path

        '''if path: 

            print('>>>>')

            torch.save(self.model.state_dict(),path) 

            print('END')'''

    def begin_fit(self):

        self.best_metric=[0]

    def after_epoch(self):

        mean_metric=self.avg_stats.valid_stats.avg_stats[1]

        if mean_metric>self.best_metric:

            self.bad_metrics=self.iters

            self.best_metric=mean_metric

            if self.path:

                print('Saving..... ',mean_metric)

                torch.save(self.run.model.state_dict(),self.path) 

        else:

            self.bad_metrics-=1

            if self.bad_metrics==0:

                self.run.model.load_state_dict(torch.load(self.path))

                raise CancelTrainException()

 
class VQEmbedding(nn.Module):

    def __init__(self, K, D):

        super().__init__()

        self.embedding = nn.Embedding(K, D)

        self.embedding.weight.data.uniform_(-1./K, 1./K)



    def forward(self, z_e_x):

        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()

        latents = vq(z_e_x_, self.embedding.weight)

        return latents



    def straight_through(self, z_e_x):

        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()

        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())

        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()



        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,

            dim=0, index=indices)

        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)

        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()



        return z_q_x, z_q_x_bar





class ResBlock(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.block = nn.Sequential(

            nn.ReLU(True),

            nn.Conv2d(dim, dim, 3, 1, 1),

            nn.BatchNorm2d(dim),

            nn.ReLU(True),

            nn.Conv2d(dim, dim, 1),

            nn.BatchNorm2d(dim)

        )



    def forward(self, x):

        return x + self.block(x)

import torch

from torch.autograd import Function



class VectorQuantization(Function):

    @staticmethod

    def forward(ctx, inputs, codebook):

        with torch.no_grad():

            embedding_size = codebook.size(1)

            inputs_size = inputs.size()

            inputs_flatten = inputs.view(-1, embedding_size)



            codebook_sqr = torch.sum(codebook ** 2, dim=1)

            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)



            # Compute the distances to the codebook

            distances = torch.addmm(codebook_sqr + inputs_sqr,

                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)



            _, indices_flatten = torch.min(distances, dim=1)

            indices = indices_flatten.view(*inputs_size[:-1])

            ctx.mark_non_differentiable(indices)



            return indices



    @staticmethod

    def backward(ctx, grad_output):

        raise RuntimeError('Trying to call `.grad()` on graph containing '

            '`VectorQuantization`. The function `VectorQuantization` '

            'is not differentiable. Use `VectorQuantizationStraightThrough` '

            'if you want a straight-through estimator of the gradient.')



class VectorQuantizationStraightThrough(Function):

    @staticmethod

    def forward(ctx, inputs, codebook):

        indices = vq(inputs, codebook)

        indices_flatten = indices.view(-1)

        ctx.save_for_backward(indices_flatten, codebook)

        ctx.mark_non_differentiable(indices_flatten)



        codes_flatten = torch.index_select(codebook, dim=0,

            index=indices_flatten)

        codes = codes_flatten.view_as(inputs)



        return (codes, indices_flatten)



    @staticmethod

    def backward(ctx, grad_output, grad_indices):

        grad_inputs, grad_codebook = None, None



        if ctx.needs_input_grad[0]:

            # Straight-through estimator

            grad_inputs = grad_output.clone()

        if ctx.needs_input_grad[1]:

            # Gradient wrt. the codebook

            indices, codebook = ctx.saved_tensors

            embedding_size = codebook.size(1)



            grad_output_flatten = (grad_output.contiguous()

                                              .view(-1, embedding_size))

            grad_codebook = torch.zeros_like(codebook)

            grad_codebook.index_add_(0, indices, grad_output_flatten)



        return (grad_inputs, grad_codebook)



vq = VectorQuantization.apply

vq_st = VectorQuantizationStraightThrough.apply
def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:

        try:

            nn.init.xavier_uniform_(m.weight.data)

            m.bias.data.fill_(0)

        except AttributeError:

            print("Skipping initialization of ", classname)

class VectorQuantizedVAE(nn.Module):

    def __init__(self, input_dim, dim, K=512):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(input_dim, dim, 4, 2, 1),

            nn.BatchNorm2d(dim),

            nn.ReLU(True),

            nn.Conv2d(dim, dim, 4, 2, 1),

            ResBlock(dim),

            ResBlock(dim),

        )



        self.codebook = VQEmbedding(K, dim)



        self.decoder = nn.Sequential(

            ResBlock(dim),

            ResBlock(dim),

            nn.ReLU(True),

            nn.ConvTranspose2d(dim, dim, 4, 2, 1),

            nn.BatchNorm2d(dim),

            nn.ReLU(True),

            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),

            nn.Tanh()

        )



        self.apply(weights_init)



    def encode(self, x):

        z_e_x = self.encoder(x)

        latents = self.codebook(z_e_x)

        return latents



    def decode(self, latents):

        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)

        x_tilde = self.decoder(z_q_x)

        return x_tilde



    def forward(self, x):

        x=x['X']

        z_e_x = self.encoder(x)

        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)

        x_tilde = self.decoder(z_q_x_st)

        return x_tilde, z_e_x, z_q_x

def loss_fn(outputs,targets):

    loss_classes = F.mse_loss(outputs[0],targets['y']) +F.mse_loss(outputs[1], outputs[2].detach())+ F.mse_loss(outputs[2], outputs[1].detach())

    return loss_classes

   

def metric(outputs,target):

    output = outputs[0].detach().cpu().numpy()

    y=target['y'].detach().cpu().numpy()

    return np.array([((output-y)**2).mean()])
dataset.drop(16840,inplace=True)

dataset=dataset.reset_index(drop=True)
from sklearn.model_selection import train_test_split

train,valid=train_test_split(dataset,test_size=0.2)



train_loader=torch.utils.data.DataLoader(DataSet(train.reset_index(drop=True)),batch_size=32,drop_last=True,shuffle=True)

valid_loader=torch.utils.data.DataLoader(DataSet(valid.reset_index(drop=True)),batch_size=64)

data = DataBunch(train_loader,valid_loader)
def get_learner(model,opt,data, loss_func,

                cb_funcs=None):

    return Learner(model, data, loss_func, opt,cb_funcs=cb_funcs)
model=VectorQuantizedVAE(3,64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

cbfs = [Recorder,

        TrainEvalCallback,

        partial(AvgStatsCallback,metric),

        CudaCallback,

        ProgressCallback,

       partial(EarlyStopingCallback,5)]

learn = get_learner(model,optimizer,data,loss_fn, cb_funcs=cbfs)
learn.fit(25)
for x,y in train_loader:

    break
x['X']=x['X'].to(torch.device('cuda'))
frr=learn.model.encode(x['X'])
plt.imshow(x['X'][0].permute(1,2,0).cpu())
rr=learn.model.decode((frr-dif/5).to(torch.long))
rr
dif = (frr[1]-frr[0])
(dif/5).to(torch.int)
dif.dtype
plt.imshow(rr[2].permute(1,2,0).detach().cpu())
dataset['ttt']=dataset['path'].apply(lambda x:x[-4:]=='.png')
dataset[dataset['ttt']==False]