%load_ext autoreload

%autoreload 2



%matplotlib inline
import json

import os

import torch

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Dataset

from PIL import Image

import torchvision.models as models

from torch import nn

import torch.nn.functional as F

from torch import optim

import torchvision

import pandas as pd
dic = json.loads(open('/kaggle/input/hackathon-blossom-flower-classification/cat_to_name.json').read())
classes = list(dic.values())
loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])  



root = r'/kaggle/input/hackathon-blossom-flower-classification/flower_data/flower_data/train'

train_image_files = []

for x in os.scandir(root):

    for y in os.scandir(x):

        if y.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG"):

            train_image_files.append(y.path)



class train_dataset(Dataset):

    def __init__(self, train_image_files, dic, classes):

        self.train_image_files = train_image_files

        self.dic = dic

        self.classes = classes

    def __getitem__(self, i):

        image_data = Image.open(self.train_image_files[i]).resize((256, 256))

        image_data = loader(image_data)

        image_data.to(torch.float)

        return image_data, self.classes.index(self.dic[self.train_image_files[i].split('/')[7]])

    def __len__(self):

        return len(self.train_image_files)



root = r'/kaggle/input/hackathon-blossom-flower-classification/flower_data/flower_data/valid/'

valid_image_files = []

for x in os.scandir(root):

    for y in os.scandir(x):

        if y.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG"):

            valid_image_files.append(y.path)



class valid_dataset(Dataset):

    def __init__(self, valid_image_files, dic, classes):

        self.valid_image_files = valid_image_files

        self.dic = dic

        self.classes = classes

    def __getitem__(self, i):

        image_data = Image.open(self.valid_image_files[i]).resize((256, 256))

        image_data = loader(image_data)

        image_data.to(torch.float)

        return image_data, self.classes.index(self.dic[self.valid_image_files[i].split('/')[7]])

    def __len__(self):

        return len(self.valid_image_files)
train = train_dataset(train_image_files, dic, classes)

valid = valid_dataset(valid_image_files, dic, classes)
train_dl = DataLoader(train, batch_size = 64, shuffle = True, drop_last=True)

valid_dl = DataLoader(valid, batch_size = 64, shuffle=False)
model = models.resnet34(pretrained=True)

fc_features = model.fc.in_features

model.fc = nn.Linear(fc_features, 102)
from typing import *



def listify(o):

    if o is None: return []

    if isinstance(o, list): return o

    if isinstance(o, str): return [o]

    if isinstance(o, Iterable): return list(o)

    return [o]

def accuracy(out, targ): return (torch.argmax(out, dim=1)==targ).float().mean()
import re



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

    #定义Exception异常类

    def __call__(self, cb_name):         

        f = getattr(self, cb_name, None)

        if f and f(): return True

        return False

class CancelTrainException(Exception): pass     

class CancelEpochException(Exception): pass     

class CancelBatchException(Exception): pass    
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
class Runner():

    def __init__(self, cbs=None, cb_funcs=None):

        cbs = listify(cbs)                      

        for cbf in listify(cb_funcs):           

            cb = cbf()                          

            setattr(self, cb.name, cb)          

            cbs.append(cb)

        self.stop,self.cbs = False,[TrainEvalCallback()]+cbs

    @property

    def opt(self):       return self.learn.opt

    @property

    def model(self):     return self.learn.model

    @property

    def loss_func(self): return self.learn.loss_func

    @property

    def data(self):      return self.learn.data



    def one_batch(self, xb, yb):

        try:

            self.xb,self.yb = xb,yb

            self('begin_batch')

            self.pred = self.model(self.xb)

            self('after_pred')

            self.loss = self.loss_func(self.pred, self.yb)

            self('after_loss')

            if not self.in_train: return

            self.loss.backward()

            self('after_backward')

            self.opt.step()

            self('after_step')

            self.opt.zero_grad()

        except CancelBatchException: self('after_cancel_batch') #

        finally: self('after_batch')



    def all_batches(self, dl):

        self.iters = len(dl)

        try:

            for xb,yb in dl: self.one_batch(xb, yb)

        except CancelEpochException: self('after_cancel_epoch')



    def fit(self, epochs, learn):

        self.epochs,self.learn,self.loss = epochs,learn,torch.tensor(0.)



        try:

            for cb in self.cbs: cb.set_runner(self)

            self('begin_fit')

            for epoch in range(epochs):

                self.epoch = epoch

                if not self('begin_epoch'): self.all_batches(self.data.train_dl)



                with torch.no_grad(): 

                    if not self('begin_validate'): self.all_batches(self.data.valid_dl)

                self('after_epoch')

            

        except CancelTrainException: self('after_cancel_train')

        finally:

            self('after_fit')

            self.learn = None



    def __call__(self, cb_name):

        for cb in sorted(self.cbs, key=lambda x: x._order):

            f = getattr(cb, cb_name, None)                 

            if f and f(): return True                     

        return False
class AvgStats():

    def __init__(self, metrics, in_train): self.metrics,self.in_train = listify(metrics),in_train

    

    def reset(self):

        self.tot_loss,self.count = 0.,0 

        self.tot_mets = [0.] * len(self.metrics) 

        

    @property

    def all_stats(self): return [self.tot_loss.item()] + self.tot_mets 

    @property

    def avg_stats(self): return [o/self.count for o in self.all_stats] 

    

    def __repr__(self): 

        if not self.count: return ""

        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}" 



    def accumulate(self, run):

        bn = run.xb.shape[0]

        self.tot_loss += run.loss * bn 

        self.count += bn              

        for i,m in enumerate(self.metrics):

            self.tot_mets[i] += m(run.pred, run.yb) * bn 



class AvgStatsCallback(Callback):

    _order = 1

    def __init__(self, metrics):

        self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False) #train数据和valid数据分别定义一个AvgStats

        

    def begin_epoch(self):

        self.train_stats.reset()

        self.valid_stats.reset()

        

    def after_loss(self):

        stats = self.train_stats if self.in_train else self.valid_stats

        with torch.no_grad(): stats.accumulate(self.run)

    

    def after_epoch(self):

        print(self.train_stats)

        print(self.valid_stats)
class DataBunch():

    def __init__(self, train_dl, valid_dl, c=None):

        self.train_dl,self.valid_dl,self.c = train_dl,valid_dl,c

        

    @property

    def train_ds(self): return self.train_dl.dataset

        

    @property

    def valid_ds(self): return self.valid_dl.dataset
class Learner():

    def __init__(self, model, opt, loss_func, data):

        self.model,self.opt,self.loss_func,self.data = model,opt,loss_func,data
device = torch.device('cuda',0)

torch.cuda.set_device(device)
class CudaCallback(Callback): 

    def begin_fit(self): self.model.cuda() 

    def begin_batch(self): self.run.xb,self.run.yb = self.xb.cuda(),self.yb.cuda()
data = DataBunch(train_dl, valid_dl)

learn = Learner(model, optim.SGD(model.parameters(), lr = 0.1), F.cross_entropy, data)

stats = AvgStatsCallback([accuracy])

run = Runner(cbs=stats, cb_funcs =  CudaCallback)
run.fit(3, learn)
torch.save(model, 'model.pkl')
root = r'/kaggle/input/hackathon-blossom-flower-classification/test set/test set/'

test_image_files = []

for x in os.scandir(root):

    if x.name.endswith(".jpg") or x.name.endswith(".jpeg") or x.name.endswith(".JPG"):

        test_image_files.append(x.path)
test_name = []

test_label = []

for i in range(len(test_image_files)):

    image_data = Image.open(test_image_files[i]).convert("RGB").resize((256, 256))

    image_data = loader(image_data)

    image_data = image_data.to(torch.float).unsqueeze(0)

    pred = classes[int(model(image_data.cuda()).argmax())]

    test_name.append(test_image_files[i].split('/')[-1]) 

    test_label.append(pred)
ans = pd.DataFrame([test_name,test_label]).rename({0:'image_name',1:'predict_label'}).T
ans
ans.to_csv('pred.csv')