# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tokenizers

import torch

MAX_LEN = 128

TRAIN_BATCH_SIZE = 32

VALID_BATCH_SIZE=32

EPOCHS=10

BERT_PATH='bert-base-uncased'

TRAINING_FILE = '/kaggle/input/tweet-sentiment-extraction/train.csv'

SAMPLE_FILE ='/kaggle/input/tweet-sentiment-extraction/sample_submission.csv'

class config:

    MAX_LEN = 128

    TRAIN_BATCH_SIZE = 64

    VALID_BATCH_SIZE = 16

    EPOCHS = 5

    BERT_PATH = "../input/bert-base-uncased/"

    MODEL_PATH = "model.bin"

    #TRAINING_FILE = "../input/tweet-train-folds/train_folds.csv"

    TOKENIZER = tokenizers.BertWordPieceTokenizer(

        f"{BERT_PATH}/vocab.txt", 

        lowercase=True

    )

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



        
config.TOKENIZER.decode([103])
class TweetDataset:

    def __init__(self,tweet,sentiment,selected_text,MAX_LEN,TOKENIZER):

        self.tweet = tweet

        self.sentiment = sentiment

        self.selected_text = selected_text

        self.max_len = MAX_LEN

        self.tokenizer = TOKENIZER

    def __len__(self):

        return len(self.tweet)

    def __getitem__(self,item):

        tweet = self.tweet[item]

        selected_text =self.selected_text[item]

        len_sel_text = len(selected_text)

        idx0 = -1

        idx1 = -1

        for ind in (i for i,e in enumerate(tweet) if e == selected_text[0]):

            if tweet[ind:ind+len_sel_text] == selected_text:

                idx0=ind

                idx1=ind+len_sel_text-1

                break

        char_targets = [0]*len(tweet)

        if idx0 != -1 and idx1 !=-1:

            for j in range(idx0,idx1+1):

                    char_targets[j] = 1

        sentiment_id = {

        'positive': 3893,

        'negative': 4997,

        'neutral': 8699}

        

        tok_tweet =self.tokenizer.encode(tweet)

        tok_tweet_ids = tok_tweet.ids[1:-1]

        tok_tweet_offsets = tok_tweet.offsets[1:-1]

        #print(tok_tweet_ids,' IDS')

        t=True

        targets = []

        for j , (offset1,offset2) in enumerate(tok_tweet_offsets):

            if sum(char_targets[offset1:offset2])>0:

                targets.append(j)

        

        #print(targets,'<<<<<<<')

        #print(non_zero)

        token_type_ids = [0]*len(tok_tweet_ids)

        token_type_ids=[0]*3 + [1]*len(tok_tweet_ids + [102])

        tok_tweet_ids=[101] + [sentiment_id[self.sentiment[item]]] + [102] + tok_tweet_ids + [102] 

        tweet_offsets = [(0, 0)] * 3 + tok_tweet_offsets + [(0, 0)]

        mask = [1]*len(tok_tweet_ids)

        padding_len = self.max_len - len(tok_tweet_ids)



        if padding_len>1:

            ids = tok_tweet_ids+ [0]*padding_len

            mask = mask + [0]*padding_len

            token_type_ids += [0]*(padding_len)

            tweet_offsets = tweet_offsets + ([(0, 0)] * padding_len)

        #print(targets[0],targets[-1],res_targets[(targets[0]+3+1):(targets[-1]+3)])

        #res_targets[(targets[0]+3+1):(targets[-1]+3)]=0

        #print('------')

        return {

            'ids':torch.tensor(ids,dtype=torch.long),                       

            'mask':torch.tensor(mask,dtype=torch.long),

            'token_type_ids':torch.tensor(token_type_ids,dtype=torch.long),

            'padding_len':torch.tensor(padding_len,dtype=torch.long),

            },{'tweet_offsets':torch.tensor(tweet_offsets,dtype=torch.long),

            'orig_selected':self.selected_text[item],

            'sentiment':self.sentiment[item],

            'orig_tweet': self.tweet[item],

            'targets_start':torch.tensor(targets[0]+3,dtype=torch.long),#torch.tensor(targets_start,dtype=torch.float),

            'targets_end':torch.tensor(targets[-1]+3,dtype=torch.long)}
df=pd.read_csv(TRAINING_FILE).dropna().reset_index(drop=True)

dset = TweetDataset(tweet = df.text.values,

                    sentiment = df.sentiment.values,

                    selected_text = df.selected_text.values,

                   MAX_LEN=MAX_LEN,

                   TOKENIZER=config.TOKENIZER)

#dset[12538]
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

ids=skf.get_n_splits(df['text'], df['sentiment'])
df['kfold']=0

for ids,(train_index, test_index) in enumerate(skf.split(df['text'], df['sentiment'])):

    df['kfold'].loc[test_index]=ids
#from transformers import *

import torch.nn as nn

import transformers

import torch

from sklearn import model_selection

from transformers.modeling_bert import BertPreTrainedModel

import scipy
class BERTBaseUncased(transformers.BertPreTrainedModel):

    def __init__(self, conf,drop_sample=5):

        super(BERTBaseUncased, self).__init__(conf)

        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH, config=conf)

        self.drop_out = nn.Dropout(0.5)

        self.drop = nn.Dropout(0.2)

        #self.lstm = nn.LSTM(768,768,bidirectional=True)

        self.l0 = nn.Linear(768 * 2, 2)

        torch.nn.init.normal_(self.l0.weight, std=0.02)

        self.drop_sample=drop_sample

    

    def forward(self, x):



        _,out1,out= self.bert(

            x['ids'],

            attention_mask=x['mask'],

            token_type_ids=x['token_type_ids']

        )

        out = torch.cat((out1.unsqueeze(1).repeat(1,128,1),out[-1]), dim=-1)

        #out,_=self.lstm(self.drop(out[-1].permute(1,0,2)),(self.drop(out1.unsqueeze(0).repeat(2,1,1)),self.drop(out1.unsqueeze(0).repeat(2,1,1))))

        #out=out.permute(1,0,2)

        #================

        res_start=[]

        res_end=[]

        if self.training:

            for i in range(self.drop_sample):

                outt = self.drop_out(out)

                logits = self.l0(outt)



                start_logits, end_logits = logits.split(1, dim=-1)



                start_logits = start_logits.squeeze(-1)

                end_logits = end_logits.squeeze(-1)

                res_start.append(start_logits)

                res_end.append(end_logits)

            return res_start, res_end

        else:

            outt = self.drop_out(out)

            logits = self.l0(outt)



            start_logits, end_logits = logits.split(1, dim=-1)



            start_logits = start_logits.squeeze(-1)

            end_logits = end_logits.squeeze(-1)

            return start_logits,end_logits
np.random.randint(20)
def loss_fn(outputs,targets):

    loss_=0

    if type(outputs[0]) is list:

        for i in range(len(outputs[0])):

            loss_+= nn.CrossEntropyLoss()(outputs[0][i],targets['targets_start']) + nn.CrossEntropyLoss()(outputs[1][i],targets['targets_end'])

        loss_=loss_/len(outputs[0])    

    else:

        loss_=nn.CrossEntropyLoss()(outputs[0],targets['targets_start']) + nn.CrossEntropyLoss()(outputs[1],targets['targets_end'])

    return loss_

def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))

def calculate_jaccard_score(original_tweet, target_string, sentiment_val, idx_start, idx_end, offsets,

    verbose=False):

    if idx_end < idx_start:

        idx_end = idx_start

    

    filtered_output  = ""

    for ix in range(idx_start, idx_end + 1):

        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]

        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:

            filtered_output += " "



    if sentiment_val == "neutral" or len(original_tweet.split()) < 2:

        filtered_output = original_tweet



    jac = jaccard(target_string, filtered_output)

    return jac, filtered_output

def metric(outputs,target):

    tweet,selected_tweet,tweet_sentiment,offsets =target['orig_tweet'],target['orig_selected'],target['sentiment'],target['tweet_offsets'].cpu().numpy()    

    outputs_start,outputs_end = outputs[0].softmax(dim=1).detach().cpu().numpy(),outputs[1].softmax(dim=1).detach().cpu().numpy()

    jac=[]



    for ind in range(len(tweet)):

            jac.append(calculate_jaccard_score(tweet[ind], selected_tweet[ind], tweet_sentiment[ind], outputs_start[ind].argmax(),outputs_end[ind].argmax(), offsets[ind])[0])   

    return np.mean(jac)
from tqdm import  tqdm_notebook

def step(data_loader,model,optimizer,device,scheduler=None,is_train=False,metric=metric):

    model.train() if is_train else model.eval()

    losses = []

    metrics=[]

    tk0 = tqdm_notebook(data_loader,total=len(data_loader))

    for i,d in enumerate(tk0):

        ids = d['ids'].to(device)

        mask = d['mask'].to(device)

        token_type_ids =d['token_type_ids'].to(device)

        targets_start = d['targets_start'].to(device)

        targets_end   = d['targets_end'].to(device)

        optimizer.zero_grad()

        outputs = model(ids,mask,token_type_ids)

        loss = loss_fn(outputs,[targets_start,targets_end])

        if is_train:

            loss.backward()

            optimizer.step()

            if scheduler is not None:

                scheduler.step()

        else:

            metrics.append(metric((d['orig_tweet'],d['orig_selected'],d['sentiment'],outputs[0],outputs[1],d['tweet_offsets'])))

        losses.append(loss.item())  

        tk0.set_postfix(loss=losses[-1])



    return losses,metrics
TRAIN_BATCH_SIZE=2

MAX_LEN=128

EPOCHS = 2

dfx=pd.read_csv(TRAINING_FILE).dropna().reset_index(drop=True)

sample =pd.read_csv(SAMPLE_FILE)

df_train,df_valid = model_selection.train_test_split(dfx,random_state=42,test_size=0.2)

df_train = df_train.reset_index(drop=True)

df_valid = df_valid.reset_index(drop=True)



target_cols = list(sample.drop('textID',axis=1).columns)

train_targets = df_train[target_cols].values

valid_targets = df_valid[target_cols].values



train_dataset = TweetDataset(df_train.text.values,df_train.sentiment.values,df_train.selected_text.values,MAX_LEN,config.TOKENIZER)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=TRAIN_BATCH_SIZE,shuffle=True)



valid_dataset = TweetDataset(df_valid.text.values,df_valid.sentiment.values,df_valid.selected_text.values,MAX_LEN,config.TOKENIZER)

valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=16,shuffle=False)
data = DataBunch(train_loader,valid_loader)

model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)

model_config.output_hidden_states = True

model = BERTBaseUncased(conf=model_config)

param_optimizer = list(model.named_parameters())

no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

optimizer_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},

    ]

optimizer = transformers.AdamW(optimizer_parameters, lr=1e-4/2)#'''
sched = combine_scheds([0.3, 0.7], [sched_cos(1e-4/2,1e-4), sched_cos(1e-4, 1e-4/5)])
'''cbfs = [Recorder,

        partial(AvgStatsCallback,metric),

        CudaCallback,

       ProgressCallback,

       partial(EarlyStopingCallback,1,'model')]

        #partial(ParamScheduler, 'lr', sched)]'''
'''def get_learner(model,opt,data, loss_func,

                cb_funcs=None):

    return Learner(model, data, loss_func, opt,cb_funcs=cb_funcs)



learn = get_learner(model,optimizer,data,loss_fn, cb_funcs=cbfs)'''
#learn.fit(4)
df.kfold.value_counts()


def get_learner(model,opt,data, loss_func,

                cb_funcs=None):

    return Learner(model, data, loss_func, opt,cb_funcs=cb_funcs)

from tqdm import tqdm_notebook

def run_fold(k):

    TRAIN_BATCH_SIZE=32

    MAX_LEN=128

    

    sample =pd.read_csv(SAMPLE_FILE)

    df_train,df_valid = df[df.kfold!=k],df[df.kfold==k]

    df_train = df_train.reset_index(drop=True)

    df_valid = df_valid.reset_index(drop=True)



    target_cols = list(sample.drop('textID',axis=1).columns)

    train_targets = df_train[target_cols].values

    valid_targets = df_valid[target_cols].values



    #tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = TweetDataset(df_train.text.values,df_train.sentiment.values,df_train.selected_text.values,MAX_LEN,config.TOKENIZER)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=TRAIN_BATCH_SIZE,shuffle=True)

    valid_dataset = TweetDataset(df_valid.text.values,df_valid.sentiment.values,df_valid.selected_text.values,MAX_LEN,config.TOKENIZER)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=16,shuffle=False)

    data = DataBunch(train_loader,valid_loader)

    

    model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)

    model_config.output_hidden_states = True

    model = BERTBaseUncased(conf=model_config)

    params = list(model.named_parameters())

    def is_backbone(n):

        return "bert" in n

    lrr=2e-5

    optimizer_grouped_parameters = [

            {"params": [p for n, p in params if is_backbone(n)], "lr": lrr},

            {"params": [p for n, p in params if not is_backbone(n)], "lr":lrr * 100},

        ]



    optimizer = torch.optim.AdamW(

            optimizer_grouped_parameters, lr=lrr, weight_decay=0

        )

    #cbfs = [Recorder,partial(AvgStatsCallback,metric),CudaCallback,ProgressCallback,partial(EarlyStopingCallback,1)]

    #sched = combine_scheds([0.3, 0.7], [sched_cos(1e-4/2,1e-4), sched_cos(1e-4, 1e-4/3)])

    cbfs = [Recorder,

        partial(AvgStatsCallback,metric),

        CudaCallback,

        ProgressCallback,

        #partial(ParamScheduler, 'lr', sched),

        partial(EarlyStopingCallback,1)]

    learn = get_learner(model,optimizer,data,loss_fn, cb_funcs=cbfs)

    

    learn.fit(10)    

    test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

    test['selected_text']=test['text'].apply(lambda x:x)

    dset = TweetDataset(tweet = test.text.values,

                    sentiment = test.sentiment.values,

                    selected_text = test.selected_text.values,

                   MAX_LEN=MAX_LEN,

                   TOKENIZER=config.TOKENIZER)

    res_start=[]

    res_end=[]

    bs=16

    for x,y in tqdm_notebook(torch.utils.data.DataLoader(dset,batch_size=bs,shuffle=False)):

        for key in x.keys():

            x[key]=x[key].cuda()

        output = learn.model(x)

        res_start.append(output[0].detach().cpu().numpy())

        res_end.append(output[1].detach().cpu().numpy())

    return np.concatenate(res_start),np.concatenate(res_end)

start1,end1=run_fold(0)
start2,end2=run_fold(1)
start3,end3=run_fold(2)
start4,end4=run_fold(3)
start = (start1+start2+start3+start4)/4

end = (end1+end2+end3+end4)/4
for x,y in dset:

    print(y)

    break
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

test['selected_text']=test['text'].apply(lambda x:x)

dset = TweetDataset(tweet = test.text.values,

                    sentiment = test.sentiment.values,

                    selected_text = test.selected_text.values,

                   MAX_LEN=MAX_LEN,

                   TOKENIZER=config.TOKENIZER)

final_output=[]

i=0

for x,y in dset:

                ids = x["ids"]

                token_type_ids = x["token_type_ids"]

                mask =x["mask"]

                sentiment = y["sentiment"]

                orig_selected = y["orig_selected"]

                orig_tweet = y["orig_tweet"]

                targets_start = y["targets_start"]

                targets_end = y["targets_end"]

                offsets = y["tweet_offsets"].numpy()

                selected_tweet = orig_selected

                tweet_sentiment = sentiment

                _, output_sentence = calculate_jaccard_score(

                    original_tweet=orig_tweet,

                    target_string=selected_tweet,

                    sentiment_val=tweet_sentiment,

                    idx_start=np.argmax(start[i]),

                    idx_end=np.argmax(end[i]),

                    offsets=offsets

                )

                final_output.append(output_sentence)

                i+=1

                
test['out']=final_output
def post_process(selected):

    return " ".join(set(selected.lower().split()))

sample = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")

sample.loc[:, 'selected_text'] = final_output

sample.selected_text = sample.selected_text.map(post_process)

sample.to_csv("submission.csv", index=False)
sample.head()