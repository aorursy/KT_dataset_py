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
!pip install torchcontrib

import numpy as np

import pandas as pd

import os

import warnings

import random

import torch 

from torch import nn

import torch.optim as optim

from sklearn.model_selection import StratifiedKFold

import tokenizers

from transformers import RobertaModel, RobertaConfig

from torchcontrib.optim import SWA

def seed_everything(seed_value):

    random.seed(seed_value)

    np.random.seed(seed_value)

    torch.manual_seed(seed_value)

    os.environ['PYTHONHASHSEED'] = str(seed_value)

    

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = True



seed = 42

seed_everything(seed)
MAX_LEN = 128

TRAIN_BATCH_SIZE = 32

VALID_BATCH_SIZE=32

EPOCHS=10

BERT_PATH='bert-base-uncased'

TRAINING_FILE = '/kaggle/input/tweet-sentiment-extraction/train.csv'

SAMPLE_FILE ='/kaggle/input/tweet-sentiment-extraction/sample_submission.csv'

class config:

    MAX_LEN = 128

    TRAIN_BATCH_SIZE = 32

    VALID_BATCH_SIZE = 16

    EPOCHS = 5

    BERT_PATH = "../input/roberta-base"

    #TRAINING_FILE = "../input/tweet-train-folds/train_folds.csv"

    '''TOKENIZER = tokenizers.BertWordPieceTokenizer(

        f"{BERT_PATH}/vocab.txt", 

        lowercase=True

    )'''

    TOKENIZER = tokenizers.ByteLevelBPETokenizer(

    vocab_file=f"{BERT_PATH}/vocab.json", 

    merges_file=f"{BERT_PATH}/merges.txt", 

    lowercase=True,

    add_prefix_space=True

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

        
from albumentations.core.transforms_interface import DualTransform, BasicTransform

class NLPTransform(BasicTransform):

    """ Transform for nlp task."""

    LANGS = {

        'en': 'english',

        'it': 'italian', 

        'fr': 'french', 

        'es': 'spanish',

        'tr': 'turkish', 

        'ru': 'russian',

        'pt': 'portuguese'

    }



    @property

    def targets(self):

        return {"data": self.apply}

    

    def update_params(self, params, **kwargs):

        if hasattr(self, "interpolation"):

            params["interpolation"] = self.interpolation

        if hasattr(self, "fill_value"):

            params["fill_value"] = self.fill_value

        return params



    def get_sentences(self, text, lang='en'):

        return sent_tokenize(text, self.LANGS.get(lang, 'english'))

class ExcludeNumbersTransform(NLPTransform):

    """ exclude any numbers """

    def __init__(self, always_apply=False, p=0.5):

        super(ExcludeNumbersTransform, self).__init__(always_apply, p)



    def apply(self, data, **params):

        text,sent, lang = data

        text = re.sub(r'[0-9]', '', text)

        text = re.sub(r'\s+', ' ', text)

        

        sent = re.sub(r'[0-9]', '', sent)

        sent = re.sub(r'\s+', ' ', sent)

        return text,sent, lang

class ExcludeHashtagsTransform(NLPTransform):

    """ Exclude any hashtags with # """

    def __init__(self, always_apply=False, p=0.5):

        super(ExcludeHashtagsTransform, self).__init__(always_apply, p)



    def apply(self, data, **params):

        text,sent, lang = data

        text = re.sub(r'#[\S]+\b', '', text)

        text = re.sub(r'\s+', ' ', text)

        

        sent = re.sub(r'#[\S]+\b', '', sent)

        sent = re.sub(r'\s+', ' ', sent)

        

        return text,sent, lang

class ExcludeUsersMentionedTransform(NLPTransform):

    """ Exclude @users """

    def __init__(self, always_apply=False, p=0.5):

        super(ExcludeUsersMentionedTransform, self).__init__(always_apply, p)



    def apply(self, data, **params):

        text, sent,lang = data

        text = re.sub(r'@[\S]+\b', '', text)

        text = re.sub(r'\s+', ' ', text)

        

        sent = re.sub(r'@[\S]+\b', '', sent)

        sent = re.sub(r'\s+', ' ', sent)

        

        return text, sent,lang

class ExcludeUrlsTransform(NLPTransform):

    """ Exclude urls """

    def __init__(self, always_apply=False, p=0.5):

        super(ExcludeUrlsTransform, self).__init__(always_apply, p)



    def apply(self, data, **params):

        text,sent, lang = data

        text = re.sub(r'https?\S+', '', text)

        text = re.sub(r'\s+', ' ', text)

        

        sent = re.sub(r'https?\S+', '', sent)

        sent = re.sub(r'\s+', ' ', sent)

        return text,sent, lang
import albumentations



def get_train_transforms(is_train):

    if is_train:

        return albumentations.Compose([

            ExcludeNumbersTransform(p=0.0),

            ExcludeHashtagsTransform(p=0.0),

            ExcludeUsersMentionedTransform(p=0.0),

            ExcludeUrlsTransform(p=0.0),

        ])

    else:

        return albumentations.Compose([

        ExcludeNumbersTransform(p=0),

        ExcludeHashtagsTransform(p=0),

        ExcludeUsersMentionedTransform(p=0),

        ExcludeUrlsTransform(p=0),

    ])
np.random.randint(2)


class TweetDataset:

    def process_data(self,tweet, selected_text, sentiment, tokenizer, max_len):

        if self.is_train and np.random.rand()>0.2:

            new_df = self.selected_text[self.sentiment==sentiment]

            ids=np.random.randint(len(new_df))

            tweet=tweet[:tweet.index(selected_text)]+' '+new_df[ids]+" "+tweet[(tweet.index(selected_text)+len(selected_text)):]

            selected_text=new_df[ids]

        tweet = " " + " ".join(str(tweet).split())

        selected_text = " " + " ".join(str(selected_text).split())



        len_st = len(selected_text) - 1

        idx0 = None

        idx1 = None

        #print(tweet,selected_text)

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):

            if " " + tweet[ind: ind+len_st] == selected_text:

                idx0 = ind

                idx1 = ind + len_st - 1

                break



        char_targets = [0] * len(tweet)

        if idx0 != None and idx1 != None:

            for ct in range(idx0, idx1 + 1):

                char_targets[ct] = 1



        tok_tweet = tokenizer.encode(tweet)

        input_ids_orig = tok_tweet.ids

        tweet_offsets = tok_tweet.offsets



        target_idx = []

        for j, (offset1, offset2) in enumerate(tweet_offsets):

            if sum(char_targets[offset1: offset2]) > 0:

                target_idx.append(j)

        #print(target_idx)



        targets_start = target_idx[0]

        targets_end = target_idx[-1]



        sentiment_id = {

            'positive': 1313,

            'negative': 2430,

            'neutral': 7974

        }



        input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]

        token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)

        mask = [1] * len(token_type_ids)

        tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]

        targets_start += 4

        targets_end += 4



        padding_length = max_len - len(input_ids)

        if padding_length > 0:

            input_ids = input_ids + ([1] * padding_length)

            mask = mask + ([0] * padding_length)

            token_type_ids = token_type_ids + ([0] * padding_length)

            tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

        return {'ids': torch.tensor(input_ids, dtype=torch.long),

            'mask': torch.tensor(mask, dtype=torch.long),

            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)},{'targets_start': torch.tensor(targets_start, dtype=torch.long),

            'targets_end': torch.tensor(targets_end, dtype=torch.long),

            'orig_tweet': tweet,

            'orig_selected': selected_text,

            'sentiment': sentiment,

            'tweet_offsets': torch.tensor(tweet_offsets, dtype=torch.long)}

    def __init__(self,tweet,sentiment,selected_text,MAX_LEN,TOKENIZER,is_train):

        self.tweet = tweet

        self.sentiment = sentiment

        self.selected_text = selected_text

        self.max_len = MAX_LEN

        self.tokenizer = TOKENIZER

        self.is_train=is_train

        self.transforms=get_train_transforms(is_train)

    def __len__(self):

        return len(self.tweet)

    def __getitem__(self,item):

        sent = self.tweet[item]

        selected_text=self.selected_text[item]

        if self.sentiment[item]!='neutral':

            sent,selected_text,_=self.transforms(data=(sent,selected_text, 'en'))['data']

        return  self.process_data(

            sent, 

            selected_text, 

            self.sentiment[item],

            self.tokenizer,

            self.max_len

        )
df=pd.read_csv(TRAINING_FILE).dropna().reset_index(drop=True)





#-------

df.loc[2940]['text']=df.loc[2940]['text'][1:][:2]+" "+df.loc[2940]['text'][1:][2:]

df.loc[2940]['selected_text']=df.loc[2940]['selected_text'][1:][:2]+" "+df.loc[2940]['selected_text'][1:][2:]

df.loc[8444]['text']=df.loc[8444]['text'][1:2]+' '+ df.loc[8444]['text'][2:6]+' '+df.loc[8444]['text'][6:8]+' '+df.loc[8444]['text'][8:10]+' '+df.loc[8444]['text'][10:16]+" "+df.loc[8444]['text'][16:18]+' '+df.loc[8444]['text'][18:]

df.loc[8444]['selected_text']=df.loc[8444]['selected_text'][1:2]+' '+ df.loc[8444]['selected_text'][2:6]+' '+df.loc[8444]['selected_text'][6:8]+' '+df.loc[8444]['selected_text'][8:10]+' '+df.loc[8444]['selected_text'][10:16]+" "+df.loc[8444]['selected_text'][16:18]



df.loc[8718]['text']= df.loc[8718]['text'].replace('#itsucks','it sucks')

df.loc[8718]['selected_text']=df.loc[8718]['selected_text'].replace('#itsucks','it sucks')





#-----------

df.loc[10531]['text']= df.loc[10531]['text'].replace('#fail','fail')

df.loc[10531]['selected_text']=df.loc[10531]['selected_text'].replace('#fail','fail')



df.loc[10576]['text']=df.loc[10576]['text'].replace('#liesboystell','lies boys tell')

df.loc[10576]['selected_text']=df.loc[10576]['selected_text'].replace('liesboystell','lies boys tell')



df.loc[11744]['text']=df.loc[11744]['text'].replace('#shortcakefail','shortcake fail')

df.loc[11744]['selected_text']='shortcake fail'



df.loc[14838]['text'] = df.loc[14838]['text'].replace('was boring but had to eat nonetheless',' was boring but had to eat nonetheless' )



df.loc[14934]['text']=df.loc[14934]['text'].replace('#sad',' sad')

df.loc[14934]['selected_text']='sad'



df.loc[15565]['text'] = df.loc[15565]['text'][1:]



df.loc[17299]['text']=df.loc[17299]['text'].replace('#fail',' fail')

df.loc[17299]['selected_text']='fail'



df.loc[17418]['text']=df.loc[17418]['text'].replace('#itsucks',' it sucks')

df.loc[17418]['selected_text']='it sucks'

ids=18516

df.loc[ids]['text']=df.loc[ids]['text'].replace('#yourock',' you rock')

df.loc[ids]['selected_text']='you rock'

ids=18626

df.loc[ids]['text']=df.loc[ids]['text'].replace('#HappyMothersDay',' Happy Mothers Day')

df.loc[ids]['selected_text']='Happy Mothers Day'



ids=21559

df.loc[ids]['text']=df.loc[ids]['text'].replace('#FAIL',' fail')

df.loc[ids]['selected_text']='fail'



ids=21954

df.loc[ids]['text']=df.loc[ids]['text'].replace('#FRUSTRADED :@',' FRUSTRADED')



ids=22587

df.loc[ids]['selected_text']='thank you so much'



ids=23672

df.loc[ids]['text']=df.loc[ids]['text'].replace('#NO WAYYY I couldn`t `tune in`',' no way i couldn`t tune in')

df.loc[ids]['selected_text']=' no way i couldn`t tune in'



ids=23741

df.loc[ids]['text']=df.loc[ids]['text'].replace('#excited',' excited')

df.loc[ids]['selected_text']='excited'

ids=24681

df.loc[ids]['selected_text']='Wave looks interesting'

ids=26093

df.loc[ids]['text']=df.loc[ids]['text'].replace('#heartbreak',' heartbreak')

df.loc[ids]['selected_text']='heart break'

#========

df.loc[7926]['selected_text']=df.loc[7926]['selected_text'][2:] +' you'



df.loc[166]['selected_text']=df.loc[166]['selected_text'][2:]

df.loc[251]['selected_text']=df.loc[251]['selected_text'][2:]

df.loc[1986]['selected_text']=df.loc[1986]['selected_text'][2:]



df.loc[706]['selected_text']=df.loc[706]['selected_text'][2:]

df.loc[6112]['selected_text']=df.loc[6112]['selected_text'][6:]

df.loc[11678]['selected_text']=df.loc[11678]['selected_text'][2:]

df.loc[12038]['selected_text']=df.loc[12038]['selected_text'][2:]

df.loc[23923]['selected_text']=df.loc[23923]['selected_text'][2:]

df.drop([26686,25690,24681,25690,26686,2269,697,1032],inplace=True)

df=df.dropna().reset_index(drop=True)

df.drop([6056,21591,23425,23669,26088],inplace=True)

df=df.dropna().reset_index(drop=True)



dset = TweetDataset(tweet = df.text.values,

                    sentiment = df.sentiment.values,

                    selected_text = df.selected_text.values,

                   MAX_LEN=MAX_LEN,

                   TOKENIZER=config.TOKENIZER,

                   is_train=True)

#dset[12538]
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
class TweetModel(nn.Module):

    def __init__(self):

        super(TweetModel, self).__init__()

        

        config = RobertaConfig.from_pretrained(

            '../input/roberta-base/config.json', output_hidden_states=True)    

        self.roberta = RobertaModel.from_pretrained(

            '../input/roberta-base/pytorch_model.bin', config=config)

        self.fc = nn.Linear(config.hidden_size, 2)

        nn.init.normal_(self.fc.weight, std=0.02)

        nn.init.normal_(self.fc.bias, 0)



    def forward(self, x):

        _, _, hs = self.roberta(x['ids'], x['mask'])

         

        x = torch.stack([hs[-1], hs[-2], hs[-3]])

        x = torch.mean(x, 0)

        perm = None

        '''if self.training:

            perm=torch.randperm(x.size()[0])

            y=x[perm,:,:]

            x = self.fc(0.9*x+0.1*y.detach())

        else:'''

        x = self.fc(x)

        start_logits, end_logits = x.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)

        end_logits = end_logits.squeeze(-1)

                

        return start_logits, end_logits,perm
import torch

import torch.nn as nn







class LabelSmoothSoftmaxCEV1(nn.Module):

    '''

    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients

    '''



    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):

        super(LabelSmoothSoftmaxCEV1, self).__init__()

        self.lb_smooth = lb_smooth

        self.reduction = reduction

        self.lb_ignore = ignore_index

        self.log_softmax = nn.LogSoftmax(dim=1)



    def forward(self, logits, label):

        '''

        args: logits: tensor of shape (N, C, H, W)

        args: label: tensor of shape(N, H, W)

        '''

        # overcome ignored label

        logits = logits.float() # use fp32 to avoid nan

        with torch.no_grad():

            num_classes = logits.size(1)

            label = label.clone().detach()

            ignore = label == self.lb_ignore

            n_valid = (ignore == 0).sum()

            label[ignore] = 0

            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes

            label = torch.empty_like(logits).fill_(

                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()



        logs = self.log_softmax(logits)

        loss = -torch.sum(logs * label, dim=1)

        loss[ignore] = 0

        if self.reduction == 'mean':

            loss = loss.sum() / n_valid

        if self.reduction == 'sum':

            loss = loss.sum()



        return loss

    

def loss_fn(outputs,targets):    

    loss_=0

    '''if outputs[2] is not None:

        loss_+=0.9*(nn.CrossEntropyLoss()(outputs[0],targets['targets_start']) + nn.CrossEntropyLoss()(outputs[1],targets['targets_end']))

        loss_+=0.1*(nn.CrossEntropyLoss()(outputs[0][outputs[2]],targets['targets_start']) + nn.CrossEntropyLoss()(outputs[1][outputs[2]],targets['targets_end']))

    else:'''

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



train_dataset = TweetDataset(df_train.text.values,df_train.sentiment.values,df_train.selected_text.values,MAX_LEN,config.TOKENIZER,is_train=True)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=TRAIN_BATCH_SIZE,shuffle=True)



valid_dataset = TweetDataset(df_valid.text.values,df_valid.sentiment.values,df_valid.selected_text.values,MAX_LEN,config.TOKENIZER,is_train=False)

valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=16,shuffle=False)
train_dataset[0][0]['ids']
#config.TOKENIZER.decode(list(train_dataset[0][0]['ids'])),list(train_dataset[0][1]['orig_selected'])
data = DataBunch(train_loader,valid_loader)

for x,y in valid_loader:

    break
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

    TRAIN_BATCH_SIZE=64

    MAX_LEN=128

    

    sample =pd.read_csv(SAMPLE_FILE)

    df_train,df_valid = df[df.kfold!=k],df[df.kfold==k]

    df_train=df_train[df_train.sentiment!='neutral']

    df_train = df_train.reset_index(drop=True)

    df_valid = df_valid.reset_index(drop=True)



    target_cols = list(sample.drop('textID',axis=1).columns)

    train_targets = df_train[target_cols].values

    valid_targets = df_valid[target_cols].values



    #tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = TweetDataset(df_train.text.values,df_train.sentiment.values,df_train.selected_text.values,MAX_LEN,config.TOKENIZER,is_train=True)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=TRAIN_BATCH_SIZE,shuffle=True)

    valid_dataset = TweetDataset(df_valid.text.values,df_valid.sentiment.values,df_valid.selected_text.values,MAX_LEN,config.TOKENIZER,is_train=False)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=16,shuffle=False)

    data = DataBunch(train_loader,valid_loader)

    

    model = TweetModel()

    params = list(model.named_parameters())

    optimizer = optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999))

    #cbfs = [Recorder,partial(AvgStatsCallback,metric),CudaCallback,ProgressCallback,partial(EarlyStopingCallback,1)]

    #sched = combine_scheds([0.3, 0.7], [sched_cos(1e-4/2,1e-4), sched_cos(1e-4, 1e-4/3)])

    cbfs = [Recorder,

        partial(AvgStatsCallback,metric),

        CudaCallback,

        ProgressCallback,

        #partial(ParamScheduler, 'lr', sched),

        partial(EarlyStopingCallback,3,'model_'+str(k))]

    learn = get_learner(model,optimizer,data,loss_fn, cb_funcs=cbfs)

    

    learn.fit(8)    

    test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

    test['selected_text']=test['text'].apply(lambda x:x)

    dset = TweetDataset(tweet = test.text.values,

                    sentiment = test.sentiment.values,

                    selected_text = test.selected_text.values,

                   MAX_LEN=MAX_LEN,

                   TOKENIZER=config.TOKENIZER,

                       is_train=False)

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
start3,end3=run_fold(3)
start4,end4=run_fold(4)
start = (start1+start2+start3+start4)/4

end = (end1+end2+end3+end4)/4
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

test['selected_text']=test['text'].apply(lambda x:x)

dset = TweetDataset(tweet = test.text.values,

                    sentiment = test.sentiment.values,

                    selected_text = test.selected_text.values,

                   MAX_LEN=MAX_LEN,

                   TOKENIZER=config.TOKENIZER,

                   is_train=False)

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