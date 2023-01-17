%load_ext autoreload

%autoreload 2



%matplotlib inline
import torch
from pandas_summary import DataFrameSummary

from IPython.display import display

from sklearn import metrics
# ALl fastai imports

from fastai.imports import * # it imports has basic packages like pandas, numpy

from fastai.structured import *

from fastai.transforms import *

from fastai.conv_learner import *

from fastai.model import *

from fastai.dataset import *

from fastai.sgdr import *

from fastai.plots import *
import random

import time



def strTimeProp(start, end, format, prop):

    stime = time.mktime(time.strptime(start, format))

    etime = time.mktime(time.strptime(end, format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(format, time.localtime(ptime))





def randomDate(start, end, prop):

    return strTimeProp(start, end, '%m/%d/%Y %I:%M %p', prop)
df = pd.DataFrame()

l1 = []

for i in range(10):

    l1.append(randomDate("1/1/2008 1:30 PM", "1/1/2009 4:50 AM", random.random()))



df['date'] = l1
df
add_datepart(df, 'date')

df
df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})

df
train_cats(df)

print(df)

print(df.col2.cat.categories)
df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})

train_cats(df)

df # col2 is categorical now
x, y, nas = proc_df(df, 'col1')

print(x) # cat to num (no response)

print(y) # response

print(nas)
! ls ../input/data22/data2/
PATH = "../input/data22/data2"
import os

os.makedirs('{PATH}/tmp', exist_ok=True)

!ls {PATH}
os.makedirs('/cache/tmp', exist_ok=True)

!ln -fs /cache/tmp {PATH}
arch=resnet34

data = ImageClassifierData.from_paths(PATH, trn_name='train', val_name='valid', tfms=tfms_from_model(arch, 224))

learn = ConvLearner.pretrained(arch, data, precompute=True)

learn.fit(0.01, 3)