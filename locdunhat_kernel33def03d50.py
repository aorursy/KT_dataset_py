%matplotlib inline

import os

import numpy as np 

import pandas as pd 

import datetime as dt

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import datetime as dt

from plotly.offline import init_notebook_mode, iplot

from tqdm import tqdm

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

import warnings

from collections import Counter
plt.rcParams['figure.figsize'] = [16, 10]

plt.rcParams['font.size'] = 14

warnings.filterwarnings('ignore')

pd.options.display.max_columns = 99

sns.set_palette(sns.color_palette('tab20', 20))
start = dt.datetime.now()
base = '/kaggle/input/'

dirs = os.listdir(base)

print(dirs)
train_path = '/kaggle/input/train_labels.csv'

test_path = '/kaggle/input/test_labels.csv'

valid_path = '/kaggle/input/valid_labels.csv'



train_inp = pd.read_csv(base + 'train_labels.csv')

test_inp = pd.read_csv(base + 'test_labels.csv')

valid_inp = pd.read_csv(base + 'val_labels.csv')





print ("Total Images in the train:", len(train_inp))

print ("Total Labels in the train:", len(train_inp['class'].value_counts()))

print ("")



print ("Total Images in the test:", len(test_inp))



print ("")



print ("Total Images in the valid:", len(valid_inp))

print ("Total Labels in the valid:", len(valid_inp['class'].value_counts()))
train_inp.head()

print(train_inp['class'].value_counts())
train_labels = Counter(train_inp['class'].value_counts())

train=train_inp['class'].value_counts()

xvalues = list(train.keys())

yvalues = list(train_labels.keys())

print(xvalues)

print(yvalues)

trace1 = go.Bar(x=xvalues, y=yvalues, opacity=0.8, name="class", marker=dict(color='rgba(20, 20, 20, 1)'))

layout = dict(width=800, title='Distribution of different labels in the train dataset', legend=dict(orientation="h"));



fig = go.Figure(data=[trace1], layout=layout);

iplot(fig);
valid_labels = Counter(valid_inp['class'].value_counts())

valid=valid_inp['class'].value_counts()

xvalues = list(valid.keys())

yvalues = list(valid_labels.keys())

print(xvalues)

print(yvalues)

trace1 = go.Bar(x=xvalues, y=yvalues, opacity=0.8, name="class", marker=dict(color='rgba(20, 20, 20, 1)'))

layout = dict(width=800, title='Distribution of different labels in the valid dataset', legend=dict(orientation="h"));



fig = go.Figure(data=[trace1], layout=layout);

iplot(fig);