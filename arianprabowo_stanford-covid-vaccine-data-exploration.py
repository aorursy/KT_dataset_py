# https://www.kaggle.com/ricopue/second-structure-plot-and-info-with-forgi
# https://www.kaggle.com/erelin6613/openvaccine-rna-visualization/
!conda install -y -c bioconda forgi
!conda install -y -c bioconda viennarna
import warnings
warnings.filterwarnings('ignore')
import os

#Basic data manipulation libraries
import pandas as pd, numpy as np
import math, json, gc, random, os, sys
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm


#Deep Learning Libraries
import torch

#Library for model evaluation
from sklearn.model_selection import train_test_split, KFold

import forgi.graph.bulge_graph as fgb
import forgi.visual.mplotlib as fvm
train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)
sample_sub = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')

target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    return np.transpose(
        np.array(df[cols].applymap(lambda seq: [token2int[x] for x in seq]).values.tolist()),
        (0, 2, 1))

train_inputs = preprocess_inputs(train[train.signal_to_noise > 1])
train_labels = np.array(train[train.signal_to_noise > 1][target_cols].values.tolist()).transpose((0, 2, 1))
display(train.info())
display(train.describe())
display(train.head())
train.loc[0]
for col in train.loc[0].index:
    msg = ''
    col_type = type(train.loc[0][col])
    if col_type == str:
        msg = str(len(train.loc[0][col])) + ' ' + train.loc[0][col]
    elif col_type == list:
        msg = 'list.len=' + str(len(train.loc[0][col]))
    elif type(train.loc[0][col]).__module__ == np.__name__:
        msg = train.loc[0][col]
    else:
        msg = 'others'
    print('#',col,':',msg)
def plot_sample(i=None):
    if i is None:
        samp = train.sample(1)
    else:
        samp = train.loc[i:i]
    rna = []
    seq = samp.loc[samp.index[0], 'sequence']
    struct = samp.loc[samp.index[0], 'structure']
    bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{struct}\n{seq}')[0]
    plt.figure(figsize=[20,15])
    fvm.plot_rna(bg)
plot_sample(0)
bpps_1 = np.load('../input/stanford-covid-vaccine/bpps/id_00073f8be.npy')
print(bpps_1.shape)
plt.figure(figsize=[20,20])
plt.imshow(bpps_1)
plt.figure(figsize=[20,5])
_ = plt.hist(bpps_1.flatten(),log=1,bins=1000)
plt.figure(figsize=[10,10])
plt.imshow(bpps_1>0)
print((bpps_1==0).sum())
print(107**2)
plt.figure(figsize=[20,5])
_ = plt.hist(np.arcsinh(bpps_1*10**4).flatten(),log=1,bins=250)
plt.figure(figsize=[20,20])
plt.imshow(np.arcsinh(bpps_1*10**4))
