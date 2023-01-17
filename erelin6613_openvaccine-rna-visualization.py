!pip3 install -q forgi[all]

!conda install -y -c bioconda viennarna
import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import forgi.graph.bulge_graph as fgb

import forgi.visual.mplotlib as fvm



%matplotlib inline

plt.rcParams['figure.figsize'] = (14, 6);
root_dir = '../input/stanford-covid-vaccine'

train_path = os.path.join(root_dir, 'train.json')

test_path = os.path.join(root_dir, 'test.json')

bpps_dir = os.path.join(root_dir, 'bpps')
train = pd.read_json(train_path, lines=True)

test = pd.read_json(test_path, lines=True)

train.head(3)
train.columns, test.columns
for c in train.columns:

    if train[c].dtype == 'object':

        print(c, ':', train[c].apply(lambda x: len(x))[0])
def plot_sample():

    samp = train.sample(1)

    rna = []

    seq = samp.loc[samp.index[0], 'sequence']

    struct = samp.loc[samp.index[0], 'structure']

    bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{struct}\n{seq}')[0]

    fvm.plot_rna(bg)

    plt.show()
plot_sample()
plot_sample()
plot_sample()
plot_sample()
plot_sample()
plot_sample()