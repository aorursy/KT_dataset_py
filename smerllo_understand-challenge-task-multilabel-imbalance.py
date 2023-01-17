import os 

import pandas as pd 

import numpy as np

os.listdir('../input/lish-moa/')
train_feats = pd.read_csv('../input/lish-moa/train_features.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
print('shape of train_feats             ....', train_feats.shape)

print('shape of train_targets_scored    ....', train_targets_scored.shape)

print('shape of train_targets_nonscored ....', train_targets_nonscored.shape)

print('shape of test_features           ....', test_features.shape) 

print('shape of sample_submission       ....', sample_submission.shape)
print('number of unique sig_id ...', train_feats.sig_id.nunique())
train_feats[[c for c in train_feats.columns if 'g-' in c]]
train_feats[[c for c in train_feats.columns if 'c-' in c]]
train_targets_scored.head()
import seaborn as sns 

import matplotlib.pyplot as plt

import numpy as np



def plot_labels(x_dict): 

    x = {k: v for k, v in sorted(x_dict.items(), key=lambda item: item[1], reverse=True)}

    keys = x.keys()

    vals = x.values()



    font = {'family': 'serif',

            'color':  'green',

            'weight': 'normal',

            'size': 16,

            }

    fig, ax = plt.subplots(figsize=(35, 10))

    plt.bar(keys, vals)

    plt.title('MoA Scored Labels ',fontdict=font)

    plt.ylabel ('Counts')

    plt.xlabel ('Labels')

    plt.xticks(list(keys))

    # plt.

    plt.show()

labels = train_targets_scored.columns[1:].to_list()

plot_labels(train_targets_scored[labels].sum().to_dict())
plot_labels({k:v for k,v in train_targets_scored[labels].sum().to_dict().items() if v>300})
print('Most popular labels ....')

list({k:v for k,v in train_targets_scored[labels].sum().to_dict().items() if v>300}.keys())