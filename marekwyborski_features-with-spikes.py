import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('seaborn')

sns.set(font_scale=2)
test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')

id_label = 'ID_code'

target_label = 'target'

features = [c for c in train.columns if c not in [id_label, target_label]]
mask_train = train[target_label] >= 0.5

train_true = train[mask_train]

train_false = train[~mask_train]
def plot_feature(feature, d1_name, d2_name, d1, d2, col1, col2):

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    bw = 0.0

    sns.kdeplot(d1, ax=ax, label=d1_name, color=col1, bw=bw)

    sns.kdeplot(d2, ax=ax, label=d2_name, color=col2, bw=bw)

    ax.set_title('{}:    {} - {}'.format(feature, d1_name, d2_name))

    plt.show()
for feature in features:

    plot_feature(feature, 'train_true', 'train_false', train_true[feature].values, train_false[feature].values, 'g', 'r')
for feature in features:

    plot_feature(feature, 'train', 'test', train[feature].values, test[feature].values, 'violet', 'b')