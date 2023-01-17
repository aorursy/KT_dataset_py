import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df_oof = pd.read_csv('../input/rnaoutputanalysis/holdouts.csv')
df_oof
df_oof['seq'] = df_oof.id_seqpos.apply(lambda v: v.split('_')[-2])

df_oof['i'] = df_oof.id_seqpos.apply(lambda v: int(v.split('_')[-1]))

df_oof = df_oof.sort_values('i')
df_oof = df_oof[df_oof.SN_filter == 1]
df_oof_react = df_oof[['seq', 'reactivity']].groupby('seq').reactivity.apply(list).reset_index()

df_oof_react
np.stack(df_oof_react.reactivity.values).shape
fig, ax = plt.subplots(1, 1, figsize=(30, 10))



oof_labels_filtered = np.stack(df_oof_react.reactivity.values)



for i in range(oof_labels_filtered.shape[0]):

    ax.plot(oof_labels_filtered[i, :], alpha=0.008, c='blue');

    ax.set_title('ractivity (holdout)')



ax.plot(np.mean(oof_labels_filtered, axis=0), alpha=1, c='darkblue', linewidth=2);

ax.plot(np.median(oof_labels_filtered, axis=0), alpha=1, c='darkblue', linewidth=2, linestyle='dashed');



ax.set_ylim(0, 4)
# EDA

df_train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)
fig, ax = plt.subplots(1, 1, figsize=(30, 10))



train_labels_filtered = np.stack(df_train[df_train.SN_filter.values == 1].reactivity.values)



for i in range(train_labels_filtered.shape[0]):

    ax.plot(train_labels_filtered[i, :], alpha=0.008, c='green');

    ax.set_title('reactivity (train)')



ax.plot(np.mean(train_labels_filtered[:, :], axis=0), alpha=1, c='darkgreen', linewidth=2);

ax.plot(np.median(train_labels_filtered[:, :], axis=0), alpha=1, c='darkgreen', linewidth=2, linestyle='dashed');



ax.set_ylim(0, 4)
from __future__ import print_function

from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets
@interact(i=widgets.IntSlider(min=0, max=train_labels_filtered.shape[0]-1, step=1, value=0))

def show(i):

    plt.figure(figsize=(30, 3))

    plt.plot(train_labels_filtered[i, :], c='blue');

    plt.plot(oof_labels_filtered[i, :], c='green');

    plt.xlim(0, 107)

    plt.gca().set_xticks(np.arange(107))

    plt.gca().set_xticklabels(df_train.iloc[i].sequence)



    a = plt.gca().twiny()

    a.set_xticks(np.arange(107))

    a.set_xticklabels(df_train.iloc[i].structure)

    a.get_yaxis().set_visible(False)



    plt.grid()
# show residuals



fig, ax = plt.subplots(1, 1, figsize=(30, 10))



residuals = oof_labels_filtered - train_labels_filtered



for i in range(residuals.shape[0]):

    ax.plot(residuals[i, :], alpha=0.01, c='red');

    ax.set_title('residuals')



ax.plot(np.mean(residuals[:, :], axis=0), alpha=1, c='darkred', linewidth=2);

ax.plot(np.median(residuals[:, :], axis=0), alpha=1, c='darkred', linewidth=2, linestyle='dashed');



ax.set_ylim(-1, 1)
np.mean(residuals[:, :].flatten())
np.median(residuals[:, :].flatten())