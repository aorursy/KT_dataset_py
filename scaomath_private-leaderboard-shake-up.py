import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

from sklearn.metrics import mean_absolute_error as mae
DIR_NAME = {'DeeperGCN':'../input/openvaccine-deepergcn/',

           'RNN':'../input/gru-lstm-with-feature-engineering-and-augmentation/',

           'AE_TF':'../input/covid-ae-pretrain-gnn-attn-cnn/',

           'AE_PT':'../input/openvaccine-pytorch-ae-pretrain/'}
test  = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json",lines=True)

test_pri = test[test["seq_length"] == 130]

test_pri.head()
sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")

id_pri = []

for i, uid in enumerate(test_pri['id']):

    id_seqpos = [f'{uid}_{x}' for x in range(130)]

    id_pri += id_seqpos

id_pri = pd.DataFrame(id_pri, columns=['id_seqpos'])
sub.loc[sub['id_seqpos'].isin(id_pri['id_seqpos'])].head()
DIR_NAME[key[1]]
subs = [] 

target_cols = ['reactivity', 'deg_Mg_50C','deg_Mg_pH10']



for key in enumerate(DIR_NAME.keys()):

    df = pd.read_csv(DIR_NAME[key[1]]+'submission.csv',

                   index_col=False, 

                   usecols=['id_seqpos']+target_cols)

    df.sort_values('id_seqpos',inplace=True, ascending=True)

    df['model'] = key[1]

    print(set(sub.id_seqpos) == set(df.id_seqpos))

    subs.append(df)



    
for t in target_cols:

    print(f'\nMean abs difference in {t}:\n')

    for i in range(len(DIR_NAME)):

        for j in range(i+1, len(DIR_NAME)):

            df_i, df_j = subs[i], subs[j]

            abs_diff= mae(subs[i][t], subs[j][t])

            print(f'submission {i} and {j}: {abs_diff:.5f}')
N_VIS = 200

RANGE_VIS = [[68, 75], [76, 83], [84, 91]]
def plot_pairplot(positions):

    id_pri = []

    for i, uid in enumerate(test_pri['id']):

        id_seqpos = [f'{uid}_{x}' for x in range(positions[0], positions[1])]

        id_pri += id_seqpos

    id_pri = pd.DataFrame(id_pri, columns=['id_seqpos'])



    subs_pri = []

    for i in range(len(DIR_NAME)): 

        df_tmp = subs[i]

        subs_pri.append(df_tmp.loc[df_tmp['id_seqpos'].isin(id_pri['id_seqpos'])])

        

    idx = np.random.randint(0,len(subs_pri[0]), N_VIS)

    subs_vis = []

    df_vis = pd.DataFrame()

    for i in range(len(DIR_NAME)):

        df_vis = subs_pri[i].iloc[idx].copy()

        df_vis.loc[:,target_cols] = df_vis[target_cols].values

        subs_vis.append(df_vis)

    df_vis = pd.concat(subs_vis)

    

    sns.set_style(style="ticks")

    sns.pairplot(df_vis, hue="model");

    

def plot_lineplot(positions):

    id_pri = []

    for i, uid in enumerate(test_pri['id']):

        id_seqpos = [f'{uid}_{x}' for x in range(positions[0], positions[1])]

        id_pri += id_seqpos

    id_pri = pd.DataFrame(id_pri, columns=['id_seqpos'])



    subs_pri = []

    for i in range(len(DIR_NAME)): 

        df_tmp = subs[i]

        subs_pri.append(df_tmp.loc[df_tmp['id_seqpos'].isin(id_pri['id_seqpos'])])

        

    idx = np.random.randint(0,len(subs_pri[0]), N_VIS)

    subs_vis = []

    df_vis = pd.DataFrame()

    for i in range(len(DIR_NAME)):

        df_vis = subs_pri[i].iloc[idx].copy()

        df_vis.loc[:,target_cols] = df_vis[target_cols].values

        subs_vis.append(df_vis)

    df_vis = pd.concat(subs_vis)



    fig, axes = plt.subplots(3,1,figsize=(10, 15))

    

    for i, col in enumerate(target_cols):

        g = sns.lineplot(data=df_vis, x="id_seqpos", y=col, hue='model', ax=axes[i])

        g.set(xticklabels=[]) 
plot_pairplot(RANGE_VIS[0])
plot_pairplot(RANGE_VIS[1])
plot_pairplot(RANGE_VIS[2])
plot_lineplot(RANGE_VIS[0])
plot_lineplot(RANGE_VIS[1])
plot_lineplot(RANGE_VIS[2])
def plot_rna_preds(seq_ids=None, n_sample=1, positions=(68,91)):

    if seq_ids is None:

        ids = test_pri['id'].sample(n=n_sample)

    else:

        ids= seq_ids

    id_pri = []

    for i, uid in enumerate(ids):

        id_seqpos = [f'{uid}_{x}' for x in range(positions[0],positions[1])]

        id_pri += id_seqpos

    id_pri = pd.DataFrame(id_pri, columns=['id_seqpos'])

    subs_pri = []

    for i in range(len(DIR_NAME)): 

        df_tmp = subs[i]

        subs_pri.append(df_tmp.loc[df_tmp['id_seqpos'].isin(id_pri['id_seqpos'])])

        

    subs_vis = []

    df_vis = pd.DataFrame()

    for i in range(len(DIR_NAME)):

        df_tmp = subs_pri[i]

        df_vis = df_tmp.loc[df_tmp['id_seqpos'].isin(id_pri['id_seqpos'])].copy()

        df_vis.loc[:,target_cols] = df_vis[target_cols].values

        subs_vis.append(df_vis)

    df_vis = pd.concat(subs_vis)

    

    fig, axes = plt.subplots(3*n_sample,1,figsize=(10, 15*n_sample))

    for j, id_seq in enumerate(ids):

        for i, col in enumerate(target_cols):

            g = sns.lineplot(data=df_vis, x="id_seqpos", y=col, hue='model', ax=axes[i+j*3])

            g.set(xticklabels=[]) 

            g.set_title(f'{id_seq}')
plot_rna_preds(seq_ids=['id_9085aafc1'])
plot_rna_preds(seq_ids=['id_4cc792927'])
plot_rna_preds(seq_ids=['id_2dc15cef2'])