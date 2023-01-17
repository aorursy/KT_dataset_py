import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt



import pickle

from tqdm import tqdm



import h2o

from h2o.automl import H2OAutoML

label_df = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
labels = label_df.columns.difference(['sig_id'])
pos_counts = {}

for i,col in enumerate(labels):

    pos_count = label_df[col].value_counts()[1]

    pos_counts[f"{i}-{col}"] = pos_count

    if pos_count<10:

        print(col,':',pos_count)
pos_df = pd.DataFrame({'label':list(pos_counts.keys()), 

              'pos_counts':list(pos_counts.values())})

pos_df_sorted = pos_df.sort_values('pos_counts', ascending = False)
fig ,ax = plt.subplots(figsize=(20,30))

sns.barplot(data=pos_df_sorted, x='pos_counts', y='label',ax=ax)
with open('../input/h2oleaderboards/LBs.pkl', 'rb') as f:

    lbs = pickle.load(f)
scores = []

for i,val in tqdm(enumerate(lbs.values()), total=206):

    if type(val) == str:

        print(val,f"{i}-{list(lbs.keys())[i]}")

        score = np.nan

    else:

        score = val['logloss'].values[0]

    scores.append(score)

    
pos_df['logloss'] = scores 
pos_df_score_sorted = pos_df.sort_values('logloss', ascending=False)
fig ,ax = plt.subplots(figsize=(20,30))

sns.barplot(data=pos_df_score_sorted, x='logloss', y='label',ax=ax)