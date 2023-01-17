import numpy as np 

import pandas as pd

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt



train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)

df = pd.DataFrame({

                "id" :  [],

                "pos" : [],

                "reactivity":  [],

                "deg_Mg_pH10": [],

                "deg_pH10":    [],

                "deg_Mg_50C":  [],

                "deg_50C":     []

            })



for index, row in tqdm(train.iterrows(), total=train.shape[0]):

    r = pd.DataFrame({

                "id" :  np.repeat(row.id, row.seq_scored),

                "pos" : np.arange(row.seq_scored),

                "reactivity":  row.reactivity,

                "deg_Mg_pH10": row.deg_Mg_pH10,

                "deg_pH10":    row.deg_pH10,

                "deg_Mg_50C":  row.deg_Mg_50C,

                "deg_50C":     row.deg_50C

            })

    df = df.append(r)
df
bp = df.boxplot(figsize = (20,15),column='reactivity',by='pos')
bp = df.boxplot(figsize = (20,15),column='deg_Mg_pH10',by='pos')
bp = df.boxplot(figsize = (20,15),column='deg_pH10',by='pos')
bp = df.boxplot(figsize = (20,15),column='deg_Mg_50C',by='pos')
bp = df.boxplot(figsize = (20,15),column='deg_50C',by='pos')
means = df.groupby(by='pos').mean()
means.loc[means.index==1]
submission = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
submission['id'] = submission.id_seqpos.str.extract(r'_(\d*)$').astype(int)
for i in tqdm(range(68)):

    submission.loc[submission['id'] == i, ['reactivity']] = means.loc[means.index==i]['reactivity'][i]

    submission.loc[submission['id'] == i, ['deg_Mg_pH10']] = means.loc[means.index==i]['deg_Mg_pH10'][i]

    submission.loc[submission['id'] == i, ['deg_pH10']] = means.loc[means.index==i]['deg_pH10'][i]

    submission.loc[submission['id'] == i, ['deg_Mg_50C']] = means.loc[means.index==i]['deg_Mg_50C'][i]

    submission.loc[submission['id'] == i, ['deg_50C']] = means.loc[means.index==i]['deg_50C'][i]
submission = submission.drop(columns=['id'])
submission.to_csv('submission.csv', index=False)
