import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



datadir = "/kaggle/input/ufcdata"



dat_raw = pd.read_csv(os.path.join(datadir, 'raw_total_fight_data.csv'), sep=";")

dat_pp = pd.read_csv(os.path.join(datadir, 'preprocessed_data.csv'))

dat_fighter = pd.read_csv(os.path.join(datadir, 'raw_fighter_details.csv'))

data = pd.read_csv(os.path.join(datadir, 'data.csv'))
# extract weight into numeric type (float since there are some NAs)

wts = dat_fighter['Weight'].str.split(" ", expand=True)

wts.loc[:,1].value_counts()  # shows all wts are in lbs.

wts.loc[:, 0].value_counts().index

wts.loc[:, 0].isna().sum()

wts.loc[:, 0].astype(float)



dat_fighter['wts'] = wts.loc[:, 0].astype(float)
wts_cts = dat_fighter['wts'].value_counts().reset_index().rename(columns={'wts': 'ct', 'index': "weight"})

wts_cts_plt = wts_cts.plot.scatter('weight', 'ct', alpha=0.6, title="weight occurance counts", figsize=(10,10))

wts_cts_plt.set_xlabel("weight (lbs)")

wts_cts_plt.set_ylabel("count")