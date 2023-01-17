# Importing the required libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Getting the dataset

df = pd.read_csv('../input/hfi_cc_2018.csv')
df.head()
df.describe()
p = sns.countplot(data=df, x="region")

_ = plt.setp(p.get_xticklabels(), rotation=90)
data_brz = df.loc[df.loc[:,'countries']=='Brazil',:]

data_rus = df.loc[df.loc[:,'countries']=='Russia',:]

data_ind = df.loc[df.loc[:,'countries']=='India',:]

data_chn = df.loc[df.loc[:,'countries']=='China',:]

data_sa = df.loc[df.loc[:,'countries']=='South Africa',:]
_ = plt.plot('year', 'hf_score', data=data_brz)

_ = plt.plot('year', 'hf_score', data=data_rus)

_ = plt.plot('year', 'hf_score', data=data_ind)

_ = plt.plot('year', 'hf_score', data=data_chn)

_ = plt.plot('year', 'hf_score', data=data_sa)

_ = plt.legend(('Brazil', 'Russia', 'India', 'China', 'South Africa'))
_ = plt.plot('year', 'hf_rank', data=data_brz)

_ = plt.plot('year', 'hf_rank', data=data_rus)

_ = plt.plot('year', 'hf_rank', data=data_ind)

_ = plt.plot('year', 'hf_rank', data=data_chn)

_ = plt.plot('year', 'hf_rank', data=data_sa)

_ = plt.legend(('Brazil', 'Russia', 'India', 'China', 'South Africa'),loc='upper right')
_ = plt.plot('year', 'ef_score', data=data_brz)

_ = plt.plot('year', 'ef_score', data=data_rus)

_ = plt.plot('year', 'ef_score', data=data_ind)

_ = plt.plot('year', 'ef_score', data=data_chn)

_ = plt.plot('year', 'ef_score', data=data_sa)

_ = plt.legend(('Brazil', 'Russia', 'India', 'China', 'South Africa'),loc='upper right')
_ = plt.plot('year', 'ef_rank', data=data_brz)

_ = plt.plot('year', 'ef_rank', data=data_rus)

_ = plt.plot('year', 'ef_rank', data=data_ind)

_ = plt.plot('year', 'ef_rank', data=data_chn)

_ = plt.plot('year', 'ef_rank', data=data_sa)

_ = plt.legend(('Brazil', 'Russia', 'India', 'China', 'South Africa'),loc='upper right')