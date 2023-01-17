# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in \

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install ksfest --upgrade
df=pd.read_csv("/kaggle/input/russian-wholesale-electricity-market/RU_Electricity_Market_PZ_dayahead_price_volume.csv",thousands=",", decimal=".", delimiter=',',parse_dates=['timestep'], encoding='utf-8')

df.head()
all(df.dtypes.values==float)
df['year']=df.timestep.map(lambda x: x.strftime('%Y'))

df['year']=df['year'].astype(int)

len(np.unique(df['year']))
df.drop('timestep',inplace=True,axis=1)
import ksfest.ksfest as ksf
ksf_t=ksf.ks_fest()
cols=['consumption_eur', 'consumption_sib', 'price_eur',

       'price_sib']
ks_df=ksf_t.get_ks(df, var_dim='year', columns = cols, sample=0.7)
ks_df
f, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(ks_df.drop('year', axis=1),yticklabels=list(ks_df.year),

           #cbar_kws={"shrink": .82},

            vmax=.7, square=True);
f, ax = plt.subplots(figsize=(10, 8))

for col in cols:

    plt.plot(ks_df.year, ks_df[col], label=str(col))

plt.xticks(rotation=90)

plt.legend()