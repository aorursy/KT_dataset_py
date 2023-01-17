# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np

import pandas as pd

import glob

import os

import matplotlib.pyplot as plt

print(os.listdir("../input"))
path = r'../input'                     # use your path

all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent



df_from_each_file = (pd.read_csv(f) for f in all_files)

df   = pd.concat(df_from_each_file, ignore_index=True)
df_India = df[df['country_or_area']=='India']
df_India
#pivoted = df_India.pivot_table(values='value',columns='year' )

df_India_year_value = df_India[['year','value']]
df_India_year_value
res = df_India_year_value.groupby('year')['value'].mean()

#a.round(decimals=2)

#a['value'].mean()



#df_India_1[df_India_1['year']==2007]['value'].mean()
res.round(decimals=2)
plt.plot(res.round(decimals=2),'r-*')