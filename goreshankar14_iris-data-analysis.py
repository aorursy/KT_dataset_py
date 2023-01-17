# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/Iris_data_sample.csv",index_col=0,na_values=['??','###'])
df['Species'].fillna(method='ffill',inplace=True)
g=df.groupby('Species')
igs=g.get_group('Iris-setosa')

mean_sp_len=igs['SepalLengthCm'].mean()

mean_sp_wdh=igs['SepalWidthCm'].mean()

mean_pt_len=igs['PetalLengthCm'].mean()
df['SepalLengthCm'].fillna(mean_sp_len,inplace=True)

df['SepalWidthCm'].fillna(mean_sp_wdh,inplace=True)

df['PetalLengthCm'].fillna(mean_pt_len,inplace=True)
sns.set(style="darkgrid")
fig,[ax_sp,ax_pt]=plt.subplots(1,2,figsize=(15, 10))

sns.regplot(x=df['SepalLengthCm'],y=df['SepalWidthCm'],fit_reg=False,ax=ax_sp)

sns.regplot(x=df['PetalLengthCm'],y=df['PetalWidthCm'],fit_reg=False,ax=ax_pt)
sns.lmplot(x='SepalLengthCm',y='SepalWidthCm',data=df,fit_reg=False,legend=True,hue='Species')
sns.lmplot(x='PetalLengthCm',y='PetalWidthCm',data=df,fit_reg=False,legend=True,hue='Species')
sns.pairplot(df,kind='scatter',hue='Species')