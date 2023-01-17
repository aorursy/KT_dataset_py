# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
# Input data files are available in the "../input/" directory.
# For exam

# Any results you write to the current directory are saved as output.

df=pd.read_csv("../input/Age_group.csv")
df=df[df.type!='Detenus']
df=df[df.category!='Foreigners']
df=df.drop(['category'],1)
df['total']=df['age_16_18']+df['age_18_30']+df['age_30_50']+df['age_50_above']
df[['gender','total']].groupby(['gender']).sum().plot(kind='bar')
df.head()
df[['year','age_16_18','age_18_30','age_30_50','age_50_above']].groupby(['year']).mean()
df2=df[['year', 'total']].groupby(['year']).sum()
df2.plot()
df3=df[['state_name', 'total']].groupby('state_name').sum()
df3.plot(kind='bar')