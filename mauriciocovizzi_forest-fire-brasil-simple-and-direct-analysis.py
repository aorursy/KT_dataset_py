# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv'

                     , encoding='latin1')
df.head()
df.columns
df.groupby('state')['number'].sum().sort_values(ascending=False)
plt.figure(figsize=(26,5))

sns.barplot(x=df['state'], y=df['number'], estimator=sum)
plt.figure(figsize=(26,5))

sns.barplot(x=df['year'], y=df['number'], estimator=sum)
heat = df.pivot_table(index='year', columns='state', values='number', aggfunc=sum)
sns.heatmap(heat)
meses={'Janeiro':1,'Fevereiro':2,'Marco':3,'Abril':4,'Maio':5,'Junho':6,'Julho':7,'Agosto':8,'Setembro':9,'Outubro':10,'Novembro':11,'Dezembro':12}
df['mes'] = df['month'].map(meses)
df.pivot_table(values='number',index='state', columns='mes', aggfunc=sum)
sns.heatmap(df.pivot_table(values='number',index='state', columns='mes', aggfunc=sum))