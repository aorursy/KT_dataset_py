# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/tv-sales-forecast'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/tv-sales-forecast/Date and model wise sale.csv')
df.head()
df.info()
## Here Date column is String so its need convert datetime
df['Date'] = pd.to_datetime(df['Date'])
df.info()
df.groupby('Model').Count.mean().sort_values(ascending = False)
## We will find only top 20 and last 20 product count sales
top20 = df.groupby('Model').Count.mean().sort_values(ascending = False).head(20)
top20
last20 = df.groupby('Model').Count.mean().sort_values(ascending = False).tail(20)
last20
top20 = pd.DataFrame(top20)

last20 = pd.DataFrame(last20)
top20
## visualiza top20 and last20 model
import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline
## top 20 model count
plt.figure(figsize=(20,10))

sns.barplot(x= top20.index, y = 'Count', data = top20, palette='summer')
## Last20 Model count
plt.figure(figsize=(20,10))

sns.barplot(x= last20.index, y = 'Count', data = last20, palette='spring')
datawise = df.groupby('Date').Count.mean().sort_values(ascending =False)
datewise = pd.DataFrame(datawise)
plt.figure(figsize=(25,7))

g=sns.barplot(x = datewise.index, y= 'Count', data = datewise)

for item in g.get_xticklabels():

    item.set_rotation(90)
datewise.resample('BM').mean()
business_month = datewise.resample('BM').mean()
plt.figure(figsize=(15,7))

g=sns.barplot(x = business_month.index, y = 'Count', data = business_month)

for item in g.get_xticklabels():

    item.set_rotation(90)
weekly = datewise.resample('W').mean()
weekly
plt.figure(figsize=(30,10))

g = sns.barplot(x=weekly.index, y = 'Count', data = weekly)

for item in g.get_xticklabels():

    item.set_rotation(90)
plt.figure(figsize=(30,10))

g = sns.lineplot(x=weekly.index, y = 'Count', data = weekly)

for item in g.get_xticklabels():

    item.set_rotation(90)
## Model M18 
m18 = df[df['Model'] =='M18']
m18 = pd.DataFrame(m18)
plt.figure(figsize=(25,7))

g = sns.barplot(x= m18.index, y = 'Count', data = m18, palette="Set1")

for item in g.get_xticklabels():

    item.set_rotation(90)