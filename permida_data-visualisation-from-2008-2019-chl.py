# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/ice-hockey-khl-dataset/KHL_v1.csv')
data
data.shape
data.info()
data.columns = data.columns.str.lower()

print(data.columns)
data
data['date']=pd.to_datetime(data['date'], format='%m/%d/%Y')

data['monthname']=data['date'].dt.month_name()

data['year']=data['date'].dt.year

data['day'] = data['date'].dt.day

data['weekday']=data['date'].dt.day_name()
data['season'].unique()
print(data['hometeam'].unique())

print('________________________')

print(data['awayteam'].unique())

print('________________________')

print(data['winner'].unique())

print('________________________')
data['add'] = data['add'].fillna('Main time')
data['agot']=data['agot'].fillna(0)

data['hgot']=data['hgot'].fillna(0)

data['agso']=data['agso'].fillna(0)

data['hgso']=data['hgso'].fillna(0)

data['totalot']= data['totalot'].fillna(0)
data
data.pivot_table(index = 'year',values = 'totalfull').plot(grid = True,color = 'red',figsize=[15,5])
data.boxplot('totalfull')
data.query('totalfull>13')
data.pivot_table(index = ['month'] , values = 'totalfull').plot.barh(color='red',grid = True,figsize = [10,5],xlim=(4,5.5))
print('Home goals leaders')

print(data.pivot_table(index= 'hometeam',values = 'hg').sort_values(by = 'hg',ascending=False).reset_index().head(10))

print('Away goals leaders')

print(data.pivot_table(index= 'awayteam',values = 'ag').sort_values(by = 'ag',ascending=False).reset_index().head(10))
print('Misses goals at home.')

print(data.pivot_table(index= 'hometeam',values = 'ag').sort_values(by = 'ag',ascending=False).reset_index().head(10))

print('Misses goals away.')

print(data.pivot_table(index= 'awayteam',values = 'hg').sort_values(by = 'hg',ascending=False).reset_index().head(10))
data.pivot_table(index = 'hometeam',values = ['hg','ag']).sort_values(by='hg',ascending=True).plot.barh(xlim=(1.5,3.5),figsize=[5,15],grid=True)
data.pivot_table(index = 'awayteam',values = ['ag','hg']).sort_values('ag',ascending=True).plot.barh(grid=True,figsize=[5,15],xlim=(1.5,3.9))
data.query('totalot>0').pivot_table(index = 'year',values = 'winner',aggfunc = 'count').plot(style= 'o-',grid=True,figsize=[15,5])
data.query('hg1>2').pivot_table(index = 'winner',values='hg1',aggfunc ='count').sort_values('hg1').plot.barh(grid=True,color='red',figsize=[10,7])
plt.figure(figsize=(20,20))

p=sns.heatmap(data.corr(), annot=True,cmap='RdYlGn',square=True) 