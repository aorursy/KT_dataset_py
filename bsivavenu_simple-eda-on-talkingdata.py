# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
for i in os.listdir('../input'):
#     print(i)
# print(os.path.getsize('../input/'+i ))
        print(i +'   ' +  str(round(os.path.getsize('../input/' + i) / 1000000,2)) + 'MB')
    
train = pd.read_csv('../input/train.csv',nrows=1000000)
test = pd.read_csv('../input/test.csv')
train_sample = pd.read_csv('../input/train_sample.csv')
train.shape,test.shape
train.columns
train.head()
test.head()
train.isnull().sum()
variables = ['ip', 'app', 'device', 'os', 'channel']
for v in variables:
    train[v] = train[v].astype('category')
    test[v]=test[v].astype('category')
#set click_time and attributed_time as timeseries
train['click_time'] = pd.to_datetime(train['click_time'])
train['attributed_time'] = pd.to_datetime(train['attributed_time'])
test['click_time'] = pd.to_datetime(test['click_time'])

#set as_attributed in train as a categorical
train['is_attributed']=train['is_attributed'].astype('category')
train[['attributed_time', 'is_attributed']][train['is_attributed']==1].describe()

train['is_attributed'].value_counts()
train[train['is_attributed']==1].sum()
#this is the blunder mistakes beginners do, actually attribute value==1 it sums up to 1693 and also it sums up remaining features values also which gives wrong
#results so beware
train.describe()
test['click_id']=test['click_id'].astype('category')
test.describe()
test.isnull().sum()
temp = train['ip'].value_counts().reset_index(name='counts')
temp.columns = ['ip', 'counts']
temp[:10]
train['timegap'] = train.attributed_time - train.click_time
train['timegap'].describe()
train['timegap'].value_counts().sort_values(ascending=False)[:10]
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
cols = ['ip', 'app', 'device', 'os', 'channel']
uniques = [len(train[col].unique()) for col in cols]
# sns.set(font_scale=1.2)
ax = sns.barplot(cols, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature (from 10,000,000 samples)')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center")