# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing dataset

data = pd.read_csv('../input/the-economists-big-mac-index/output-data/big-mac-adjusted-index.csv')

data.head(10)
data.isna().sum()
data.info()
# selecting rows based on condition 

rslt_df = data.loc[(data['local_price'] == data['dollar_price']) & (data['date'].str[0:4] == '2020')] 

rslt_df
# importing libraries for data visualizatioI

import matplotlib.pyplot as plt

import seaborn as sns
# Converting the entries in the date column to the timestamp data type to improve usability

data['date'] = data['date'].astype('datetime64[ns]') 

data.head()

type(data['date'][0])
sns.set(rc={'figure.figsize':(40,20)})

sns.set_style('whitegrid')

fig = sns.lineplot(x='date', y='dollar_price', data=data, hue = 'name')

fig.legend(loc='upper right', ncol=1)
sns.set(rc={'figure.figsize':(40,20)})

sns.set_style('whitegrid')

fig = sns.lineplot(x='date', y='local_price', data=data, 

             hue = 'name')

fig.legend(loc='upper right', ncol=1)
sns.set(rc={'figure.figsize':(40,20)})

sns.set_style('whitegrid')

fig = sns.lineplot(x='date', y='adj_price', data=data, 

             hue = 'name')

fig.legend(loc='upper right', ncol=1)
temp_data = data.groupby(["name"])['USD'].aggregate(np.mean).reset_index().sort_values('USD')

temp_data['Rank']=np.arange(1,1+len(temp_data))

dollar_ex = data.groupby(["name"])['dollar_ex'].aggregate(np.mean)
temp_data.head()
sns.set(rc={'figure.figsize':(15,20)})

sns.set_style('ticks')

sns.barplot(x=temp_data['USD'], y=temp_data['name'], data=temp_data, ci=None, order=temp_data['name'])