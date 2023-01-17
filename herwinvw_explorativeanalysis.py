# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option('display.max_rows', None)
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
train.head()
train.info()
train.describe(include='all')
train.groupby('target').size()
train[train['text'].duplicated(keep=False)]
train[train[['keyword','location','text','target']].duplicated(keep=False)]
train[train[['text','target']].duplicated(keep=False)]
train['keyword'].dropna().head()
df = train.groupby('keyword').size().reset_index(name='count')
df.sort_values(by='count', ascending = False)
train['has_keyword'] = ~train['keyword'].isnull()
train.groupby(['has_keyword','target']).size()
train['keyword'].describe()
df = train.groupby(['keyword','target']).size().reset_index(name='count')
df.sort_values(by=['keyword','count'])
df = train.groupby('location').size().reset_index(name ='count')
df = df.sort_values(by='count',ascending=False)
df['cum percent'] = round(100.0*df['count'].cumsum()/df['count'].sum())
df['nr'] = range(len(df))
df
df[df['count']>2]
from plotnine import *
train['text_len'] = train['text'].apply(len)
train.head()
train['target'] = train['target']==1
ggplot(train, aes(x='target', y='text_len',color='target'))+geom_boxplot()
