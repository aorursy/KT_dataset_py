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
import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
train.describe()
train.isnull().sum()
train = train.dropna()
test.isnull().sum()
sns.set()

fig = plt.figure(figsize=(12,6))

plt.hist(train['sentiment'],color = "steelblue")
fig = plt.figure(figsize=(12,6))

plt.hist(test['sentiment'],color = "steelblue")
train['len_text'] = train['text'].apply(lambda x:len(str(x).split()))
train.head()
train['len_target'] = train['selected_text'].apply(lambda x:len(str(x).split()))
test['len_text'] = test['text'].apply(lambda x:len(str(x).split()))
sns.kdeplot(train['len_text'])
fig = plt.figure(figsize=(12,6))

plt.hist(train['len_text'])
sns.kdeplot(train['len_target'])
plt.hist(train['len_target'])
sns.distplot(train[train['sentiment']=='positive']['len_text'])

sns.distplot(train[train['sentiment']=='negative']['len_text'])

sns.distplot(train[train['sentiment']=='neutral']['len_text'])
sns.kdeplot(train[train['sentiment']=='positive']['len_text'],label = 'positive',shade = True)

sns.kdeplot(train[train['sentiment']=='negative']['len_text'],label = 'negative', shade = True)

sns.kdeplot(train[train['sentiment']=='neutral']['len_text'],label = 'neutral',shade = True)
sns.kdeplot(train[train['sentiment']=='positive']['len_target'],label = 'positive')

sns.kdeplot(train[train['sentiment']=='negative']['len_target'],label = 'negative')

sns.kdeplot(train[train['sentiment']=='neutral']['len_target'],label = 'neutral')
sns.distplot(train[train['sentiment']=='positive']['len_target'])

sns.distplot(train[train['sentiment']=='negative']['len_target'])

sns.distplot(train[train['sentiment']=='neutral']['len_target'])
with sns.axes_style(style='ticks'):

    g = sns.factorplot("sentiment","len_text",data = train, kind = "box")

    g.set_axis_labels("Sentiment","Length of Tweets")
with sns.axes_style(style='ticks'):

    g = sns.factorplot("sentiment","len_target",data = train, kind = "box")

    g.set_axis_labels("Sentiment","Length of Selected Text")
with sns.axes_style(style='ticks'):

    g = sns.factorplot("sentiment","len_text",data = test, kind = "box")

    g.set_axis_labels("Sentiment","Length of Selected Text")