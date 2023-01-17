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
import re

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

%matplotlib inline
train_df = pd.read_csv('../input/titanic/train.csv')

train_df
train_df.describe()
train_df.info()
train_df.columns = map(str.lower, train_df.columns)

train_df
df_percent = train_df.isnull().sum()/len(train_df)*100

df_percent.sort_values(ascending=False)
sns.heatmap(train_df.isnull(),yticklabels = False, cbar= False)
train_df.groupby('survived').count().passengerid
fig = px.sunburst(train_df, path=['sex', 'survived'], values='passengerid')

fig.show()
fig = px.violin(train_df, y="age", x="sex", color="survived", hover_data=train_df.columns, points='all', range_y=[train_df.age.min()-.5, train_df.age.max()+.5])

fig.show()
fig = px.histogram(train_df, x="age", y="survived", color="pclass",  hover_data=train_df.columns)

fig.show()
sns.countplot(data= train_df, x = 'pclass', hue= 'survived')
sns.countplot(data= train_df, x = 'pclass')
sns.countplot(data = train_df, x = 'sex', hue = 'survived')
train_df.ticket.describe()
train_df.loc[train_df['ticket'] == 'CA. 2343']
train_df.cabin.dropna().astype(str).str[0]
train_df['new_cabin'] = train_df.cabin.dropna().astype(str).str[0] #tekrar incelencek NAN değerler düşmüyor.

train_df
df= train_df.groupby(by=['new_cabin','pclass']).pclass.count()

df
train_df.groupby(by='pclass').pclass.count()
train_df[train_df.pclass==1 & train_df.cabin.isnull()]

train_df[(train_df.survived ==0) & (train_df.cabin.isnull())].count()
train_df = train_df.drop(columns = 'new_cabin')

def cabinExists(dataFrame):

  dataFrame["cabin"] = dataFrame["cabin"].fillna(0)

  dataFrame.cabin = dataFrame.cabin.apply(lambda x: 0 if x == 0 else 1)

  return dataFrame
train_df = cabinExists(train_df)

train_df
train_df['male_cat'] = pd.get_dummies(train_df.sex, drop_first=True)

train_df

def age_categorize(trainSet):

    interval = (0,5,12,18,25,35,60,100)

    age_cat = ('babies', 'children', 'teenage', 'student', 'young', 'adult', 'old')

    trainSet['age_cat'] = pd.cut(trainSet.age, interval, labels = age_cat)

    return trainSet
train_df = age_categorize(train_df)

train_df
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_df.drop(columns=["survived"]), train_df["survived"], random_state = 42)
