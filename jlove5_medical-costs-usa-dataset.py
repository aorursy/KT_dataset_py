# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/insurance.csv')

df.head()
g = sns.catplot(data=df, kind='count', x='sex', palette='PRGn')
g=sns.catplot(data=df, x='smoker', kind='count', palette='PRGn')
g=sns.catplot(data=df, x='age', kind='count', palette='PRGn').set_xticklabels(rotation=90)
g=sns.catplot(data=df, x='children', kind='count', palette='PRGn')
g=sns.catplot(data=df, kind='count', x='region')
g=sns.catplot(data=df, kind='swarm', x='age', y='charges', hue='sex', col='smoker', palette='Spectral').set_xticklabels(rotation=90)
g=sns.catplot(data=df, kind='swarm', x='age', y='charges', hue='children').set_xticklabels(rotation=90)
df.columns
g=sns.catplot(data=df, kind='box', x='children', y='charges').set_xticklabels(rotation=90)
df.columns
g=sns.catplot(data=df, kind='box', x='region', y='charges')