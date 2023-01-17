# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import xgboost as xg

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/diabetes.csv")
df.head()
df.describe()
norm_df = StandardScaler()

norm_df.fit(df)
df.head()
X = df.loc[:,'Pregnancies':'Age']

y= df['Outcome']
fig  = plt.figure(figsize=(12,20))

X.hist(bins=50 , figsize=(12,8))

plt.show()
plt.bar(y , height=6)