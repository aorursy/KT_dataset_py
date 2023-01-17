# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
athlete_df = pd.read_csv('../input/athlete_events.csv')
athlete_df.head()
athlete_df.info()
athlete_df.describe(include='all')
#data missing in height, weight, age
athlete_df.columns
athlete_df.isnull().sum()
#check for null values
sns.countplot(data=athlete_df, x = 'Medal', label='Count')
sns.countplot(data=athlete_df, x = 'Sex', label='Count')

M, F = athlete_df['Sex'].value_counts()
print('Number of players that are male: ',M)
print('Number of players that are female: ',F)
sns.factorplot(x="Age", y="Sex", hue="Medal", data=athlete_df);
#age of wining medal
sns.countplot(data=athlete_df, x = 'Medal', label='Count')

X, Y, Z = athlete_df['Medal'].value_counts()
print('Number of GOLD medals: ',X)
print('Number of SILVER medals: ',Y)
print('Number of BRONZE medals: ',Z)
sns.countplot(data=athlete_df, x = 'Season', label='Count')

X, Y= athlete_df['Season'].value_counts()
print('Number of Summer Games: ',X)
print('Number of Winter Games: ',Y)
