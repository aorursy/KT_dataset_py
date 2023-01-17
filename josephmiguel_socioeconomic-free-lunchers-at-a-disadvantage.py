# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/StudentsPerformance.csv")
# I don't like spaces in my column names, lets fix that
df.columns = df.columns.map(lambda x: x.replace(' ', '_'))
#lets see the distinct values for each of the categorical columns
print(df['race/ethnicity'].unique().tolist())
print(df['parental_level_of_education'].unique().tolist())
print(df['lunch'].unique().tolist())
print(df['test_preparation_course'].unique().tolist())
# i don't like the education field, let's create some order to it. converting these over will make them play nicer with sns
mapper = {'some high school':0, 'high school':1, 'some college':2, "associate's degree":3, "bachelor's degree":4, "master's degree":5}
df['parental_level_of_education'] = df['parental_level_of_education'].map(mapper)

mapper = {'none':0, 'completed':1}
df['test_preparation_course'] = df['test_preparation_course'].map(mapper)

mapper = {'standard':1, 'free/reduced':0}
df['lunch'] = df['lunch'].map(mapper)

mapper = {'group B':1, 'group C':2, 'group A':0, 'group D':3, 'group E':4}
df['race/ethnicity'] = df['race/ethnicity'].map(mapper)

mapper = {'male':1, 'female':0}
df['gender'] = df['gender'].map(mapper)

df['avg_score'] = (df['math_score'] + df['reading_score'] + df['writing_score'])/3
df['avg_score_decile'] = (df['avg_score']/10).astype('int')
# example of some items
df.head()
# check for missing values - there are no missing values
df[(df.isnull().any(axis=1)) | (df.isna().any(axis=1))]
# lets see the size of the dataset
df.shape
# get the general math stats
df.describe()
# lets see a pairplot for a high level view
sns.pairplot(df)
# notice the high correction between the three scores
sns.heatmap(df.corr(), cmap="Blues",annot=True,annot_kws={"size": 7.5},linewidths=.5)

# scores
fig, ax = plt.subplots(1,3)
fig.set_size_inches(20, 5)
sns.countplot(x="math_score", data = df, palette="muted", ax=ax[0])
sns.countplot(x="reading_score", data = df, palette="muted", ax=ax[1])
sns.countplot(x="writing_score", data = df, palette="muted", ax=ax[2])
plt.show()
print('math', str(Counter(df[df.math_score > 90].math_score)))
print('reading', str(Counter(df[df.reading_score > 90].reading_score)))
print('writing', str(Counter(df[df.writing_score > 90].writing_score)))

fig, ax = plt.subplots(1,3)
fig.set_size_inches(20, 5)
sns.distplot(df[df.gender == 0]['math_score'], ax=ax[0], color='blue')
sns.distplot(df[df.gender == 1]['math_score'], ax=ax[0], color='red')
sns.distplot(df[df.gender == 0]['reading_score'], ax=ax[1], color='blue')
sns.distplot(df[df.gender == 1]['reading_score'], ax=ax[1], color='red')
sns.distplot(df[df.gender == 0]['writing_score'], ax=ax[2], color='blue')
sns.distplot(df[df.gender == 1]['writing_score'], ax=ax[2], color='red')
plt.show()
fig, ax = plt.subplots(1,3)
fig.set_size_inches(20, 5)
sns.distplot(df[df.gender == 0]['math_score'],    ax=ax[0], color='blue')
sns.distplot(df[df.gender == 1]['math_score'],    ax=ax[0], color='red')
sns.distplot(df[df.gender == 0]['reading_score'], ax=ax[1], color='blue')
sns.distplot(df[df.gender == 1]['reading_score'], ax=ax[1], color='red')
sns.distplot(df[df.gender == 0]['writing_score'], ax=ax[2], color='blue')
sns.distplot(df[df.gender == 1]['writing_score'], ax=ax[2], color='red')
plt.show()
fig, ax = plt.subplots(1,3)
fig.set_size_inches(20, 5)
sns.distplot(df[df['test_preparation_course'] == 0]['math_score'], ax=ax[0], color='black')
sns.distplot(df[df['test_preparation_course'] == 1]['math_score'], ax=ax[0], color='green')
sns.distplot(df[df['test_preparation_course'] == 0]['reading_score'], ax=ax[1], color='black')
sns.distplot(df[df['test_preparation_course'] == 1]['reading_score'], ax=ax[1], color='green')
sns.distplot(df[df['test_preparation_course'] == 0]['writing_score'], ax=ax[2], color='black')
sns.distplot(df[df['test_preparation_course'] == 1]['writing_score'], ax=ax[2], color='green')
plt.show()
fig, ax = plt.subplots(1,3)
fig.set_size_inches(20, 5)
sns.violinplot(y="math_score",    x='parental_level_of_education', data = df, palette="muted", ax=ax[0], )
sns.violinplot(y="reading_score", x='parental_level_of_education', data = df, palette="muted", ax=ax[1])
sns.violinplot(y="writing_score", x='parental_level_of_education', data = df, palette="muted", ax=ax[2])
plt.show()
fig, ax = plt.subplots(1,3)
fig.set_size_inches(20, 5)
sns.violinplot(y="math_score",    x='race/ethnicity', data = df, palette="muted", ax=ax[0])
sns.violinplot(y="reading_score", x='race/ethnicity', data = df, palette="muted", ax=ax[1])
sns.violinplot(y="writing_score", x='race/ethnicity', data = df, palette="muted", ax=ax[2])
plt.show()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import *
X = df[['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']]
y = df['avg_score']
regressor = RandomForestRegressor(oob_score=True)
regressor.fit(X,y)
print(X.columns.tolist())
print(regressor.feature_importances_)
sns.countplot(x='parental_level_of_education', data=df, hue='lunch')
sns.countplot(x='avg_score_decile', data=df, hue='lunch')
sns.countplot(x='avg_score_decile', data=df, hue='test_preparation_course')
sns.countplot(x='parental_level_of_education', data=df, hue='lunch')



