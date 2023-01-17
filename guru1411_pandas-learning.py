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
df = pd.read_csv("../input/train.csv")
df.columns
df['PassengerId'].head(10)
df[['Name', 'Sex']].head(10)
df['Survived'].value_counts()
df['Sex'].value_counts()
df['Pclass'].value_counts()
df['Age'].mean()
df['Fare'].max()
df['Fare'].min()
df['Fare'].mean()
df['Embarked'].value_counts()
df.shape
ml_df = df[df['Sex'] == 'male']
len(ml_df)
df_age_above_50 = df[df['Age'] > 50]
df_ml_age_above_50 = df[(df['Age'] > 50) & (df['Sex'] == 'male')]
df_ml_age_above_50
df_ml_age_between_30_and_50 = df[(df['Age'] <= 50) & (df['Age'] >= 30) & (df['Sex'] == 'male')]
df_fml_age_between_30_and_50 = df[(df['Age'] <= 50) & (df['Age'] >= 30) & (df['Sex'] == 'female')]
df_embarked_s_and_female = df[(df['Embarked'] == 'S') & (df['Sex'] == 'female')]
len(df_ml_age_between_30_and_50)
len(df_fml_age_between_30_and_50)
len(df_embarked_s_and_female)
df['Embarked'].isnull().value_counts()
df['Embarked'] = df['Embarked'].fillna("S")
df['Embarked'].isnull().value_counts()
df['Age'].isnull().value_counts()
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Age'].isnull().value_counts()
df.columns[df.isnull().any()]
df['Cabin'].isnull().value_counts()
df1 = pd.read_csv("../input/train.csv")
df1.columns[df1.isnull().any()]
df1['Age'].isnull().value_counts()
df1['Embarked'].isnull().value_counts()
df1['Cabin'].isnull().value_counts()
tmp_df_row = df1.dropna(how = 'any')
len(tmp_df_row)
tmp_df_row_all = df1.dropna(how = 'all')
len(tmp_df_row_all)
#Remove rows which has null value in a specific column



tmp_df_row_subset_age = df1.dropna(subset = ['Age'])
len(tmp_df_row_subset_age)
tmp_df_row_subset_age['Age'].isnull().value_counts()
tmp_df_row_subset_embarked = df1.dropna(subset = ['Embarked'])
len(tmp_df_row_subset_embarked)
tmp_df_row_subset_embarked['Embarked'].isnull().value_counts()
tmp_df_row_subset_cabin = df1.dropna(subset = ['Cabin'])
len(tmp_df_row_subset_cabin)
tmp_df_row_subset_cabin['Cabin'].isnull().value_counts()
df1['Cabin'].isnull().value_counts()
tmp_df_col = df1.drop(['Cabin'], axis = 1)
tmp_df_col
tmp_df = df1
tmp_df['Embarked'].isnull().value_counts()
tmp_df['Embarked'] = tmp_df['Embarked'].fillna("S")
tmp_df['Embarked'].isnull().value_counts()
tmp_df['Age'].isnull().value_counts()
tmp_df['Age'] = tmp_df['Age'].fillna(tmp_df['Age'].mean())
tmp_df['Age'].isnull().value_counts()