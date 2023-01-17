# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt # for data visualization purposes



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv(filepath_or_buffer='../input/beauty.csv')

#for Jupyter Notebook

#df = pd.read_csv(filepath_or_buffer='../input/beauty.csv', sep=';')
type(df)
df.head()
df['wage']
type(df['wage'])
df['wage'].head()
#number of rows and columns

df.shape
#column names

df.columns
#column info

df.info()
#showing simple statistics of data columns

df.describe()
#plotting with dataframes

df['wage'].hist()
plt.figure(figsize=(20,10))

df.hist()
df['female'].unique()

#men & women
df['female'].nunique()

#male & female
df['female'].value_counts()

# 0 - men (count 824), 1 - women/female (count 436)
df['looks'].nunique()
df['looks'].value_counts()
df['goodhlth'].value_counts(normalize=True)
df.iloc[0,5]
#data slices

df.iloc[:6, 5:7]
# making a new dataframe



toy_df = pd.DataFrame({'age': [17,32,56],

                       'salary': [56,69,120]},

                        index=['Kate', 'Leo', 'Max'])
toy_df
toy_df.iloc[1,0]
# .loc



toy_df.loc[['Leo', 'Max'], 'age']
df[df['wage'] > 40]
df[df['wage'] > 40]
df[(df['wage'] > 10) & (df['female'] == 1)]
def gender_id_to_str(gender_id):

    if gender_id == 1:

        return 'female'

    elif gender_id == 0:

        return 'male'

    else:

        return 'Wrong Input'
df['female'].apply(gender_id_to_str).head()
df['female'].apply(lambda gender_id :

                 'female' if gender_id == 1 

                 # elif gender_id == 0 'male'

                  else 'male').head()
df['female'].map({0: 'male', 1: 'female'}).head()
df.loc[df['female'] == 0, 'wage'].median()
df.loc[df['female'] == 1, 'wage'].median()
for (gender_id, sub_dataframe) in df.groupby('female'):

    print(gender_id)

    print(sub_dataframe.shape)
for (gender_id, sub_dataframe) in df.groupby('female'):

    print('Median wages for {} are {}'.format('men' if gender_id == 0

                                                      else 'women',                                         

                                         sub_dataframe['wage'].median()))
df.groupby('female')['wage'].median()
df.groupby(['female', 'married'])['wage'].median()
pd.crosstab(df['female'],df['married'])
# pip install seaborn / conda install seaborn

import seaborn as sns
df['educ'].nunique()
df['educ'].value_counts()
sns.boxplot(x='wage', data=df)
sns.boxplot(x='wage', data=df[df['wage'] < 30]);

#restricting the outliers
sns.boxplot(x='educ', y='wage', data=df[df['wage'] < 30]);