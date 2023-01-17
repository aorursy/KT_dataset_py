import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# list files in directory

!ls -lh "../input"
# unzip train.csv, is not needed in the kaggle kernel

#!unzip train.csv.zip
# set chmod to read train.csv, is not needed in the kaggle kernel

#!chmod +r train.csv
# read csv data to pandas data frame

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# check for missing values

assert ~df_train.isnull().values.any()

assert ~df_test.isnull().values.any()



# check for NaN values

assert ~df_train.isna().values.any()

assert ~df_test.isna().values.any()
# check column name, non-null values and dtypes

print(df_train.info(),'\n')

print(df_test.info())
# have a look at the first rows

df_train.head()
df_test.head()
# cast every column to object and get the unique elements

print(df_train.astype('object').describe(include='all').loc['unique', :],'\n')

print(df_test.astype('object').describe(include='all').loc['unique', :])
# get unique values of every column

col_values = [col for col in df_train]

unique_col_values_train = [df_train[col].unique() for col in df_train]

unique_col_values_test = [df_test[col].unique() for col in df_test]
# check if there are difference in the columns and if print them

for i, (c, a, b) in enumerate(zip(col_values, unique_col_values_train, unique_col_values_test)):

    

    if i == 0: continue # skip id_code

        

    a = set(a)

    b = set(b)

    

    print('\n'+c+':', a == b)

    

    # if the column elements are not equal, check if they are disjoint

    if not(a == b):

        print('disjoint:', a.isdisjoint(b))
sns.catplot(x='experiment', kind='count', data=df_train, height=3, aspect=10);
sns.catplot(x='plate', hue='experiment', kind='count', data=df_train, height=4, aspect=5);
sns.catplot(y='well', kind='count', data=df_train, height=40, aspect=0.25);
sns.catplot(y='well', hue='experiment', kind='count', data=df_train, height=150, aspect=0.1);
sns.catplot(y='sirna', hue='experiment', kind='count', data=df_train, height=400, aspect=0.05);
sns.catplot(x='experiment', kind='count', data=df_test, height=3, aspect=10);
sns.catplot(x='plate', hue='experiment', kind='count', data=df_test, height=4, aspect=5);
sns.catplot(y='well', kind='count', data=df_test, height=40, aspect=0.25);
sns.catplot(y='well', hue='experiment', kind='count', data=df_test, height=150, aspect=0.1);