import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

plt.style.use('fivethirtyeight')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df=pd.read_csv('/kaggle/input/my-dataset/credit_train.csv')

test_df=pd.read_csv('/kaggle/input/my-dataset/credit_test.csv')
train_df.head()
train_df.tail()
train_df.info()
train_df.columns
print('(number of rows, number of columns) : ',train_df.shape)
train_df.dtypes
train_df.isnull().sum()
train_df.notnull().sum()
object_train_df=train_df.select_dtypes(include=['object'])

object_train_df.columns
num_train_df=train_df.select_dtypes(include=['int','float'])

num_train_df.columns
for i in object_train_df.columns:

    print(train_df[i].value_counts())

    print('-'*40)
for i in object_train_df.columns:

    print(i)

    print(train_df[i].unique())

    print('-'*40)
train_df['Annual Income'].head()
train_df['Annual Income']+train_df['Monthly Debt']
train_df.iloc[2,5]
train_df.describe()
train_df.corr()
sns.heatmap(train_df.corr())
print('variance of each column')

train_df.var()
cols_to_drop=['Loan ID','Customer ID']

train_df=train_df.drop(cols_to_drop,axis=1)

train_df.columns
col_mean=train_df['Monthly Debt'].mean()

train_df['Monthly Debt']=train_df['Monthly Debt'].fillna(col_mean)

train_df['Monthly Debt'].isnull().sum()
train_df=train_df.dropna()

train_df.shape
mapping_dict = {

    "Years in current job": {

        "10+ years": 10,

        "9 years": 9,

        "8 years": 8,

        "7 years": 7,

        "6 years": 6,

        "5 years": 5,

        "4 years": 4,

        "3 years": 3,

        "2 years": 2,

        "1 year": 1,

        "< 1 year": 0,

        "n/a": 0

    }

}

train_df=train_df.replace(mapping_dict)

train_df['Years in current job']=train_df['Years in current job'].astype('int')

train_df['Years in current job'].head()
train_df.rename(columns={'Years in current job':'Years_in_current_job'},inplace=True)

train_df.columns
all_data=pd.concat([object_train_df,num_train_df],axis=1)

all_data.head()
train_df['Purpose'].value_counts().plot.bar()
train_df['Years of Credit History'].plot.hist()
train_df['Years of Credit History'].plot.kde(label='distribution')

plt.axvline(train_df['Years of Credit History'].mean(),color='red',label='mean')

plt.legend(loc='best')

plt.show()