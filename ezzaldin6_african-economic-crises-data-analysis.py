# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
african_crises_df=pd.read_csv('/kaggle/input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')

print('Dataset rows: ',african_crises_df.shape[0])

print('Dataset columns: ',african_crises_df.shape[1])

print('column names:\n',african_crises_df.columns)
def explore_df(df):

    print('Data type of each column:\n')

    print(african_crises_df.dtypes)

    cer_cols=african_crises_df.select_dtypes(include=['object','int']).columns

    for i in cer_cols:

            print('Column name: ',i,'\n')

            print(african_crises_df[i].value_counts().sort_values())

    print('missing values in each column\n')

    print(african_crises_df.isnull().sum())

explore_df(african_crises_df)
cols_to_drop=['case','cc3']

cleaned_df=african_crises_df.drop(cols_to_drop,axis=1)

labels=pd.Categorical(cleaned_df['banking_crisis'])

cleaned_df['banking_crisis']=labels.codes

cleaned_df['currency_crises']=cleaned_df['currency_crises'].replace(2,np.nan)

cleaned_df=cleaned_df.dropna()
cleaned_df['currency_crises']=cleaned_df['currency_crises'].astype(int)
apply_stats_cols=['exch_usd','gdp_weighted_default','inflation_annual_cpi']

cleaned_df[apply_stats_cols].describe()
cleaned_df.corr()
sns.heatmap(cleaned_df.corr())
plt.style.use('fivethirtyeight')

plt.figure(figsize=(25,10))

countries=cleaned_df['country'].unique().tolist()

count=1

for i in countries:

    plt.subplot(3,5,count)

    count+=1

    sns.lineplot(cleaned_df[cleaned_df.country==i]['year'],

                 cleaned_df[cleaned_df.country==i]['exch_usd'],

                 label=i,

                 color='black')

    plt.scatter(cleaned_df[cleaned_df.country==i]['year'],

                cleaned_df[cleaned_df.country==i]['exch_usd'],

                color='black',

                s=28)

    plt.plot([np.min(cleaned_df[np.logical_and(cleaned_df.country==i,cleaned_df.independence==1)]['year']),

              np.min(cleaned_df[np.logical_and(cleaned_df.country==i,cleaned_df.independence==1)]['year'])],

             [0,

              np.max(cleaned_df[cleaned_df.country==i]['exch_usd'])],

             color='green',

             linestyle='dotted',

             alpha=0.8)

    plt.text(np.min(cleaned_df[np.logical_and(cleaned_df.country==i,cleaned_df.independence==1)]['year']),

             np.max(cleaned_df[cleaned_df.country==i]['exch_usd'])/2,

             'Independence',

             rotation=-90)

    plt.scatter(x=np.min(cleaned_df[np.logical_and(cleaned_df.country==i,cleaned_df.independence==1)]['year']),

                y=0,

                s=50)

    plt.title(i)

plt.show()
plt.style.use('fivethirtyeight')

fig=plt.figure(figsize=(25,10))

countries=cleaned_df['country'].unique().tolist()

lst_len=len(countries)

for i in range(lst_len-1):

    ax=fig.add_subplot(2,6,i+1)

    c=cleaned_df[cleaned_df['country']==countries[i]]['domestic_debt_in_default'].value_counts()

    ax.bar(c.index,c.tolist(),color=['yellow','green'],width=0.4)

    ax.set_title(countries[i])

    plt.legend(loc='best')

plt.show()
plt.style.use('fivethirtyeight')

fig=plt.figure(figsize=(25,10))

countries=cleaned_df['country'].unique().tolist()

lst_len=len(countries)

for i in range(lst_len-1):

    ax=fig.add_subplot(2,6,i+1)

    c=cleaned_df[cleaned_df['country']==countries[i]]['sovereign_external_debt_default'].value_counts()

    ax.bar(c.index,c.tolist(),color=['yellow','green'],width=0.4)

    ax.set_title(countries[i])

    plt.legend(loc='best')

plt.show()
fig=plt.figure(figsize=(25,10))

countries=cleaned_df['country'].unique().tolist()

lst_len=len(countries)

for i in range(lst_len-1):

    ax=fig.add_subplot(2,6,i+1)

    c=cleaned_df[cleaned_df['country']==countries[i]]['systemic_crisis'].value_counts()

    ax.bar(c.index,c.tolist(),color=['yellow','green'],width=0.4)

    ax.set_title(countries[i])

    plt.legend(loc='best')

plt.show()
fig=plt.figure(figsize=(25,10))

countries=cleaned_df['country'].unique().tolist()

lst_len=len(countries)

for i in range(lst_len-1):

    ax=fig.add_subplot(2,6,i+1)

    c=cleaned_df[cleaned_df['country']==countries[i]]['banking_crisis'].value_counts()

    ax.bar(c.index,c.tolist(),color=['yellow','green'],width=0.4)

    ax.set_title(countries[i])

    plt.legend(loc='best')

plt.show()
fig=plt.figure(figsize=(25,10))

countries=cleaned_df['country'].unique().tolist()

lst_len=len(countries)

for i in range(lst_len-1):

    ax=fig.add_subplot(2,6,i+1)

    c=cleaned_df[cleaned_df['country']==countries[i]]['currency_crises'].value_counts()

    ax.bar(c.index,c.tolist(),color=['yellow','green'],width=0.4)

    ax.set_title(countries[i])

    plt.legend(loc='best')

plt.show()
fig=plt.figure(figsize=(25,10))

countries=cleaned_df['country'].unique().tolist()

lst_len=len(countries)

for i in range(lst_len-1):

    ax=fig.add_subplot(2,6,i+1)

    c=cleaned_df[cleaned_df['country']==countries[i]]['inflation_crises'].value_counts()

    ax.bar(c.index,c.tolist(),color=['yellow','green'],width=0.4)

    ax.set_title(countries[i])

    plt.legend(loc='best')

plt.show()