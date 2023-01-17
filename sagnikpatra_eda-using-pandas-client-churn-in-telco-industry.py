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
df = pd.read_csv('../input/telecom_churn.csv')
type(df)
df.head()
df.shape
df.columns
df.info()
df['Churn']=df['Churn'].astype('int64')
df.describe()
df.describe(include=['object'])
df['Churn'].value_counts()
df['Churn'].value_counts(normalize=True)
df.sort_values(by='Total day charge',ascending=False).head()
df.sort_values(by=['Churn','Total day charge'],ascending=[True,False]).head()
df['Churn'].mean()
df[df['Churn']==1].mean()
df[df['Churn']==1]['Total day minutes'].mean()
df[(df['Churn']==0) & (df['International plan']== 'No')] ['Total intl minutes'].max()
df.loc[0:3]
df.iloc[0:3,0:5]
df[-1:]
df.apply(np.max) 
df[df['State'].apply(lambda state:state[0]=='W')].head() 
d= {'Yes':True, 'No':False}

df['International plan']=df['International plan'].map(d) 

df.head() 
df=df.replace({'Voice mail plan':d}) 
df.head() 
columns_to_show=['Total day minutes', 'Total eve minutes', 'Total night minutes']

df.groupby(['Churn'])[columns_to_show].describe(percentiles=[]) 
columns_to_show = ['Total day minutes', 'Total eve minutes', 

                   'Total night minutes']



df.groupby(['Churn'])[columns_to_show].agg([np.mean, np.std, np.min, 

                                            np.max])
pd.crosstab(df['Churn'],df['International plan'])
df.pivot_table(['Total day calls', 'Total eve calls', 'Total night calls'],['Area code'],aggfunc='mean')
total_calls=df['Total day calls']+df['Total eve calls']+df['Total night calls']+df['Total intl calls']

df.insert(loc=len(df.columns),column='Total calls',value=total_calls)
df.head()
df.drop(['Total calls'],axis=1,inplace=True)

df.head()
df.drop([1,2]).head()
pd.crosstab(df['Churn'],df['International plan'],margins=True)
import matplotlib.pyplot as plt

import seaborn as sns

# Graphics in retina format are more sharp and legible

%config InlineBackend.figure_format = 'retina'
sns.countplot(x='International plan', hue='Churn', data=df)
pd.crosstab(df['Churn'],df['Customer service calls'],margins=True)
sns.countplot(x='Customer service calls', hue='Churn', data=df)
df['many_calls']=(df['Customer service calls']>3).astype('int')

pd.crosstab(df['Churn'],df['many_calls'],margins=True)
sns.countplot(x='many_calls', hue='Churn', data=df);
pd.crosstab(df['many_calls'] & df['International plan'] , df['Churn'])