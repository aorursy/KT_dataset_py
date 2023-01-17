# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# for plotting purposes

from matplotlib import pyplot as plt

#pip install seaborn / conda install seaborn

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/telecom_churn.csv')
df.head()
print(df.shape)
df['International plan'] = df['International plan'].map({'Yes': 1, 'No': 0})
df['Voice mail plan'] = df['Voice mail plan'].map({'Yes': 1, 'No': 0})
df['Churn'] = df['Churn'].astype('int64');
df.head()
#histograms

#correcting figure size



plt.rcParams['figure.figsize'] = (16,12)

df.drop(['State'], axis = 1).hist();
df.corr()  #correlation matrix
sns.heatmap(df.corr());
# checking for feature names having 'charge'



[feat_name for feat_name in df.columns

if 'charge' in feat_name]
# drop 'charge' from dataset

df.drop([feat_name for feat_name in df.columns

if 'charge' in feat_name], axis=1)
#checking shape of the modified dataset

df.drop([feat_name for feat_name in df.columns

if 'charge' in feat_name], axis=1).shape
#checking initial dataset

#the initial or actual dataset did not change

df.shape
#if we want the dataset to be modified (inplace=True helps in acheiving this)

df.drop([feat_name for feat_name in df.columns

if 'charge' in feat_name], axis=1, inplace=True)
#And the dataset got modified

df.shape
# Features one at a time

# Numeric features



df['Total day minutes'].describe()
plt.rcParams['figure.figsize'] = (8,6)

sns.boxplot(x='Total day minutes', data=df);
plt.rcParams['figure.figsize'] = (8,6)

df['Total day minutes'].hist();
df['State'].nunique()
df['State'].value_counts().head()
df['Churn'].value_counts()
# use normalize to get the percentage values

df['Churn'].value_counts(normalize=True)
sns.countplot(x='Churn', data=df);
states = df['State']

df.drop('State', axis = 1, inplace = True)
# Correlation of 'Total day minutes' with other numerical values in the dataset

df.corrwith(df['Total day minutes'])
plt.scatter(df['Total day minutes'], df['Customer service calls']);
pd.crosstab(df['Churn'], df['Customer service calls'])
sns.countplot(x = 'Customer service calls', hue = 'Churn', data = df);

plt.title('Customer Service Calls for Loyal & Churned');
from sklearn.manifold import TSNE
df.shape
#state = df['State']

#df.drop('State', axis = 1, inplace = True)

df.head()
tsne = TSNE(random_state=17)
%%time

X_repr = tsne.fit_transform(df)
X_repr.shape
#whole dataset

plt.rcParams['figure.figsize'] = (8,6)

plt.scatter(X_repr[:, 0], X_repr[:, 1]);
#Churned Customers

plt.rcParams['figure.figsize'] = (8,6)

plt.scatter(X_repr[df['Churn']==1, 0], 

            X_repr[df['Churn']==1, 1]);

                   
#Loyal Customers

plt.rcParams['figure.figsize'] = (8,6)

plt.scatter(X_repr[df['Churn']==0, 0], 

            X_repr[df['Churn']==0, 1]);
#Overlapping plot



plt.rcParams['figure.figsize'] = (10,8)

plt.scatter(X_repr[df['Churn']==1, 0], 

            X_repr[df['Churn']==1, 1], alpha = 0.5, c='red', 

            label='Churn')

plt.scatter(X_repr[df['Churn']==0, 0], 

            X_repr[df['Churn']==0, 1], alpha = 0.5, c='green', 

            label='Loyal');

plt.xlabel('TSNE axis #1');

plt.ylabel('TSNE axis #2');

plt.legend();

plt.title('TSNE representation');

plt.savefig('churn_tsne.png', dpi=300)
#<img src = 'churn_tsne.png'>
df.groupby('Churn')['Total day minutes',

                   'Customer service calls'].mean()
df.groupby('Churn')['Total day minutes',

                   'Customer service calls'].agg(np.median)
df.groupby('Churn')['Total day minutes',

                   'Customer service calls'].agg([np.median, np.std])
sns.boxplot(x='Churn', y = 'Total day minutes', data = df, hue = 'Churn');