# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/habermans-survival-data-set/haberman.csv',names=['Age','OperationYear','No.of.AxillaryNodes','Class'],header=None)
df
df.columns
df.isna().sum()
df['Class'].nunique()
df['Class'].value_counts()
df['Class'] = df['Class'].map({1:'yes',2:'no'})
df.head()
import seaborn as sns

import matplotlib.pyplot as plt
sns.FacetGrid(df,hue='Class',height=5).map(sns.distplot,'Age').add_legend()
sns.FacetGrid(df,hue='Class',height=5).map(sns.distplot,'OperationYear').add_legend()
sns.FacetGrid(df,hue='Class',height=5).map(sns.distplot,'No.of.AxillaryNodes').add_legend()
count,bin_edges = np.histogram(df[df['Class']=='yes']['No.of.AxillaryNodes'],bins=10)

pdf=count/(sum(count))

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)

plt.grid()
count,bin_edges = np.histogram(df[df['Class']=='no']['No.of.AxillaryNodes'],bins=10)

pdf=count/(sum(count))

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)

plt.grid()
sns.boxplot(x='Class',y='Age',data=df)

sns.violinplot(x='Class',y='Age',data=df)
sns.boxplot(x='Class',y='OperationYear',data=df)
sns.violinplot(x='Class',y='OperationYear',data=df)
sns.boxplot(x='Class',y='No.of.AxillaryNodes',data=df)
sns.violinplot(x='Class',y='No.of.AxillaryNodes',data=df)
sns.FacetGrid(data=df,hue='Class',height=7).map(plt.scatter,'Age','No.of.AxillaryNodes')

plt.grid()
sns.pairplot(data=df,hue='Class')
sns.jointplot(x='Age',y='No.of.AxillaryNodes',data=df[df['Class']=='yes'],kind='kde')
sns.jointplot(x='Age',y='No.of.AxillaryNodes',data=df[df['Class']=='no'],kind='kde')
sns.jointplot(x='Age',y='OperationYear',data=df[df['Class']=='yes'],kind='kde')
sns.jointplot(x='Age',y='OperationYear',data=df[df['Class']=='no'],kind='kde')