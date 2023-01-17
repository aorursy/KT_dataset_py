# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.cluster import AgglomerativeClustering

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/mall-customers/Mall_Customers.csv')
df.head()
df.dtypes
df=df.rename(columns={'Genre':'Gender'})
df.dtypes
df=pd.get_dummies(data=df,columns=['Gender'])
df=df.drop('Gender_Male',axis=1)
df.head()
df.isnull().sum()
df=df.drop('CustomerID',axis=1)
sns.scatterplot(x='Age',y='Spending Score (1-100)',data=df)
sns.scatterplot(x='Annual Income (k$)',y='Spending Score (1-100)',data=df)
sns.scatterplot(x='Gender_Female',y='Spending Score (1-100)',data=df)
sns.distplot(df['Spending Score (1-100)'])
sns.boxplot(df['Spending Score (1-100)'])
corr=df.corr()

ax=sns.heatmap(corr,annot=True)

def spending_bin(a):

    if(a in range(1,10)):

        return '1'

    elif(a in range(10,20)):

        return '2'

    elif(a in range(20,30)):

        return '3'

    elif(a in range(30,40)):

        return '4'

    elif(a in range(40,50)):

        return '5'

    elif (a in range(50,60)):

        return '6'

    elif (a in range(60,70)):

        return '7'

    elif (a in range(70,80)):

        return '8'

    elif (a in range(80,90)):

        return '9'

    

list5=list(map(spending_bin,df['Spending Score (1-100)']))

df['Spending_Score_bin']=list5

df.head()
sns.countplot(df['Spending_Score_bin'])
# Creating seperate dataframes for male and female

df_f=df[df['Gender_Female']==1]

df_m=df[df['Gender_Female']==0]
sns.countplot(df_f['Spending_Score_bin'])
df_female=df_f['Spending_Score_bin'].value_counts().sort_index(ascending=False)
sns.countplot(df_m['Spending_Score_bin'])
df_male=df_m['Spending_Score_bin'].value_counts().sort_index(ascending=False)
df_female
df_male

sns.barplot(x=df_m['Spending_Score_bin'].sort_values(),y=df_m['Annual Income (k$)'])
sns.barplot(x=df_f['Spending_Score_bin'].sort_values(),y=df_f['Annual Income (k$)'])
df=df.drop(['Spending_Score_bin'],axis=1)
plt.figure(figsize=[10,10])

merg = linkage(df, method='ward')

dendrogram(merg, leaf_rotation=90)

plt.title('Dendrogram')

plt.xlabel('Data Points')

plt.ylabel('Euclidean Distances')

plt.show()
hie_clus = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')

cluster2 = hie_clus.fit_predict(df)



df_h = df.copy(deep=True)

df_h['label'] = cluster2
sns.pairplot(df_h,hue='label',palette='husl')
hie_clus = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')

cluster2 = hie_clus.fit_predict(df)



df_h = df.copy(deep=True)

df_h['label'] = cluster2
sns.pairplot(df_h,hue='label',palette='husl')