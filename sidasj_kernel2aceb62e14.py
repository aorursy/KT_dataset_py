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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv("../input/finance-india/Aggregate_Expenditure.csv",encoding ='latin1')

df
rows,columns = df.shape

print("Rows: ",rows)

print("Columns: ",columns)
for i in range(columns):

    print(str(i)+". :", df.iloc[:,i].dtype)
df.isna().sum()
df.replace({"":0},inplace=True)
df
for i in range(columns):

    print(str(i)+". :", df.iloc[:,i].dtype)
lst = []

col = df.columns

print(col)

for i in col:

    if i!="State":

        lst.append(i)

print(lst)
df[lst] = df[lst].apply(pd.to_numeric)
for i in range(columns):

    print(str(i)+". :", df.iloc[:,i].dtype)
plt.figure(figsize=(30,10))

sns.heatmap(df.corr(),annot=True,linewidth=1)
plt.figure(figsize=(30,10))

sns.barplot(x='1980-81',y="State",data=df)
for i in lst:

    plt.xlabel(i)

    plt.ylabel("State")

    plt.title(i+" vs State")

    plt.figure(figsize=(30,10))

    sns.barplot(x=i,y="State",data=df)
for i in lst:

    plt.xlabel(i)

    plt.ylabel("State")

    plt.title(i+" vs State")

    plt.figure(figsize=(30,10))

    ax = sns.barplot(x="State",y = i,data=df)

    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

    #ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

    #plt.tight_layout()

    plt.show()
for i in lst:

    plt.xlabel(i)

    plt.ylabel("State")

    plt.title(i+" vs State")

    plt.figure(figsize=(30,10))

    g = sns.catplot(x = i, y= "State",kind = "bar",  height=6, aspect=11.7/6, data = df) # distribution.set_xticklabels(rotation=65,   horizontalalignment='right')

    g.set_xticklabels(rotation=90)
for i in lst:

    plt.xlabel(i)

    plt.ylabel("State")

    plt.title(i+" vs State")

    plt.figure(figsize=(30,10))

    sns.catplot(x = "State", y= i,kind = "bar",  height=6, aspect=11.7/6, data = df) # distribution.set_xticklabels(rotation=65,   horizontalalignment='right')

    plt.xticks(plt.xticks()[0], rotation=90)

    plt.tight_layout()

    plt.show()
for i in lst:

    plt.xlabel(i)

    plt.ylabel("State")

    plt.title(i+" vs State")

    plt.figure(figsize=(30,10))

    g = sns.catplot(x = "State", y= i,kind = "bar",  height=6, aspect=11.7/6, data = df) # distribution.set_xticklabels(rotation=65,   horizontalalignment='right')

    g.set_xticklabels(rotation=90)

#     plt.xticks(plt.xticks()[0], rotation=90)

#     plt.tight_layout()

#     plt.show()
!pip install ptitprince

import ptitprince as pt
for i in lst:

    plt.xlabel(i)

    plt.ylabel("State")

    plt.title(i+" vs State")

    plt.figure(figsize=(30,10))

    pt.RainCloud(x = "State", y= i, data = df) # distribution.set_xticklabels(rotation=65,   horizontalalignment='right')

#     plt.xticks(plt.xticks()[0], rotation=90)

#     plt.tight_layout()

#     plt.show()
for i in lst:

    plt.xlabel(i)

    plt.ylabel("State")

    plt.title(i+" vs State")

    plt.figure(figsize=(30,10))

    ax = sns.boxplot(x="State",y = i,data=df)

    #ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

    #plt.tight_layout()

    plt.show()
df
#Pca

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

scalar = StandardScaler()

dr = scalar.fit_transform(df.iloc[:,1:])

print(dr)

pca = PCA(n_components=2)

dr_pca = pca.fit_transform(dr)

dr_pca
dr_pca_dataframe = pd.DataFrame(dr_pca,columns=["pca_column_1","pca_column_2"])

dr_pca_dataframe
df1 = pd.concat([dr_pca_dataframe,df.iloc[:,0]],axis=1).reset_index()
df1
df1.drop("index",axis=1,inplace=True)
df1
sns.pairplot(df1)
df1
df1