# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
corona=pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")
df=corona.copy()
df.head()
df.info()
df.isnull().sum()
df["Province/State"].fillna("Unknown",inplace=True)
df.isnull().sum()
df.dtypes
df["Country"]=df["Country"].astype("category")

df["Province/State"]=df["Province/State"].astype("category")
df.dtypes
df.groupby("Country").mean()
df["Country"].value_counts()
sns.set_style("whitegrid")
sns.set(rc={"figure.figsize":(16,5)})

df["Country"].value_counts().plot.bar(color="red")
sns.set(rc={"figure.figsize":(16,12)})

df["Country"].value_counts().plot.barh(color="red")
ax=sns.barplot(x="Confirmed",y="Country",data=df,ci=None,color="yellow",alpha=0.7)

ax=sns.barplot(x="Deaths",y="Country",data=df,ci=None,color="red",alpha=0.7)

ax=sns.barplot(x="Recovered",y="Country",data=df,ci=None,color="blue",alpha=0.7)

ax.set(xlabel='Confirmed/Deaths/Recovered', ylabel='Country')
ax=sns.catplot(x="Province/State",y="Confirmed",data=df,aspect=4)

ax.set_xticklabels(rotation=90)
ax=sns.catplot(x="Province/State",y="Confirmed",data=df,aspect=4,kind="bar")

ax.set_xticklabels(rotation=90)
sns.lmplot(x="Confirmed",y="Deaths",data=df,aspect=3)
sns.set(rc={"figure.figsize":(15,8)})

sns.jointplot(x="Confirmed",y="Deaths",data=df,kind="reg")
df.head()
sns.pairplot(df,vars=["Confirmed","Deaths","Recovered"],aspect=2.5,kind="reg")
sns.distplot(df["Deaths"],hist=False)
sns.kdeplot(df["Deaths"],shade=True)
sns.FacetGrid(df,hue="Country",height=8,xlim=(0,30),ylim=(0,1.25)).map(sns.kdeplot,"Confirmed",shade=True).add_legend()