import pandas as pd

from pandas import Series,DataFrame
df_titanic=pd.read_csv("../input/train.csv")
df_titanic.head()
df_titanic.info()
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.factorplot(x="Pclass", y="Survived", col="Sex",data=df_titanic, saturation=.5, kind="bar", ci=None, aspect=.6)
def male_female_child(passenger):

    age,sex=passenger

    if age < 16:

        return "child"

    else:

        return sex
df_titanic["person"]=df_titanic[["Age","Sex"]].apply(male_female_child,axis=1)
sns.factorplot(x="Pclass", y="Survived", col="person",data=df_titanic, saturation=.5, kind="bar", ci=None, aspect=.6)
df_titanic["Age"].hist(bins=60)
df_titanic["Age"].mean()
df_titanic["person"].value_counts()
fig=sns.FacetGrid(df_titanic,hue="Sex",aspect=4)

fig.map(sns.kdeplot,"Age",shade=True)

oldest=df_titanic["Age"].max()

fig.set(xlim=(0,oldest))

fig.add_legend()
fig=sns.FacetGrid(df_titanic,hue="person",aspect=4)

fig.map(sns.kdeplot,"Age",shade=True)

oldest=df_titanic["Age"].max()

fig.set(xlim=(0,oldest))

fig.add_legend()
fig=sns.FacetGrid(df_titanic,hue="Pclass",aspect=4)

fig.map(sns.kdeplot,"Age",shade=True)

oldest=df_titanic["Age"].max()

fig.set(xlim=(0,oldest))

fig.add_legend()
fig=sns.FacetGrid(df_titanic,hue="Pclass",aspect=4)

fig.map(sns.kdeplot,"Age",shade=True)

oldest=df_titanic["Age"].max()

fig.set(xlim=(0,oldest))

fig.add_legend()
deck=df_titanic["Cabin"].dropna()
levels=[]

for level in deck:

    levels.append(level[0])

    

df_cabin=DataFrame(levels)

df_cabin.columns=["Cabin"]

    

sns.factorplot(x="Cabin",data=df_cabin,kind="count",palette="winter_d")
df_cabin=df_cabin[df_cabin.Cabin != "T"]

sns.factorplot(x="Cabin",data=df_cabin,kind="count",palette="summer")
sns.factorplot(x="Embarked",data=df_titanic,hue="Pclass",kind="count",order=["C","Q","S"])
df_titanic["Alone"]=df_titanic.SibSp + df_titanic.Parch

df_titanic["Alone"].loc[df_titanic["Alone"]>0]="With family"

df_titanic["Alone"].loc[df_titanic["Alone"]==0]="Alone"
sns.factorplot(x="Alone",data=df_titanic,kind="count",palette="Blues")
df_titanic["Survivor"]=df_titanic.Survived.map({0:"no",1:"yes"})

sns.factorplot(x="Survivor",data=df_titanic,kind="count",palette="Set1")
sns.factorplot("Pclass","Survived",data=df_titanic,hue="person",palette="Set1")
sns.lmplot("Age","Survived",data=df_titanic)
sns.lmplot("Age","Survived",hue="Pclass",data=df_titanic)
generations=[10,20,40,60,80]

sns.lmplot("Age","Survived",hue="Pclass",data=df_titanic,palette="winter",x_bins=generations)
sns.lmplot("Age","Survived",hue="Sex",data=df_titanic,palette="winter",x_bins=generations)