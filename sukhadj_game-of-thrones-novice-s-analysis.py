import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv('../input/character-deaths.csv')
df.head()
df.tail()
df.shape
df.describe()
df.isnull().any()
df["Dead"]=df["Death Year"].map({np.nan:0,np.float:1})

df["Dead"].fillna(1,inplace=True)
df
df["No of Books"]=df["GoT"]+df["CoK"]+df["SoS"]+df["FfC"]+df["DwD"]

df.drop(labels=["GoT","CoK","SoS","FfC","DwD"],axis=1)
df["Allegiances"].unique()
def house(x):

    if x in ['None',"Wildling","Night's Watch"]:

        return x

    elif "House" not in x:

        return "House "+str(x)

    else:

        return x
df["Allegiances"]=df["Allegiances"].apply(house)

df["Allegiances"].unique()
df2=df[["Dead","Allegiances"]].groupby(by="Allegiances",as_index=False).mean().sort_values(by="Dead",ascending=True)

df2
fig,ax=plt.subplots()

width=0.35

rect=ax.bar(df2["Allegiances"],df2["Dead"],color="#6B0FFF")

plt.xticks(rotation=90);
plt.figure(figsize=(15,5))

sns.set_context(font_scale=2)

sns.violinplot(x='Allegiances',y='Dead',data=df,hue='Gender',split=True)

plt.xticks(rotation=90);
df2=df[["Dead","Gender"]].groupby(by="Gender",as_index=False).mean().sort_values(by="Dead",ascending=True)

df2
fig,ax=plt.subplots()

width=0.5

#rect=ax.hist(df2,bins=20)

rect=ax.bar(df2["Gender"].map({0:"female",1:"male"}),df2["Dead"],color="#DF5BC2")

plt.xticks(rotation=0);
sns.barplot(x='Gender',y='Dead',data=df,hue="Nobility",palette="viridis")
df2=df[["Dead","Nobility"]].groupby(by="Nobility",as_index=False).mean().sort_values(by="Dead",ascending=True)

df2
fig,ax=plt.subplots()

rect=ax.bar(df2["Nobility"].map({0:"Not Noble",1:"Noble"}),df2["Dead"],color="#123456")

df2=df[["Dead","No of Books"]].groupby(by="No of Books",as_index=False).mean().sort_values(by="Dead",ascending=True)

df2
fig,ax=plt.subplots()

rect=ax.bar(df2["No of Books"],df2["Dead"],color="#650056")