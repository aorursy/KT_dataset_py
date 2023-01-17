import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv("../input/pokemon/Pokemon.csv")
df.info()
df.shape
df.head()
df.tail()
df.describe()
df.isnull().any()
df.isnull().sum()
df["Type 2"].fillna("None",inplace=True)
df.head()
df.nunique()
df["Type 1"].value_counts()
df["Type 2"].value_counts()
sns.catplot(x="Type 1",kind="count",data=df,aspect=2);
df["Type 2"].value_counts().plot.bar(color="pink");
df.head()
sns.catplot(x="Type 1",y="Attack",data=df,kind="bar",aspect=2.5);
sns.catplot(x="Type 1",y="Attack",data=df,aspect=2.5);
sns.catplot(x="Type 1",y="Attack",hue="Legendary",data=df,aspect=2.5);
sns.catplot(x="Type 1",y="Attack",hue="Legendary",kind="point",data=df,aspect=2.5);
df.head()
sns.distplot(df["Speed"]);
sns.distplot(df["Speed"],kde=False);
sns.distplot(df["Speed"],hist=False);
sns.kdeplot(df["Speed"],shade=True);
sns.FacetGrid(df,hue="Generation",size=7).map(sns.kdeplot,"Speed").add_legend();
sns.FacetGrid(df,hue="Generation",size=7).map(sns.kdeplot,"Speed",shade=True).add_legend();
sns.FacetGrid(df,hue="Generation",size=7,col="Legendary").map(sns.kdeplot,"Speed",shade=True).add_legend();
df.head()
sns.boxplot(x="Generation",y="Total",data=df);
sns.boxplot(x="Generation",y="Total",hue="Legendary",data=df);
sns.boxplot(x="Legendary",y="Total",data=df,palette="Set3");
sns.boxplot(x="Legendary",y="Total",data=df,hue="Generation",palette="Set3");
df.head()
sns.pairplot(df,vars=["Attack","Sp. Atk","Defense","Sp. Def"],kind="reg");
sns.pairplot(df,vars=["Attack","Sp. Atk","Defense","Sp. Def"],hue="Legendary",kind="reg");
df.head()
sns.lmplot(x="Attack",y="Defense",data=df,hue="Legendary",size=5);
sns.lmplot(x="Attack",y="Defense",data=df,hue="Generation",size=5);
df.head()
sns.set(style="darkgrid")

sns.jointplot(x="Attack",y="Defense",data=df);
sns.jointplot(x="Attack",y="Defense",kind="reg",data=df);
sns.jointplot(x="Attack",y="Defense",kind="hex",size=7,ratio=10,data=df);
df.head()
df_attack=df.pivot_table(index="Type 1",columns="Generation",values="Attack").copy()
df_attack
df_attack.fillna(0,inplace=True)

df_attack
plt.figure(figsize = (8,8))

sns.heatmap(df_attack);
plt.figure(figsize = (8,8))

sns.heatmap(df_attack,linewidths=0.5,annot=True);
df.head()
sns.set_context("poster")

sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(10,8)) 

sns.lineplot(x="Generation",y="HP",data=df);
fig, ax = plt.subplots(figsize=(10,8)) 

sns.lineplot(x="Attack",y="Defense",hue="Legendary",data=df);
fig, ax = plt.subplots(figsize=(10,8)) 

sns.lineplot(x="Attack",y="Defense",style="Legendary",data=df);
fig, ax = plt.subplots(figsize=(10,8)) 

sns.lineplot(x="Attack",y="Defense",style="Legendary",data=df,markers=True);