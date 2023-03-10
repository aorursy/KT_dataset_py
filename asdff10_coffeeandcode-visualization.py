import numpy as np

import pandas as pd

import seaborn as sns
df=pd.read_csv("../input/coffee-and-code/CoffeeAndCodeLT2018.csv")
df.head()
df.info()
df.dtypes
df.nunique()
df.head(2)
sns.set(rc={"figure.figsize":(15,7)},style="whitegrid")
sns.countplot(x="Gender",data=df,palette="brg")
sns.countplot(x="AgeRange",data=df,hue="Gender")
sns.countplot(x="CodingHours",data=df)
sns.countplot(x="CodingHours",data=df,hue="Gender",palette="Pastel1")
sns.countplot(df["CoffeeCupsPerDay"],palette="RdBu_r")
sns.countplot(x="CoffeeCupsPerDay",data=df,hue="Gender",palette="Pastel1")
sns.countplot(df["CoffeeTime"],palette=sns.color_palette("rainbow",7))
sns.countplot(df["CoffeeType"],palette=sns.color_palette("husl",8))
df.head()
sns.catplot(x="CoffeeTime",y="CodingHours",data=df,aspect=2.5,hue="Gender")
sns.catplot(x="CoffeeTime",y="CoffeeCupsPerDay",data=df,aspect=2.5)
sns.catplot(x="CoffeeTime",y="CoffeeCupsPerDay",data=df,aspect=2.5,hue="Gender")
sns.catplot(x="AgeRange",y="CoffeeCupsPerDay",data=df,hue="Gender",aspect=2.5,kind="point")
df.head(2)
sns.boxplot(x="CoffeeTime",y="CoffeeCupsPerDay",data=df)
sns.boxplot(x="AgeRange",y="CodingHours",data=df)
sns.boxplot(x="Gender",y="CodingHours",hue="Gender",data=df,palette="Set1")
sns.kdeplot(df["CodingHours"],shade=True)
sns.FacetGrid(hue="Gender",data=df,aspect=2.5,height=5).map(sns.kdeplot,"CodingHours",shade=True).add_legend()
sns.FacetGrid(hue="AgeRange",data=df,aspect=2.5,height=5).map(sns.kdeplot,"CodingHours",shade=True).add_legend()
sns.FacetGrid(hue="CoffeeTime",data=df,aspect=2.5,height=5).map(sns.kdeplot,"CodingHours",shade=True).add_legend()
sns.lmplot(x="CoffeeCupsPerDay",y="CodingHours",data=df,aspect=2.5)
sns.lmplot(x="CoffeeCupsPerDay",y="CodingHours",data=df,hue="Gender",aspect=2.5)
sns.pairplot(df,aspect=2.5)
sns.pairplot(df,aspect=2.5,hue="Gender")
sns.pairplot(df,aspect=2.5,hue="Gender",kind="reg")