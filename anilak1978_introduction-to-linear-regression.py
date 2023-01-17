# load neccessary pyton libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
# load our dataset and review the first 5 observations

df=pd.read_csv("https://raw.githubusercontent.com/anilak1978/medium-article-data/master/teengamb.csv", index_col=0) #bypass the first column as it is unnamed

df.head()
# look at missing values

df.isnull().sum().sum()
# look at summary statistics

df.info()

df.describe()
# lets look correlation

corr=df.corr()

plt.figure(figsize=(15,10))

sns.heatmap(corr, annot=True)
# looking at distribution for predictor and response variable

plt.figure(figsize=(15,10))

sns.distplot(df["gamble"])

plt.title("Gamble distribution")

plt.figure(figsize=(15,10))

sns.distplot(df["income"])

plt.title("Income distribution")
# looking at distribution for predictor and response variable

plt.figure(figsize=(15,10))

sns.boxplot(x="gamble", y="income", hue="sex", data=df)
# Looking at linear relationship 

plt.figure(figsize=(15,10))

sns.scatterplot(x="income", y="gamble", hue="sex", data=df)

plt.title("Income vs Gamble")
# Linear Regression Line 

plt.figure(figsize=(15,10))

sns.regplot(x="income", y="gamble", data=df)

plt.title("Linear Relationship Between Income and Gamble")
# Load prostatedataset

df_2=pd.read_csv("https://raw.githubusercontent.com/anilak1978/medium-article-data/master/prostate.csv", index_col=0)

df_2.head()
# Look at missing data and statistical summary

df_2.isnull().sum().sum()

df_2.info()

df_2.describe()
# look at correlation

df_2.corr()
plt.figure(figsize=(15,10))

sns.regplot(x="lcavol", y="lpsa", data=df_2)

plt.title("Linear Relationship between Cancer Volume and Prostate Specific Antigen")