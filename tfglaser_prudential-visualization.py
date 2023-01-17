import pandas as pd
from pandas import Series,DataFrame

import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

import seaborn as sns
# get training & test csv files as a DataFrame
train_df = pd.read_csv("../input/train.csv" )
test_df    = pd.read_csv("../input/test.csv")

# preview the data
train_df.head()
test_df.head()
fig, axis1 = plt.subplots(1,1,figsize=(20,5))
sns.countplot(x='Response', data=train_df, ax=axis1, order=[1,2,3,4,5,6,7,8])
fig, axis1 = plt.subplots(1,1,figsize=(20,5))
sns.countplot(x='Employment_Info_2', data=train_df, ax=axis1)
fig, axis1 = plt.subplots(1,1,figsize=(20,5))
sns.countplot(x='Employment_Info_2', hue="Response", data=train_df, ax=axis1, hue_order=[1,2,3,4,5,6,7,8])
g = sns.FacetGrid(train_df, col="Response", col_wrap=2, size=5, col_order=[1,2,3,4,5,6,7,8])
g.map(sns.countplot, "Employment_Info_2")
facet = sns.FacetGrid(train_df, hue="Response",aspect=4, hue_order=[1,2,3,4,5,6,7,8], palette="RdBu")
facet.map(sns.kdeplot,'Ins_Age')
facet.set(xlim=(0, train_df['Ins_Age'].max()))
facet.add_legend()
facet = sns.FacetGrid(train_df, hue="Response",aspect=4, hue_order=[1,2,3,4,5,6,7,8], palette="RdBu")
facet.map(sns.kdeplot,'Ht')
facet.set(xlim=(0.4, train_df['Ht'].max()))
facet.add_legend()
facet = sns.FacetGrid(train_df, hue="Response",aspect=4, hue_order=[1,2,3,4,5,6,7,8], palette="RdBu")
facet.map(sns.kdeplot,'Ht')
facet.set(xlim=(0.4, train_df['Ht'].max()))
facet.add_legend()
facet = sns.FacetGrid(train_df, hue="Response",aspect=4, hue_order=[1,2,3,4,5,6,7,8], palette="RdBu")
facet.map(sns.kdeplot,'BMI')
facet.set(xlim=(0, 1.0))
facet.add_legend()
ax = sns.boxplot(x="Response", y="BMI", data=train_df, order=[1,2,3,4,5,6,7,8], palette="RdBu")
ax.set(ylim=(0, 1.1))
fig, ax = plt.subplots(1,1,figsize=(14,8))
cm = plt.cm.get_cmap('RdBu')
ax = plt.scatter(train_df["Ins_Age"], train_df["BMI"], c=train_df["Response"], alpha=0.5, cmap=cm)
plt.xlim=(0, 1.0)
plt.ylim=(0, 1.0)
plt.title("Age vs BMI")
plt.xlabel("Age")
plt.ylabel("BMI")
cbar=plt.colorbar(ax)
cbar.ax.set_ylabel('Response', rotation=270)
fig, axis1 = plt.subplots(1,1,figsize=(20,5))
sns.countplot(x='InsuredInfo_6', hue="Response", data=train_df, ax=axis1, hue_order=[1,2,3,4,5,6,7,8], palette="RdBu")
facet = sns.FacetGrid(train_df, hue="Response",aspect=4, hue_order=[1,2,3,4,5,6,7,8], palette="RdBu")
facet.map(sns.kdeplot,'Insurance_History_5')
facet.set(xlim=(0, 0.01))
facet.add_legend()
fig, axis1 = plt.subplots(1,1,figsize=(20,5))
sns.countplot(x='Family_Hist_1', hue="Response", data=train_df, ax=axis1, hue_order=[1,2,3,4,5,6,7,8])