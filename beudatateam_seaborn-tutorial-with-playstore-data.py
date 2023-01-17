# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

import seaborn as sns
dataset = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
dataset.head()
dataset.describe()
dataset.info()
print("Category unique values:",dataset["Category"].unique())



print("Content rating unique values:",dataset["Content Rating"].unique())



print("Type unique values:",dataset["Type"].unique())



# Drop nan values

dataset.drop(dataset[dataset["Type"] == "0"].index,inplace = True)

dataset.drop(dataset[dataset["Type"].isna()].index,inplace = True)

dataset.drop(dataset[dataset["Category"] == "1.9"].index,inplace = True)

dataset["Content Rating"].dropna(inplace = True)
# Clean Price column

dataset["Price"] = dataset["Price"].apply(lambda x: float(x.replace("$",'')))

# Convert reviews to int

dataset["Reviews"] = dataset["Reviews"].apply(lambda x: int(x))
# Clean Installs column

dataset['Installs'] = dataset['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)

dataset['Installs'] = dataset['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)

dataset['Installs'] = dataset['Installs'].apply(lambda x: int(x))
# Make another dataset for Size analysis as dropping rows will reduce the amount of dataset

def kb_to_mb(row):

    

    if "k" in str(row):

        row = row.replace('k','')

        size = float(row)/1000

    else:

        row = row.replace("M",'').replace(",",'').replace("+",'')

        size = float(row)

    return size

ds_clear_size = dataset[dataset["Size"] != 'Varies with device']

ds_clear_size["Size"] = ds_clear_size["Size"].apply(kb_to_mb)
sns.distplot(dataset["Rating"],kde = True,bins = 20)
sns.jointplot(y = "Size",x = "Rating",data=ds_clear_size,kind="hex")
sns.pairplot(ds_clear_size,kind = "scatter",diag_kind = "hist",hue = "Type")
ax = sns.stripplot(x = "Content Rating",y = "Rating",data=dataset,jitter = True,hue = "Type",dodge = True)

ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
ax = sns.swarmplot(x = "Content Rating",y = "Rating",data=dataset[1500:2000],hue = "Type")

ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
ax = sns.boxplot(x = "Content Rating",y = "Rating",data=dataset,hue = "Type")

ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
ax = sns.boxenplot(x = "Content Rating",y = "Rating",data=dataset,hue = "Type")

ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
ax = sns.violinplot(x = "Content Rating",y = "Rating",data=dataset,hue = "Type",split = True,inner = "quartile")

ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
sns.countplot(x = "Type",data=dataset,hue = "Content Rating")
ax = sns.barplot(x = "Content Rating",y = "Rating",data=dataset,hue = "Type",ci="sd")

ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
dataset.corr()
sns.heatmap(ds_clear_size.corr(),annot = True,linewidth = .5)