import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
data.head()
lst = []

for i in data.columns:

    if data[i].isnull().sum() >0:

        lst.append(i)
lst
data = data[lst]
data
data.isnull().sum()
df = data.copy()

#comparing their means

df.isnull().mean()
for i in df.columns:

    print(i, ":", df[i].size, ":", df[i].isnull().sum())
lst = []

for i in df.columns:

    if df[i].isnull().sum() < 100:

        lst.append(i)

        df[i + "_mean"] = df[i].fillna(df[i].value_counts().sort_values(ascending = False).index[0])
df.isnull().sum()
def plots(df,var):

    plt.figure(figsize =(10,5))

    val = np.random.randint(100000,999999)

    col = "#" + str(val)

    sns.countplot(df[var], color = col, label = var)

    plt.legend()

    plt.show()
for i in df.columns:

    if df[i].isnull().sum()<100:

        plots(df,i)
for i in df.columns:

    if df[i].isnull().sum()<100 and df[i].isnull().sum()>=1:

        df = df.drop(columns = i, axis = 1)
df.isnull().sum()
for i in df.columns:

    if df[i].isnull().sum()<1000 and df[i].isnull().sum() >= 1:

        df[i + "_exposure"] = np.where(df[i].isnull(), 1, 0)
df.isnull().sum()
df[['LotFrontage_exposure', 'LotFrontage']][0:20]
#now replace that nan with something else

#Machine has learn that something happend in that place
for i in df.columns:

    if df[i].isnull().sum() >1000:

        df[i] =  df[i].fillna("missing")

df
df.isnull().sum()
from random import sample

def randomchange(df,var):

    df[var + '_random'] = df[var]

    global random_sample

    

    random_sample = df[var].dropna().sample(df[var].isnull().sum())

    

    random_sample.index = df[df[var].isnull()].index

    

    df.loc[df[var].isnull(), var + '_random'] = random_sample
randomchange(df, "FireplaceQu")

randomchange(df, "LotFrontage")
df['FireplaceQu_random'].isnull().sum()
def plots(df, var):

    plt.figure(figsize =(10,5))

    val = np.random.randint(100000,999999)

    col = "#" + str(val)

    sns.countplot(df[var], color = col, label = var)

    plt.legend()

    plt.show()
for i in ['FireplaceQu_random', "FireplaceQu", "LotFrontage", "LotFrontage_random"]:

    plots(df, i)