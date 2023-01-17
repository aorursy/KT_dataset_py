import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/titanic/train.csv")
df = data.copy()

df.head()
df.columns
df.info()
df.isnull().sum()
def central_tendency(df,var,dic):

    for i,j in dic.items():

        df[var + "_" + i] = df[var].fillna(j)
df = df[['Age', 'Cabin', 'Embarked']]
from statistics import mode

lst = [np.around(df['Age'].mean(),2), df['Age'].median(), df['Age'].mode()[0]]

dic = {

    "mean" : lst[0],

    "median" : lst[1],

    "mode" : lst[2]

}


central_tendency(df,'Age', dic)
fig = plt.figure(figsize = (10,4))



sns.kdeplot(df['Age'], color = 'r')



sns.kdeplot(df['Age_mean'], color = 'pink')



sns.kdeplot(df['Age_median'], color = "purple")



sns.kdeplot(df['Age_mode'])

plt.tight_layout()

plt.show()
for i in df.columns:    

    if "age" in str(i).lower():

        print(i, ":", df[i].std()) 
df = data.copy()
df = df[['Age',]]
from random import sample

def randomchange(df,var):

    df[var + '_random'] = df[var]

    global random_sample

    random_sample = df[var].dropna().sample(df[var].isnull().sum())

    random_sample.index = df[df[var].isnull()].index

    

    df.loc[df[var].isnull(), var + '_random'] = random_sample
randomchange(df,'Age')
fig = plt.figure(figsize = (10,4))



sns.kdeplot(df['Age'], color = 'r')



sns.kdeplot(df['Age_random'], color = "black")

plt.tight_layout()

plt.show()
for i in df.columns:

    print(i, ":", df[i].std())
df = data.copy()

df = df[['Age']]
df['Age_Feature'] = np.where(df['Age'].isnull(), 1,0)
df
df = data.copy()

df = df[['Age']]
figure = plt.hist(df['Age'], bins = 50)



plt.figure()
extreme = df['Age'].mean() + 3 * df['Age'].std()
df['Age_distribution_imputation'] = df['Age'].fillna(extreme)
df[df['Age'].isnull()]
plt.figure(figsize = (10,4))

plt.subplot(1,2,1)

sns.distplot(df['Age'], kde = False, bins = 30)

plt.subplot(1,2,2)

sns.distplot(df['Age_distribution_imputation'], kde = False, bins = 30)
for i in df.columns:

    print(i , ":", df[i].std())
df = data.copy()
df = df[['Age']]
figure = plt.hist(df['Age'], bins = 50)



plt.figure()
#fill the value with outliers

#that is 

df['Age_left'] = df['Age'].fillna(0)

df['Age_right'] = df['Age'].fillna(90)
plt.figure(figsize = (10,4))

plt.subplot(1,3,1)

sns.distplot(df['Age'], kde = False, bins = 30)

plt.subplot(1,3,2)

sns.distplot(df['Age_left'], kde = False, bins = 30)

plt.subplot(1,3,3)

sns.distplot(df['Age_right'], kde = False, bins = 30)

plt.show()
for i in df.columns:

    print(i , ":", df[i].std())