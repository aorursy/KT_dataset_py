import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

import numbers

import sklearn.linear_model as skl

from collections import Counter

print(os.listdir("../input"))

#df_bike = pd.read_csv("../input/nyc-east-river-bicycle-counts.csv")

df = pd.read_csv("../input/epirecipes/epi_r.csv")

df.head()
df.isnull().sum().sum()
df = df.dropna()
df = df.drop(df[df.calories>10000].index)
#Check if the vairable is int or float

print("int",all(isinstance(i,int) for i in df.calories))

print("float",all(isinstance(i,float) for i in df.calories))
x_train = df.calories.values.reshape(-1,1)

y_train = df.dessert
clf = skl.LogisticRegression(solver='newton-cg')

clf.fit(x_train, y_train)

prob = clf.predict_proba(x_train)
fig, ax1 = plt.subplots(1,1)

ax1.scatter(x_train, y_train, color = 'orange')

ax1.plot(x_train, prob[:,1], color = 'lightblue')
ax2 = sns.regplot(x = x_train, y = y_train, logistic= True)