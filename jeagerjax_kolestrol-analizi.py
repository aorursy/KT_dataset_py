# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/heart-disease-uci/heart.csv")

df.columns

df.head(9)


sns.countplot(x="sex", data=df, palette="bwr")

plt.show()
a = df.groupby('sex').mean()

pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
plt.figure(figsize=(9,4))

plt.plot(df.thalach)

print("Highest value: ", df.thalach.max())

print("Mean value: ", df.thalach.mean())

print("Lowest value: ", df.thalach.min())

plt.xlabel("Number of Poeple")

plt.ylabel("Heart Rate")

plt.title("Max Heart Rate")

plt.show()
plt.scatter(df.trestbps,df.age,color = "red",label=" 1",alpha=0.3)

plt.scatter(df.trestbps,df.age,color = "yellow",label="0",alpha=0.3)

plt.xlabel("Resting Blood Pressure ")

plt.ylabel("Age")

plt.legend()

plt.show()
plt.figure(figsize=(14,8))

sns.heatmap(df.corr(), linewidths=.01, annot = True, cmap='Greens')

plt.show()
plt.scatter(df.trestbps,df.chol,color = "Black",label="1",alpha=0.3)

plt.scatter(df.trestbps,df.chol,color = "red",label="0",alpha=0.3)

plt.xlabel("Resting Blood Pressure ")

plt.ylabel("Cholesterol")

plt.legend()

plt.show()