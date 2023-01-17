# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/heart-disease-uci/heart.csv")
data.info()
data.head(10)
data.corr() # method = pearson , standard correlation coefficient
data.corr(method = "kendall") #Kendall Tau correlation coefficient
# correlation map

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data.corr(), annot=True, linewidths=.3, fmt= '.1f',ax=ax)

plt.show()
data.columns
sns.countplot(x='sex', data=data, palette="mako_r")

plt.xlabel("Sex (0 = Female , 1 = Male)")

plt.show()
pd.crosstab(data.age,data.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
plt.scatter(x=data.age[data.target==1], y=data.trestbps[(data.target==1)], c="red")

plt.scatter(x=data.age[data.target==0], y=data.trestbps[(data.target==0)])

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Resting Blood Pressure")

plt.show()
data.info()
data.describe()