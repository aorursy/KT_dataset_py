# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import sys

import os

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data= pd.read_csv("../input/heart.csv")

print("there are", len(data.columns), "columns: ")

print (data.columns)

data.columns = [c.replace(' ', '_') for c in data.columns]

data.info()
#We will now analyze the genders first. 

#We can see that HEart attacks are more common in male than female. Lets do further analyses.
data.head()
sns.countplot(x="sex", data=data)

plt.show()


plt.scatter(x=data.age[data.target==1], y=data.thalach[(data.target==1)], c ="red")

plt.scatter(x=data.age[data.target==0], y=data.thalach[(data.target==0)], c ="green")

plt.xlabel("age")

plt.ylabel("max heart rate")

plt.title("Heart Rate vs Age")

plt.show()


sns.countplot(x="cp", data=data, hue="target")

plt.show()
sns.countplot(x="ca", data=data, hue="target")

plt.show()
sns.countplot(x="exang", data=data, hue="target")

plt.show()
sns.countplot(x="ca", data=data, hue="target")

plt.show()
data.describe()

#We can see visualization of statistical calculations.

data.boxplot(column="thalach", by="target")

# ages value by sex

plt.show()
df3 = data[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']]





plt.figure(figsize = (15,15))

sns.pairplot(df3)

plt.show()
plt.figure(figsize=(10,10))

sns.heatmap(data.corr(),annot=True,fmt='.1f')

plt.show()