# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt
test = pd.read_csv("../input/titanic-machine-learning-from-disaster/test.csv")

train = pd.read_csv("../input/titanic-machine-learning-from-disaster/train.csv")
train.head()

test.head()
train.info()
print("0=Tidak Selamat and 1=Selamat")

fig = plt.figure()

train.Survived.value_counts(True).plot(kind="bar",alpha=0.5)

plt.show()
plt.subplot2grid((2,3) , (0,0))

train.Survived.value_counts(True).plot(kind="bar",alpha=0.5)

plt.title("Survived")



plt.subplot2grid((2,3) , (0,2))

plt.scatter(train.Survived, train.Age,alpha=0.1)

plt.title("age survived")



plt.subplot2grid((2,3) , (1,1))

train.Pclass.value_counts(True).plot(kind="bar",alpha=0.5)

plt.title("Class")





plt.show()