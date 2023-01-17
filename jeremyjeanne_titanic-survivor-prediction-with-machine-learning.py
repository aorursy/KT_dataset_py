# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Extract Data with panda

df=pd.read_csv("../input/train.csv")

fig = plt.figure(figsize=(18,6))



plt.subplot2grid((2,3),(0,0))

df.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("Survived")



plt.subplot2grid((2,3),(0,1))

plt.scatter(df.Survived,df.Age,alpha=0.1)

plt.title("Age wrt Survived")



plt.subplot2grid((2,3),(0,2))

df.Pclass.value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("Class")



plt.subplot2grid((2,3),(1,0),colspan=2)

for x in [1,2,3]:

    df.Age[df.Pclass ==x].plot(kind="kde")

plt.title("Class wrt Age")

plt.legend(("first class","second class","third class"))



plt.subplot2grid((2,3),(1,2))

df.Embarked.value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("Embarked")



plt.show()







#Return a tuple representing the dimensionality of the DataFrame.

print(df.shape)



# Counts for each row

print(df.count())
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Extract Data with panda

df=pd.read_csv("../input/train.csv")

fig = plt.figure(figsize=(18,6))



plt.subplot2grid((3,4),(0,0))

df.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("Survived")



plt.subplot2grid((3,4),(0,1))

df.Survived[df.Sex=="male"].value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("Men Survived")



plt.subplot2grid((3,4),(0,2))

df.Survived[df.Sex=="female"].value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("Women Survived")



plt.subplot2grid((3,4),(0,3))

df.Sex[df.Survived==1].value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("Sex of Survived")



plt.subplot2grid((3,4),(1,0),colspan=4)

for x in [1,2,3]:

    df.Survived[df.Pclass ==x].plot(kind="kde")

plt.title("Class wrt Surivied")

plt.legend(("first class","second class","third class"))



plt.subplot2grid((3,4),(2,0))

df.Survived[(df.Sex=="male") & (df.Pclass==1)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("Rich Men Survived")



plt.subplot2grid((3,4),(2,1))

df.Survived[(df.Sex=="female") & (df.Pclass==1)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("Rich Women Survived")



plt.subplot2grid((3,4),(2,2))

df.Survived[(df.Sex=="male") & (df.Pclass==3)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("Poor Men Survived")



plt.subplot2grid((3,4),(2,3))

df.Survived[(df.Sex=="female") & (df.Pclass==3)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("Poor Women Survived")



plt.show()
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Extract Data with panda

train=pd.read_csv("../input/train.csv")



train["Hyp"]=0

train.loc[train.Sex=="female","Hyp"]=1



train["Result"]=0

train.loc[train.Survived== train["Hyp"],"Result"]=1



print(train["Result"].value_counts(normalize=True))