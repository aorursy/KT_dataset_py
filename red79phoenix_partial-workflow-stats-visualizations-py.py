import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )



train.describe()
train.isnull().sum()
# Replacing missing ages with median

median_age = train['Age'].dropna().median()

if len(train.Age[ train.Age.isnull() ]) > 0:

    train.loc[ (train.Age.isnull()), 'Age'] = median_age

train["Survived"][train["Survived"]==1] = "Survived"

train["Survived"][train["Survived"]==0] = "Died"

train["ParentsAndChildren"] = train["Parch"]

train["SiblingsAndSpouses"] = train["SibSp"]
plt.figure()

sns.pairplot(data=train[["Fare","Survived","Age","ParentsAndChildren","SiblingsAndSpouses","Pclass"]],

             hue="Survived", dropna=True)

#plt.savefig("1_seaborn_pair_plot.png")
sns.swarmplot(x='Age',y='Sex',hue='Survived',data=train)