# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set_style('whitegrid')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
titanic_df = pd.read_csv('../input/train.csv')

titanic_df.head()
titanic_df.tail()
titanic_df.info()
ax = sns.countplot(x="Survived", data=titanic_df)
ax = sns.countplot(x="Sex", data=titanic_df,palette="muted")
ax = sns.countplot(x="Pclass", data=titanic_df,palette="muted")
g = sns.factorplot(x="Pclass", hue="Sex", kind="count",

                   data=titanic_df,size=6,palette="muted")

g.set_ylabels("Number of Passengers")
g = sns.factorplot(x="Pclass", hue="Survived", kind="count",

                   data=titanic_df,size=6,palette="muted")

g.set_ylabels("Number of Passengers")
g = sns.factorplot(x="Sex", hue="Survived", kind="count",

                   data=titanic_df,size=6,palette="muted")

g.set_ylabels("Number of Passengers")
g = sns.factorplot(x="Sex", y="Survived",hue="Pclass", kind="bar",

                   data=titanic_df,size=6,palette="muted")

g.set_ylabels("Survival Probability")
# only in titanic_df, fill the two missing values with the most occurred value, which is "S".

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

g = sns.factorplot(x="Embarked", y="Survived", data=titanic_df,

                   size=8, kind="bar")

g.set_ylabels("Survival Probability")
# only in titanic_df, fill the two missing values with the most occurred value, which is "S".

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

g = sns.factorplot(x="Embarked", y="Survived", hue='Pclass',data=titanic_df,

                   size=8, kind="bar")

g.set_ylabels("survival probability")

# only in titanic_df, fill the two missing values with the most occurred value, which is "S".

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

g = sns.factorplot(x="Embarked", y="Survived", hue='Sex',data=titanic_df,

                   size=8, kind="bar")

g.set_ylabels("survival probability")
titanic_df = pd.read_csv('../input/train.csv')

titanic_df["Age"] = titanic_df["Age"].astype(str).map(lambda l: l.split(',')[0]).astype(float)

titanic_df["Age"] = titanic_df["Age"].fillna(100)

# We can try this too 

#titanic_df["Age"] = titanic_df["Age"].fillna(0)

# ... or this

#titanic_df["Age"] = titanic_df["Age"].fillna(titanic_df["Age"].median())



kde_value = True

plt.figure(211)

sns.distplot(titanic_df["Age"],kde=kde_value)

plt.figure(212)

sns.kdeplot(titanic_df["Age"],shade=True)
titanic_df = pd.read_csv('../input/train.csv')

titanic_df["Fare"] = titanic_df["Fare"].astype(str).map(lambda l: l.split(',')[0]).astype(float)

titanic_df["Fare"] = titanic_df["Fare"].fillna(1000)

# ... again

#titanic_df["Fare"] = titanic_df["Fare"].fillna(0)

#titanic_df["Fare"] = titanic_df["Fare"].fillna(titanic_df["fare"].median())

kde_value = True

plt.figure(211)

sns.distplot(titanic_df["Fare"],kde=kde_value)

plt.figure(212)

sns.kdeplot(titanic_df["Fare"],shade=True)
titanic_df = pd.read_csv('../input/train.csv')



var = "Age"

titanic_df[var] = titanic_df[var].astype(str).map(lambda l: l.split(',')[0]).astype(float)

titanic_df[var] = titanic_df[var].fillna(100)

#titanic_df[var] = titanic_df[var].fillna(0)

#titanic_df[var] = titanic_df[var].fillna(titanic_df[var].median())



var = "Fare"

titanic_df[var] = titanic_df[var].astype(str).map(lambda l: l.split(',')[0]).astype(float)

titanic_df[var] = titanic_df[var].fillna(1000)

#titanic_df[var] = titanic_df[var].fillna(0)

#titanic_df[var] = titanic_df[var].fillna(titanic_df[var].median())



sns.jointplot(x="Age", y="Fare", data=titanic_df)
titanic_df = pd.read_csv('../input/train.csv')



titanic_df = titanic_df.dropna(subset = ['Cabin','Age','Embarked','Fare'])

survived = titanic_df[titanic_df['Survived']==1]

cabins_survived  = np.sort(np.unique(survived['Cabin']))

dead = titanic_df[titanic_df['Survived']==0]

cabins_dead  = np.sort(np.unique(dead['Cabin']))

cabins  = np.sort(np.unique(titanic_df['Cabin']))

print(cabins)


people_in_the_cabin = titanic_df[titanic_df['Cabin']=='G6']



people_in_the_cabin

full_saved_family = titanic_df[titanic_df['Cabin']=='B96 B98']



full_saved_family