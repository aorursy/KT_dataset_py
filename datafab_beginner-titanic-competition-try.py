# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Importing libraries

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline
# First we need to load our training and test data:



data_test = pd.read_csv('../input/test.csv')

data_train = pd.read_csv('../input/train.csv')
# Check the data structure of both training and test data:

data_test.info()
data_train.info()
data = [data_train, data_test]

data_train.drop("PassengerId", axis=1, inplace=True)

for d in data:

    d.drop("Ticket", axis=1, inplace=True)

    

data[0].head()
data_train.Pclass.describe()
sns.distplot(data_train.Pclass, bins=3)
byPclassSex = data_train.groupby(["Pclass", "Sex"], as_index=False)["Survived"].agg(["mean", "count", "sum"])

byPclassSex = byPclassSex.reset_index()

sns.barplot(data=byPclassSex, x="Pclass", y="mean", hue="Sex")
sns.barplot(data=byPclassSex, x="Sex", y="mean")
# Fill Age with mean

for d in data:

    d.Age.fillna(d.Age.mean(), inplace=True)
f = plt.figure()

ax = f.add_subplot(111)



ax.set_xlim(0, 80)



sns.distplot(data_train.Age, bins=8, ax=ax)
for d in data:

    d["AgeCluster"] = pd.cut(d.Age, bins=[0, 3, 10, 50, 80], labels=["Baby", "Toddler", "Adult", "Elder"])



data_train.AgeCluster.value_counts().sort_index()
byAgeSex = data_train.groupby(["AgeCluster", "Sex"])["Survived"].mean().reset_index()

byAgeSex
sns.barplot(data=byAgeSex, x="AgeCluster", y="Survived", hue="Sex")
data_train.Name.head()
def getTitle(name):

    if name != name:

        return None

    else:

        name = name.strip().split(",")[1].split('.')[0].strip()

        if name == 'Ms':

            name = 'Miss'

        elif name == 'Dr ':

            name = 'Mr'

        elif name == 'Master':

            name = 'Mr'

        elif name == 'Dr':

            name = 'Mr'

        elif name == 'Rev':

            name = 'Mr'

        elif name.strip() == 'Mlle':

            name = 'Miss' # Mlle equals Miss

        elif name == 'Col':

            name = 'Mr'

        elif name == 'Major':

            name = 'Mr'

        elif name == 'Capt':

            name = 'Mr'

        elif name == 'Don':

            name = 'Mr'

        elif name == 'Sir':

            name = 'Mr'

        elif name == 'Lady':

            name = 'Mrs'

        elif name == 'the Countess':

            name = 'Mrs'

        elif name == 'Dona':

            name = "Mrs"

        elif name == 'Jonkheer':

            name = 'Mr'

        elif name == 'Mme':

            name = 'Miss' # Only 24 years old and traveling alone, therefor highly possible not married

        return name.strip()

        
for d in data:

    d["Title"] = d.Name.apply(getTitle)

    d.drop("Name", axis = 1, inplace = True)
for d in data:

    print(d.Title.value_counts())
sns.barplot(x = data_train.Title, y = data_train.Survived)
data[0].head()
sns.barplot(data = data_train, x = "Parch", y = "Survived", hue = "Sex")
# Doesn't seem to have any influence -> Remove Parch:

for d in data:

    d.drop("Parch", axis = 1, inplace = True)
def getCabinShort(c):

    if c != c:

        return None

    else:

        return c[0]
for d in data:

    d["CabinShort"] = d.Cabin.apply(getCabinShort)
data_train.CabinShort.value_counts()
sns.barplot(data = data_train, x = "CabinShort", y = "Fare")
sns.barplot(data = data_train, x = "CabinShort", y = "Survived")
sns.violinplot(data = data_train, y = "Fare", x = "Survived")
sns.barplot(data = data_train, y = "Fare", x = "Survived")
for d in data:

    d.drop("Cabin", axis = 1, inplace=True)

    d.drop("CabinShort", axis = 1, inplace=True)
f = plt.figure(figsize=(10,5))

ax1 = f.add_subplot(121)

ax2 = f.add_subplot(122)



sns.barplot(data = data_train, y = "Embarked", x = "Fare", ax=ax1)

sns.barplot(data = data_train, y = "Embarked", x = "Survived", ax=ax2)
for d in data:

    d.drop("Embarked", axis = 1, inplace=True)
for d in data:

    d["Sex"] = d.Sex.map({'male': 0, 'female': 1})

    d["Title"] = d.Title.map({'Mr': 0, 'Mrs': 1, 'Miss': 2})

    d["AgeCluster"] = d.AgeCluster.map({'Baby': 0, 'Toddler': 1, 'Adult': 2, 'Elder':3})

     

for d in data:

    d.drop("Age", axis = 1, inplace=True)
sns.distplot(data_train.Fare)

print(data_train.Fare.describe())
bins = [0, 10, 30, 50, 100, 550]

labels = ["cheapest", "cheap", "mid", "expensive", "insane!!!"]



for d in data:

    d["FareCluster"] = pd.cut(d.Fare, bins = bins, labels = labels)

    

data_train.FareCluster.value_counts()
sns.barplot(data=data_train, x="FareCluster", y="Survived")
# Map FareCluster aswell and remove Fare

for d in data:

    d["FareCluster"] = d.FareCluster.map({"cheapest": 0, "cheap": 1, "mid": 2, "expensive": 3, "insane!!!": 4})

    d.drop("Fare", axis=1, inplace=True)
sns.heatmap(data_train.corr(), annot=True)
data_train.dropna(inplace=True)

data_test.FareCluster.fillna(1.0, inplace=True)
X = data_train.drop("Survived", axis = 1)

y = data_train.Survived
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)

clf.fit(X, y)
important_features = pd.Series(data=clf.feature_importances_,index=X.columns)

important_features.sort_values(ascending=False,inplace=True)

important_features
pred = clf.predict(data_test.drop("PassengerId", axis = 1))
clf.score(X, y)
data_test["Survived"] = pred
data_test.head()
# data_test[["PassengerId", "Survived"]].to_csv("submission.csv", index=False)