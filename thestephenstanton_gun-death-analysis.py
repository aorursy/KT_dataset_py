import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



guns = pd.read_csv("../input/guns.csv", index_col=0)
print(guns.shape)

guns.head()
guns.notnull().sum() * (100/guns.shape[0])
guns = guns.dropna(axis=0, how="any")
guns = guns[guns.intent != "Undetermined"]
print(guns.month.value_counts(sort=False))

print(guns.year.value_counts(sort=False))
from datetime import datetime

guns["date"] = guns.apply(lambda row: datetime(row.year, row.month, 1), axis=1)

del guns["year"]

del guns["month"]



# while I'm here just gonna make police into a boolean for readability 

guns.police = guns.police.astype(bool)
guns.head()
intentAndPolice = guns.groupby([guns.intent, guns.police]).intent.count().unstack('police')

plot = intentAndPolice.plot(kind="bar", stacked=True)

plot.legend(labels=["Police Involved", "Police Not Involved"])

plot.set_xlabel("Intent")

plot.set_ylabel("Count")

plt.show()
intentAndPolice
plt.hist(guns.age, range(0, 100))

plt.xlabel("Age")

plt.ylabel("Count")

plt.title("Death Distribution by Age")

plt.show()
plt.hist(guns.age[guns.intent == "Homicide"], range(0, 100))

plt.xlabel("Age")

plt.ylabel("Count")

plt.title("Homicide")

plt.show()
plt.hist(guns.age[guns.intent == "Suicide"], range(0, 100))

plt.xlabel("Age")

plt.ylabel("Count")

plt.title("Suicide")

plt.show()
plt.hist(guns.age[guns.intent == "Accidental"], range(0, 100))

plt.xlabel("Age")

plt.ylabel("Count")

plt.title("Accidental")

plt.show()
intentAndSex = guns.groupby([guns.intent, guns.sex]).intent.count().unstack('sex')

plot = intentAndSex.plot(kind="bar", stacked=True)

plot.legend(labels=["Female", "Male"])

plot.set_xlabel("Intent")

plot.set_ylabel("Count")

plt.show()
intentAndSex
intentAndRace = guns.groupby([guns.intent, guns.race]).intent.count().unstack('race')

plot = intentAndRace.plot(kind="bar", stacked=True)

plot.set_xlabel("Intent")

plot.set_ylabel("Count")

plt.show()
intentAndRace
intentAndRace = intentAndRace.div(intentAndRace.sum(1).astype(float), axis=0)

plot = intentAndRace.plot(kind="bar", stacked=True)

plot.set_xlabel("Intent")

plot.set_ylabel("Count")

plt.legend(bbox_to_anchor=(1.1,0.9))

plt.show()
guns.hispanic.value_counts() * (100/guns.hispanic.shape[0])
del guns["hispanic"]
guns.head()
intentAndPlace = guns.groupby([guns.intent, guns.place]).intent.count().unstack('place')

plot = intentAndPlace.plot(kind="bar", stacked=True)

plot.set_xlabel("Intent")

plot.set_ylabel("Count")

plt.show()
intentAndPlace = intentAndPlace.div(intentAndPlace.sum(1).astype(float), axis=0)

plot = intentAndPlace.plot(kind="bar", stacked=True)

plot.set_xlabel("Intent")

plot.set_ylabel("Count")

plt.legend(bbox_to_anchor=(1.1,0.9))

plt.show()
indexOfOthers = guns[(guns.place != "Home") & (guns.place != "Street")].index

guns.loc[indexOfOthers, "place"] = "Other"
intentAndPlace = guns.groupby([guns.intent, guns.place]).intent.count().unstack('place')

plot = intentAndPlace.plot(kind="bar", stacked=True)

plot.set_xlabel("Intent")

plot.set_ylabel("Count")

plt.show()
intentAndPlace = intentAndPlace.div(intentAndPlace.sum(1).astype(float), axis=0)

plot = intentAndPlace.plot(kind="bar", stacked=True)

plot.set_xlabel("Intent")

plot.set_ylabel("Count")

plt.legend(bbox_to_anchor=(1.1,0.9))

plt.show()
intentAndEducation = guns.groupby([guns.intent, guns.education]).intent.count().unstack('intent')

plot = intentAndEducation.plot(kind="bar", stacked=True)

plot.set_xlabel("Education")

plot.set_ylabel("Count")

plt.show()
guns = pd.read_csv("../input/guns.csv", index_col=0)



intentAndMonth = guns.groupby([guns.intent, guns.month]).intent.count().unstack('intent')

plot = intentAndMonth.plot(kind="bar", stacked=True)

plt.legend(bbox_to_anchor=(1.1,0.9))

plot.set_xlabel("Month")

plot.set_ylabel("Count")

plt.show()
import time



import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn import neighbors

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier
guns = pd.read_csv("../input/guns.csv", index_col=0)



# Prep the data (See Analysis.ipynb)

del guns["year"]

del guns["month"]

del guns["hispanic"]

guns = guns.dropna(axis=0, how="any")

guns = guns[guns.intent != "Undetermined"]

indexOfOthers = guns[(guns.place != "Home") & (guns.place != "Street")].index

guns.loc[indexOfOthers, "place"] = "Other"



guns = guns.apply(LabelEncoder().fit_transform)

        

X = guns.iloc[:, 1:]

y = guns.iloc[:, 0]



XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = .20)
# Just gonna try a simple knn classifier, will be good base to compare to

# After testing, n = 10 seems to be about the best we will get

knn = neighbors.KNeighborsClassifier(n_neighbors=10)

knn.fit(XTrain, yTrain)

accuracy = knn.score(XTest, yTest)

print("KNN: {0} %".format(accuracy * 100))
# So since this is a classification problem, we can't use linear regression,

# it will give too much weight to data far from the decision frontier 

# I still want to use a linear approach, so I will choose logistic regression 

logReg = LogisticRegression()

logReg.fit(XTrain, yTrain)

accuracy = logReg.score(XTest, yTest)

print("Logistic Regression: {0} %".format(accuracy * 100))
# Might as well try a decision tree

decisionTree = DecisionTreeClassifier()

decisionTree.fit(XTrain, yTrain)

accuracy = decisionTree.score(XTest, yTest)

print("Decision Tree: {0} %".format(accuracy * 100))



print("Limiting Linear SVC to 5000 points or it will take ages")

time.sleep(1) # just so the message above will show
# Using the ultimate machine learning cheat sheet, it says given

# the parameters, I should choose linear svc

svc = LinearSVC()

svc.fit(XTrain[:5000], yTrain[:5000])

svc.score(XTest, yTest)

accuracy = svc.score(XTest, yTest)

print("SVC: {0} %".format(accuracy * 100))