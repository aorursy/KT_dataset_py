import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")

data_all = [data_train,data_test]

full = data_train.append(data_test)

print("train data shape: ", data_train.shape)

print("test data shape: ", data_test.shape)

print("full data shape:", full.shape)

data_train.head(5)
# 

data_train.describe(include=["O"])
data_train.info()
data_train.describe(exclude=["O"])
# 利用heatmap直接绘制pairwise的特征相关性

cmap = sns.diverging_palette(200,10,as_cmap=True)

sns.heatmap(data=data_train.corr(),annot=True,square=True,cmap=cmap)
# 探查连续变量与survived的关系

facet = sns.FacetGrid(data_train,row="Survived",col="Sex",aspect=1.2)

facet.map(sns.distplot,"Age")
facet = sns.FacetGrid(data_train,row="Survived",col="Sex",aspect=1.0)

facet.map(sns.distplot,"Fare")
# 探查离散特征

facet = sns.FacetGrid(data_train)

ax1 = facet.map(sns.barplot,"Sex","Survived")
facet = sns.FacetGrid(data_train)

facet.map(sns.barplot,"Embarked","Survived")
facet = sns.FacetGrid(data_train)

facet.map(sns.barplot,"Pclass","Survived")
# 计算具体的值

g = data_train.groupby("Pclass",as_index=False)["Survived"].mean()

g.sort_values(by="Survived",ascending=False)
# 将性别变成binary values

full["Sex"] = full["Sex"].map({"male":1,"female":0})

full.head(2)
# find the median age value for each subset

ageValues = np.zeros((2,3))

for i in range(2):

    for j in range(3):

        df = full[(full["Sex"]==i) & (full["Pclass"]==j+1)]["Age"].dropna()

        guessAge = df.median()

        ageValues[i,j]=guessAge

print(ageValues)



# replace nan by median age value

for i in range(2):

    for j in range(3):

        full.loc[(full["Sex"]==i) & \

                 (full["Pclass"]==j+1) & \

                 (full["Age"].isnull()),"Age"] = ageValues[i,j]

out = pd.cut(full["Age"].values,5)

def discrete_age(age):

    for i in range(len(out.categories)):

        if (age in out.categories[i]):

            return i

full["newAge"] = full["Age"].map(discrete_age).astype("int")

full.head(4)
# 因为fare只有一个是nan，所以将其赋予median即可

full.loc[full["Fare"].isnull(),"Fare"] = full["Fare"].median()

fare_out  = pd.qcut(full["Fare"],5)

def discrete_fare(fare):

    for i in range(len(fare_out.values.categories)):

        if(int(fare) in fare_out.values.categories[i]):

            return i

full["newFare"] = full["Fare"].map(discrete_fare)

full[full["newFare"].isnull()]
full['Title'] = full.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(full["Title"],full["Sex"])
full['Title'] = full['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

                                       'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

full['Title'] = full['Title'].replace('Mlle', 'Miss')

full['Title'] = full['Title'].replace('Ms', 'Miss')

full['Title'] = full['Title'].replace('Mme', 'Mrs')
titles = pd.get_dummies(full["Title"])
full.head(5)
full["FamilySize"] = full["SibSp"] + full["Parch"]

full["IsAlone"] = 0

full.loc[full["FamilySize"]==1,"IsAlone"] = 1
embarked = pd.get_dummies(full["Embarked"])
data = pd.concat([embarked,titles,full.Sex,full.SibSp,full.Parch,full.newAge,full.FamilySize,full.IsAlone],axis=1)
data.head(2)
from sklearn.model_selection import train_test_split

X = data[:891]

y = data_train["Survived"]

X_train, X_valid, y_train, y_valid = train_test_split(\

    X, y, test_size=0.33, random_state=42)

print(X_train.shape)

print(X_valid.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

acc_train = round(logreg.score(X_train, y_train) * 100, 2)

acc_valid = round(logreg.score(X_valid,y_valid) * 100, 2)

print("train acc:", acc_train)

print("valid acc:", acc_valid)
coeff_df = pd.DataFrame(X_train.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines

svc = SVC()

svc.fit(X_train, y_train)

acc_train = round(svc.score(X_train, y_train) * 100, 2)

acc_valid = round(svc.score(X_valid,y_valid) * 100, 2)

print("train acc:", acc_train)

print("valid acc:", acc_valid)
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train,y_train)

acc_train = round(knn.score(X_train, y_train) * 100, 2)

acc_valid = round(knn.score(X_valid,y_valid) * 100, 2)

print("train acc:", acc_train)

print("valid acc:", acc_valid)
gaussian = GaussianNB()

gaussian.fit(X_train,y_train)

acc_train = round(gaussian.score(X_train, y_train) * 100, 2)

acc_valid = round(gaussian.score(X_valid,y_valid) * 100, 2)

print("train acc:", acc_train)

print("valid acc:", acc_valid)
perceptron = Perceptron()

perceptron.fit(X_train,y_train)

acc_train = round(perceptron.score(X_train, y_train) * 100, 2)

acc_valid = round(perceptron.score(X_valid,y_valid) * 100, 2)

print("train acc:", acc_train)

print("valid acc:", acc_valid)
linear_svc = LinearSVC()

linear_svc.fit(X_train,y_train)

acc_train = round(linear_svc.score(X_train, y_train) * 100, 2)

acc_valid = round(linear_svc.score(X_valid,y_valid) * 100, 2)

print("train acc:", acc_train)

print("valid acc:", acc_valid)
sgd = SGDClassifier()

sgd.fit(X_train,y_train)

acc_train = round(sgd.score(X_train, y_train) * 100, 2)

acc_valid = round(sgd.score(X_valid,y_valid) * 100, 2)

print("train acc:", acc_train)

print("valid acc:", acc_valid)
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train,y_train)

acc_train = round(decision_tree.score(X_train, y_train) * 100, 2)

acc_valid = round(decision_tree.score(X_valid,y_valid) * 100, 2)

print("train acc:", acc_train)

print("valid acc:", acc_valid)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train,y_train)

acc_train = round(random_forest.score(X_train, y_train) * 100, 2)

acc_valid = round(random_forest.score(X_valid,y_valid) * 100, 2)

print("train acc:", acc_train)

print("valid acc:", acc_valid)