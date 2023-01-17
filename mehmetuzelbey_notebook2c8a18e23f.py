import pandas as pd

import numpy as np

import time



from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display



from sklearn import metrics

import seaborn as sns

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
train= pd.read_csv("train.csv")

test= pd.read_csv("test.csv")
train.head()
Train_X = train.copy()
Train_X["Sex"] = Train_X["Sex"].astype("category")
Train_X.head()
Train_X["Sex"] = Train_X["Sex"].cat.codes
Train_X.head()
Train_X.tail()
Train_X["Age"].fillna(round(Train_X["Age"].mean()),inplace=True)
Train_X.tail()
Train_X["Fare"].isnull().sum()
Train_X["Embarked"].fillna("S",inplace=True)
Train_X["Embarked"].isnull().sum()
Train_X["Embarked"] = Train_X["Embarked"].astype("category")

Train_X["Embarked"] = Train_X["Embarked"].cat.codes

Train_X.head()
Train_X["Cabin"].fillna("XX", inplace=True)
Train_X.head()
Train_X["Cabin"] = Train_X["Cabin"].astype("category")

Train_X["Cabin"] = Train_X["Cabin"].cat.codes

Train_X.head()
Train_X = Train_X.drop(["Ticket"],axis=1)
Train_X.head()
Train_X[["Cabin","Survived"]].groupby(["Cabin"],as_index=False).mean()
sns.heatmap(Train_X.corr(),annot=True,linewidths=0.2,cmap="YlGnBu")

fig=plt.gcf()

fig.set_size_inches(20,12)

plt.show()
test.head()
Test_X = test.copy()

Test_X["Sex"] = Test_X["Sex"].astype("category")

Test_X["Sex"] = Test_X["Sex"].cat.codes
Test_X["Age"].fillna(round(Test_X["Age"].mean()),inplace=True)

Test_X["Embarked"].fillna("S",inplace=True)

Test_X["Embarked"] = Test_X["Embarked"].astype("category")

Test_X["Embarked"] = Test_X["Embarked"].cat.codes

Test_X["Cabin"].fillna("XX", inplace=True)

Test_X["Cabin"] = Test_X["Cabin"].astype("category")

Test_X["Cabin"] = Test_X["Cabin"].cat.codes

Test_X["Fare"].fillna(Test_X["Fare"].mean(),inplace=True)

Test_X = Test_X.drop(["Ticket","Name","PassengerId"],axis=1)

Test_X.head()
Train_Y = Train_X["Survived"]

Train_x = Train_X.drop(["Name","Survived","PassengerId"],axis=1)
Train_x.shape, Train_Y.shape, Test_X.shape
random_forest = RandomForestClassifier(n_estimators=500, max_leaf_nodes=100, max_depth=50,min_samples_split=4)

%time random_forest.fit(Train_x, Train_Y)
randomf_Y_preds = random_forest.predict(Test_X)

random_forest.score(Train_x, Train_Y)

randomf_acc = random_forest.score(Train_x, Train_Y)

randomf_acc
rf_parameters={"max_depth":[50,100,150,200,250],

          "max_features":[2,3,5,6,7,8],

          "n_estimators":[100,300,500,700],

          "min_samples_split":[2,3,4,6,8,10],

          "max_leaf_nodes":[50,80,100,150]}
rf_model = RandomForestClassifier()
rf_cv_model= GridSearchCV(rf_model,

                         rf_parameters,

                         cv=5,

                         n_jobs=-1,

                         verbose=2)
%time rf_cv_model.fit(Train_x,Train_Y)
print("The Best Parameters: "+str(rf_cv_model.best_params_))
randomf_tuned = RandomForestClassifier(max_depth=200,max_features=3,max_leaf_nodes=150,min_samples_split=8,n_estimators=100)

randomf_tuned.fit(Train_x,Train_Y)
random_forest_Y_pred = randomf_tuned.predict(Test_X)

random_forest.score(Train_x, Train_Y)

random_forest_accuracy = randomf_tuned.score(Train_x, Train_Y)

random_forest_accuracy
random_forest_submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": random_forest_Y_pred})

random_forest_submission.to_csv('random_forest_submission.csv', index=False)