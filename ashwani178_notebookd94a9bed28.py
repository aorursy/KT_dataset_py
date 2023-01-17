# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import KFold

from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
titanic=pd.read_csv('../input/train.csv')

print(titanic.head())

titanic.describe()
titanic.hist(bins=10,figsize = (9,7), grid = False)
import seaborn as sns

ax = sns.boxplot(x="Survived",y="Age",data=titanic)

ax = sns.stripplot(x="Survived",y="Age", data= titanic, jitter = True)

sns.plt.show()
ax = sns.boxplot(x="Survived",y="Fare",data=titanic)

ax = sns.stripplot(x="Survived",y="Fare", data= titanic, jitter = True)

sns.plt.show()
ax = sns.boxplot(x="Survived",y="Embarked",data=titanic)

ax = sns.stripplot(x="Survived",y="Embarked", data= titanic, jitter = True)

sns.plt.show()
data =pd.concat([titanic["Survived"],titanic["Age"]], axis=1)

data.plot.scatter(x="Age",y="Survived", ylim = (0,1))
corr = titanic.corr()

f,ax = plt.subplots(figsize=(25,16))

sns.heatmap(corr,cmap='inferno',linewidth=0.1,vmax=1.0,square=True,annot=True)
titanic.describe()

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
print(titanic["Sex"].unique())

titanic.loc[(titanic["Sex"] == "male"),"Sex"] = 0

titanic.loc[(titanic["Sex"] == "female"),"Sex"] = 1
titanic["Embarked"] = titanic["Embarked"].fillna("S")
print(titanic["Embarked"].unique())

titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 1

titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 2

titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 3
import numpy as np

import pandas as pd

titanic=pd.read_csv('../input/train.csv')

titanic_test = pd.read_csv('../input/test.csv')

titanic_test["Age"]= titanic_test["Age"].fillna(titanic["Age"].median())

titanic_test.loc[titanic_test["Sex"]=="male","Sex"] = 0

titanic_test.loc[titanic_test["Sex"]=="female","Sex"] = 1

titanic_test["Embarked"]= titanic_test["Embarked"].fillna("S")

titanic_test.loc[titanic_test["Embarked"] == "S","Embarked"] = 0

titanic_test.loc[titanic_test["Embarked"] == "C","Embarked"] = 1

titanic_test.loc[titanic_test["Embarked"] == "Q","Embarked"] = 2

titanic_test["Fare"]= titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic["Embarked"] = titanic["Embarked"].fillna("S")
print(titanic["Sex"].unique())

titanic.loc[(titanic["Sex"] == "male"),"Sex"] = 0

titanic.loc[(titanic["Sex"] == "female"),"Sex"] = 1

print(titanic["Embarked"].unique())

titanic["Embarked"] = titanic["Embarked"].fillna("S")

titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0

titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1

titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
predictors= ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

print (titanic[predictors].iloc[0])

print (titanic["Survived"].iloc[0])

print (type(titanic[predictors].iloc[0]))

print (type(titanic["Survived"].iloc[0]))
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic.describe()
predictors= ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

lgr=LogisticRegression(random_state=1)

lgr.fit(titanic[predictors], titanic["Survived"])

prediction= lgr.predict(titanic_test[predictors]) 

scores = cross_validation.cross_val_score(lgr, titanic[predictors], titanic["Survived"], cv=3)

print(scores.mean())

submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": prediction

    }) 
import numpy as np

import pandas as pd

from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn import cross_validation

predictors= ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

rfc= RandomForestClassifier(random_state=1, n_estimators = 50, min_samples_split=4, min_samples_leaf=2)

kfr= cross_validation.KFold(titanic.shape[0], random_state=1, n_folds = 3)

scores= cross_validation.cross_val_score(rfc, titanic[predictors], titanic["Survived"], cv=kfr)

print (scores.mean())

submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": prediction

}) 

submission.to_csv("titanic_submission.csv", index=False)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

clf_knn = KNeighborsClassifier(

    n_neighbors=10,

    weights='distance'

    )

clf_knn = clf_knn.fit(titanic[predictors],titanic["Survived"])

score_knn = cross_val_score(clf_knn,titanic[predictors],titanic["Survived"], cv=5).mean()

print(score_knn)
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score

predictors= ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

clf = GradientBoostingClassifier(n_estimators=100, learning_rate = 1.0, max_depth=1)

clf.fit(titanic[predictors],titanic["Survived"])

score = cross_val_score(clf,titanic[predictors],titanic["Survived"],cv=5).mean()

print (score)

test_pred= clf.predict(titanic_test[predictors])

predictions = [(value for value in test_pred)]

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeClassifier

predictors= ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

dt = DecisionTreeClassifier()

dlf = AdaBoostClassifier(n_estimators=100, base_estimator = dt,learning_rate=1)

dlf.fit(titanic[predictors],titanic["Survived"])

score= cross_val_score(dlf,titanic[predictors],titanic["Survived"],cv=5).mean()

print(score)

test_pred= dlf.predict(titanic_test[predictors])

from sklearn import svm

sv= svm.SVC(gamma='auto',kernel='linear', random_state=1,tol=0.0001,max_iter=-1)

sv.fit(titanic[predictors],titanic["Survived"])

score = cross_val_score(sv,titanic[predictors],titanic["Survived"],cv=5).mean()

print (score)