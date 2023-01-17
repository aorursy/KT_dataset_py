# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
titanic_train = pd.read_csv("../input/train.csv")
titanic_test = pd.read_csv("../input/test.csv")
titanic_tcopy = titanic_test.copy()
titanic_train.info()
titanic_train.head()
# from sklearn.preprocessing import OneHotEncoder
# onehotencoder = OneHotEncoder(categorical_features = 'all')
# titanic_train = onehotencoder.fit_transform(titanic_train).toarray()
corr_matrix = titanic_train.corr()
corr_matrix["Survived"].sort_values(ascending=False)
print (titanic_train[["Sex", "Survived"]].groupby(['Sex']).mean())
titanic_train['Family'] = titanic_train['SibSp']+titanic_train['Parch']
titanic_train.loc[(titanic_train['Family']<=3),'Family'] = 0
titanic_train.loc[(titanic_train['Family']>3),'Family'] = 1
titanic_tcopy['Family'] = titanic_tcopy['SibSp']+titanic_tcopy['Parch']
titanic_tcopy.loc[(titanic_tcopy['Family']<=3),'Family'] = 0
titanic_tcopy.loc[(titanic_tcopy['Family']>3),'Family'] = 1
print (titanic_train[["Family", "Survived"]].groupby(['Family']).mean())
corr_matrix = titanic_train.corr()
corr_matrix["Survived"].sort_values(ascending=False)
embark = {'C':0, 'Q':1, 'S':2}
titanic_train['Embarked'] = titanic_train['Embarked'].map(embark)
titanic_tcopy['Embarked'] = titanic_tcopy['Embarked'].map(embark)
print (titanic_train[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean())
titanic_train["FareCat"] = pd.cut(titanic_train["Fare"],4)
median = titanic_train["Age"].median()
titanic_train["Age"].fillna(median,inplace=True)
titanic_train["AgeCat"] = pd.cut(titanic_train["Age"],9)
print (titanic_train[["FareCat", "Survived"]].groupby(['FareCat'], as_index=False).mean())
titanic_train.loc[(titanic_train['Fare']>-0.512)&(titanic_train['Fare']<=128.082),'Fare'] = 0
titanic_train.loc[(titanic_train['Fare']>128.082)&(titanic_train['Fare']<=256.165),'Fare'] = 1
titanic_train.loc[(titanic_train['Fare']>256.165)&(titanic_train['Fare']<=384.247),'Fare'] = 2
titanic_train.loc[(titanic_train['Fare']>384.247)&(titanic_train['Fare']<=512.3292),'Fare'] = 3

titanic_tcopy.loc[(titanic_tcopy['Fare']>-0.512)&(titanic_tcopy['Fare']<=128.082),'Fare'] = 0
titanic_tcopy.loc[(titanic_tcopy['Fare']>128.082)&(titanic_tcopy['Fare']<=256.165),'Fare'] = 1
titanic_tcopy.loc[(titanic_tcopy['Fare']>256.165)&(titanic_tcopy['Fare']<=384.247),'Fare'] = 2
titanic_tcopy.loc[(titanic_tcopy['Fare']>384.247)&(titanic_tcopy['Fare']<=512.3292),'Fare'] = 3
print (titanic_train[["Fare", "Survived"]].groupby(['Fare'], as_index=False).mean())
print (titanic_train[["AgeCat", "Survived"]].groupby(['AgeCat'], as_index=False).mean())
titanic_train.loc[(titanic_train['Age']>0.34)&(titanic_train['Age']<=8.378),'Age'] = 0
titanic_train.loc[(titanic_train['Age']>8.378)&(titanic_train['Age']<=16.336),'Age'] = 1
titanic_train.loc[(titanic_train['Age']>16.336)&(titanic_train['Age']<=24.294),'Age'] = 2
titanic_train.loc[(titanic_train['Age']>24.294)&(titanic_train['Age']<=32.252),'Age'] = 3
titanic_train.loc[(titanic_train['Age']>32.252)&(titanic_train['Age']<=40.21),'Age'] = 4
titanic_train.loc[(titanic_train['Age']>40.21)&(titanic_train['Age']<=48.168),'Age'] = 5
titanic_train.loc[(titanic_train['Age']>48.168)&(titanic_train['Age']<=56.126),'Age'] = 6
titanic_train.loc[(titanic_train['Age']>56.126)&(titanic_train['Age']<=64.084),'Age'] = 7
titanic_train.loc[(titanic_train['Age']>64.084)&(titanic_train['Age']<=72.042),'Age'] = 8
titanic_train.loc[(titanic_train['Age']>72.042)&(titanic_train['Age']<=80.0),'Age'] = 9

titanic_tcopy.loc[(titanic_tcopy['Age']>0.34)&(titanic_tcopy['Age']<=8.378),'Age'] = 0
titanic_tcopy.loc[(titanic_tcopy['Age']>8.378)&(titanic_tcopy['Age']<=16.336),'Age'] = 1
titanic_tcopy.loc[(titanic_tcopy['Age']>16.336)&(titanic_tcopy['Age']<=24.294),'Age'] = 2
titanic_tcopy.loc[(titanic_tcopy['Age']>24.294)&(titanic_tcopy['Age']<=32.252),'Age'] = 3
titanic_tcopy.loc[(titanic_tcopy['Age']>32.252)&(titanic_tcopy['Age']<=40.21),'Age'] = 4
titanic_tcopy.loc[(titanic_tcopy['Age']>40.21)&(titanic_tcopy['Age']<=48.168),'Age'] = 5
titanic_tcopy.loc[(titanic_tcopy['Age']>48.168)&(titanic_tcopy['Age']<=56.126),'Age'] = 6
titanic_tcopy.loc[(titanic_tcopy['Age']>56.126)&(titanic_tcopy['Age']<=64.084),'Age'] = 7
titanic_tcopy.loc[(titanic_tcopy['Age']>64.084)&(titanic_tcopy['Age']<=72.042),'Age'] = 8
titanic_tcopy.loc[(titanic_tcopy['Age']>72.042)&(titanic_tcopy['Age']<=80.0),'Age'] = 9

print (titanic_train[["Age", "Survived"]].groupby(['Age'], as_index=False).mean())
titanic_train.loc[titanic_train['Sex']=="male",'Gender'] = 0
titanic_train.loc[titanic_train['Sex'] == "female", 'Gender'] = 1

titanic_tcopy.loc[titanic_tcopy['Sex']=="male",'Gender'] = 0
titanic_tcopy.loc[titanic_tcopy['Sex'] == "female", 'Gender'] = 1
print (titanic_train[["Gender", "Survived"]].groupby(['Gender'], as_index=False).mean())
titanic_train.head()
titanic_train = titanic_train.drop(["PassengerId","Name","Sex","AgeCat","Ticket","FareCat","Cabin","SibSp","Parch"], axis=1)
titanic_tcopy1= titanic_tcopy.copy()
titanic_tcopy = titanic_tcopy.drop(["PassengerId","Name","Sex","Ticket","Cabin","SibSp","Parch"], axis=1)
print (titanic_train[["Fare", "Survived"]].groupby(['Fare'], as_index=False).mean())
corr_matrix = titanic_train.corr()
corr_matrix["Survived"].sort_values(ascending=False)
from sklearn.tree import DecisionTreeClassifier

titanic_train.fillna({"Pclass":3,"Age":titanic_train['Age'].median(),"SibSp":0,"Parch":0,"Fare":titanic_train['Fare'].median(),"Embarked":0,"Gender":0},inplace=True)
titanic_tcopy.fillna({"Pclass":3,"Age":titanic_train['Age'].median(),"SibSp":0,"Parch":0,"Fare":titanic_train['Fare'].median(),"Embarked":0,"Gender":0},inplace=True)
# titanic_tcopy1 = titanic_tcopy1.dropna(subset=["Pclass","Age","SibSp","Parch","Fare","Embarked","Gender"])
titanic_label = titanic_train["Survived"]
titanic_data = titanic_train.drop("Survived",axis=1)
tree_clas = DecisionTreeClassifier()
tree_clas.fit(titanic_data,titanic_label)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

rf_clas = RandomForestClassifier(
    n_estimators=200,
    max_depth=15)
# rf_clas = GradientBoostingClassifier()
# rf_clas = XGBClassifier() #SVC(probability=True)
rf_clas.fit(titanic_data, titanic_label)
from sklearn.metrics import mean_squared_error,accuracy_score

survival_prediction = tree_clas.predict(titanic_data)
tree_mse = mean_squared_error(titanic_label, survival_prediction)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
survival_pred = rf_clas.predict(titanic_data)
rf_mse = mean_squared_error(titanic_label, survival_pred)
rf_rmse = np.sqrt(rf_mse)
accuracy_score(titanic_label,survival_pred)
survivors = tree_clas.predict(titanic_tcopy)
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": survivors
    })
# submission.to_csv('../output/gender_submission.csv', index=False)
