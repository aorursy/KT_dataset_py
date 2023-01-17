# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current sessio
data_Test=pd.read_csv("../input/titanic/test.csv")
data_Train=pd.read_csv("../input/titanic/train.csv")
Test=data_Test.copy()
Train=data_Train.copy()
Test
Train
Train.head()
tb=[Train,Test]
Data=pd.concat(tb)
sns.barplot(x="Sex",y="Survived",data=Data)
Data.isnull().sum()
Test
Data["Title"]=Data["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
Data
Data[["Title","Age"]].groupby("Title").sum()
Data["Title"].value_counts()
Data[["Age","Title"]].groupby("Title").mean()
Data["Title"]=Data["Title"].replace(["Mme"],"Mrs")
Data["Title"]=Data["Title"].replace(["Ms","Mlle"],"Miss")
Data["Title"]=Data["Title"].replace(["Countess","Lady","Sir"],"Royal")
Data["Title"]=Data["Title"].replace(["Don","Dona","Jonkheer","Major","Rev","Col","Capt","Dr"],"Rare")
Data[["Age","Title"]].groupby("Title").mean()

for i in Data["Title"]:
    if i=="Master":
        Data["Age"]=Data["Age"].fillna(5)
    elif i=="Miss":
        Data["Age"]=Data["Age"].fillna(22)
    elif i=="Mr":
        Data["Age"]=Data["Age"].fillna(32)
    elif i=="Mrs":
        Data["Age"]=Data["Age"].fillna(40)
    elif i=="Rare":
        Data["Age"]=Data["Age"].fillna(40)
    else:
        Data["Age"]=Data["Age"].fillna(43)
Data["Title"].value_counts()
Data["Fare"]=Data["Fare"].fillna(12)
Data.isnull().sum()
sns.barplot(x="Embarked",y="Fare",data=Data)
Data["Embarked"]=Data["Embarked"].fillna('C')
Data.isnull().sum()
Data=Data.drop("Ticket",axis=1)
Data=Data.drop("Cabin",axis=1)
Data=Data.drop("Name",axis=1)
Data.head()
#Tr=dict(tuple(Data.groupby("Survived")))
#NorTrain=pd.concat(Tr)
#NorTrain=NorTrain.sort_values(by=["PassengerId"])
#NorTrain
NorTrain=Data.loc[lambda Data:~(Data["Survived"].isnull())]
NorTrain
NorTest=Data.loc[lambda Data: Data["Survived"].isnull()]
NorTest
from sklearn import model_selection
from sklearn import svm
help(model_selection)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
ySurvived=NorTrain["Survived"]
NorTrain=NorTrain.drop(["Survived"],axis=1)
pidTrain=NorTrain["PassengerId"]
NorTrain=NorTrain.drop(["PassengerId"],axis=1)
pidFinal=NorTest["PassengerId"]
NorTrain.dtypes
enTrain=pd.get_dummies(NorTrain)
enTest=pd.get_dummies(NorTest)
FinalTrain,Final=enTrain.align(enTest,join='left',axis=1)
Final["Title_Royal"]=Final["Title_Royal"].fillna(FinalTrain["Title_Royal"])
Final
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 20)]
# Number of features to consider at every split
max_features = [0.2, 0.5, 1, 2, 3]
# Maximum number of levels in tree
bootstrap_features=["True", "False"]
verbose=[0,1,2,3]
base_estimator=[SVC(), "None"]


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'bootstrap_features': bootstrap_features,
               'verbose': verbose,
               'base_estimator': base_estimator,
              'oob_score':"True"}

bag_ran=RandomizedSearchCV(BaggingClassifier(),param_distributions = random_grid, n_iter = 100, 
                           cv = 3, random_state=42, n_jobs = -1)
bag_ran.fit(FinalTrain,ySurvived)
bag_ran.best_params_
bag_first=BaggingClassifier(n_estimators=76, max_features=3, bootstrap_features="True",
                           oob_score="True",verbose=2)
f=np.mean(cross_val_score(bag_first, FinalTrain, ySurvived, cv=10))
print(f)
from sklearn.model_selection import GridSearchCV
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num =20)]
# Number of features to consider at every split
max_features = [ 2, 3, 4]
# Maximum number of levels in tree
verbose=[0,2,3]


param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'bootstrap_features': ["False"],
               'verbose': verbose}
bag_grid = GridSearchCV(estimator = BaggingClassifier(), param_grid = param_grid,
                        cv = 3, n_jobs = -1, verbose = 2)
bag_grid.fit(FinalTrain,ySurvived)
bag_grid.best_params_
bag_final=BaggingClassifier(n_estimators=21,max_features=4,bootstrap_features="False" , verbose=0)
f=np.mean(cross_val_score(bag_final, FinalTrain, ySurvived, cv=10))
print(f)
bag_final.fit(FinalTrain,ySurvived)
Predictions=bag_final.predict(Final)
#list(zip(FinalTrain["Pclass","Age", "SibSp", "Parch", "Fare", 
#                    "Sex_female", "Sex_male", "Embarked_C", "Embarked_Q", "Embarked_S", "Title_Master",
#                    "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Rare", "Title_Royal"], bag_final.feature_importances_))#
ID=pidFinal.array
Predictions=Predictions.astype(int)
Submission = pd.DataFrame({
    'PassengerId':ID,
    'Survived':Predictions
})
Submission.to_csv('Submission.csv', index = False)