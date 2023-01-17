import numpy as np 

import pandas as pd 

import os

from catboost import Pool, CatBoostClassifier, cv

import lightgbm as lgbm

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

traind=pd.read_csv("../input/titanic/train.csv")

testd=pd.read_csv("../input/titanic/test.csv")
traind.head()
testd.head()
traind.describe()
# Check the Missing Values

traind.isnull().sum()

testd.isnull().sum()
del traind["PassengerId"]

del traind["Ticket"]

del traind["Cabin"]

del testd["PassengerId"]

del testd["Ticket"]

del testd["Cabin"]
# Gropuing by age and sex and finding medians

trainmedians = traind.groupby('Sex')['Age'].median()

testmedians = testd.groupby('Sex')['Age'].median()
trainmedians
testmedians
traind = traind.set_index(['Sex'])

testd = testd.set_index(['Sex'])
traind
# Filling the missing age values with calculated medians

traind['Age'] =traind['Age'].fillna(trainmedians)

testd['Age'] =testd['Age'].fillna(testmedians)
# Resetting the index

traind = traind.reset_index()

testd=testd.reset_index()
# Only keep title from Name column either Mr, Mrs, Miss, Rev like dat



Title = []



for i in range(len(traind)):

    Title.append(traind["Name"][i].split(",")[1].split(".")[0].lstrip().rstrip())

traind["Name"] = Title
Title = []



for i in range(len(testd)):

    Title.append(testd["Name"][i].split(",")[1].split(".")[0].lstrip().rstrip())

testd["Name"] = Title
farmedian=testd["Fare"].median()



index = list(np.where(testd['Fare'].isna())[0])[0]



testd["Fare"][index]=farmedian
# As oonly two embarked values are missing I removed those two rows

traind=traind.dropna()
traind
traind.isnull().sum()
testd.isnull().sum()
labels=traind["Survived"]
del traind["Survived"]
traind
testd
traind.dtypes
cat_features_index = np.where(traind.dtypes != float)[0]
xtrain,xtest,ytrain,ytest = train_test_split(traind,labels,train_size=.85)
model = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True)
model.fit(xtrain,ytrain,cat_features=cat_features_index,eval_set=(xtest,ytest))
pred = model.predict(testd)

pred = pred.astype(np.int)
testpass=pd.read_csv("../input/titanic/test.csv")

submission = pd.DataFrame({'PassengerId':testpass['PassengerId'],'Survived':pred})
submission.to_csv('catboost.csv',index=False)