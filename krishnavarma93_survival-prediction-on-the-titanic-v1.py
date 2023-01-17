###Import Necessary modules

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import seaborn as sn
###Read the data
data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")
###Understanding the dataset

data_train.columns
data_test.columns
data_train.info()
data_test.info()
data_train.Embarked.unique()
###Combining train and test data for data preparation
###Dropping the dependent varaible in the train data

entire_data = data_train.drop(["Survived"],axis=1).append(data_test)
len(entire_data)
entire_data.head(5)
###Dropping varaibles that seems less important in the first iteration, let's add them in the next iteration

entire_data.drop(["PassengerId","Name","Ticket","Cabin","Embarked"],axis=1,inplace=True)
entire_data.info()
###filling the missing values of Age and Fare

entire_data.Age.fillna(entire_data.Age.mean(), inplace=True)
entire_data.Fare.fillna(entire_data.Fare.mean(), inplace=True)
entire_data.info()
###Converting Sex varaible to numerical

entire_data.Sex.replace("female",0,inplace=True)
entire_data.Sex.replace("male",1,inplace=True)
entire_data.head(5)
entire_data.describe().T
###Modelling using RandomForest Classifier
train_X = entire_data[0:len(data_train)]
len(train_X)
train_X.info()
train_Y = data_train["Survived"]
len(train_Y)
test = entire_data[len(data_train)::]
len(test)
model=RandomForestClassifier(oob_score=True,n_estimators=100)
model.fit(train_X,train_Y)
print(model.oob_score_)
predict_accuracy = pd.DataFrame({"actual" : train_Y, "predicted" : model.predict(train_X)})
cm = metrics.confusion_matrix(predict_accuracy.actual,predict_accuracy.predicted,[1,0])
sn.heatmap(cm, annot=True,
         fmt='.2f',
         xticklabels = ["Survived", "Not Survived"] , yticklabels = ["Survived", "Not Survived"] )
predict_test = model.predict(test)
submission = pd.DataFrame({"PassengerId": data_test.PassengerId, "Survived": predict_test})
submission.to_csv("my_submission.csv", index=False)