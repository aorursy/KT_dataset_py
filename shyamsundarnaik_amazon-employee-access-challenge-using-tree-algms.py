import os

print(os.listdir("../input/amazon-employee-access-challenge/"))
#Load the training dataset and importing basic packages

import pandas as pd

import numpy as np

trainDf = pd.read_csv('../input/amazon-employee-access-challenge/train.csv')

testDf = pd.read_csv('../input/amazon-employee-access-challenge/test.csv')
#Observing how first five rows look like of train dataset

trainDf.head()
#Observing test dataset

testDf.head()
#Check the different columns and types of train Dataset

trainDf.dtypes
#Check the different columns and types of test Dataset

testDf.dtypes
#Checking volume of train dataset

trainDf.shape
#Checking volume of test dataset

testDf.shape
#Check the null values 

trainDf.isna().sum()
for i in trainDf.columns:

    print(i, trainDf[i].nunique())
num_cont, num_desc = [],[]

for i in trainDf.columns:

    if trainDf[i].nunique() > 350:

        num_cont.append(i)

    else:

        num_desc.append(i)

print(num_cont)

print(num_desc)
len(list(trainDf.columns))
#Checking what kind of value does the Resource Column have

import seaborn as sns

import matplotlib.pyplot as plt

for i in trainDf.var().index:

    sns.distplot(trainDf[i],kde=False)

    plt.show()
plt.figure(figsize=(20,10))

sns.heatmap(trainDf.corr())
#importing the Libraries

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier

from xgboost import XGBClassifier
#Setting Y and X 

y=trainDf['ACTION']

x=trainDf.drop('ACTION',axis=1)
#Split data set into into train and test

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=43)
#Display the shape of the train and test datasets

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#Which are models used for the classifier

models=[DecisionTreeClassifier(),RandomForestClassifier(),BaggingClassifier(),XGBClassifier(),]
#import libraries for confusion matrix and accuracy score

from sklearn.metrics import confusion_matrix,accuracy_score

final_accuracy_scores=[]

#Iterate each model

for i in models:

    dt=i

    #Make the model suitable for X_train and Y_train

    dt.fit(X_train,y_train)

    #Predict the test dataset

    dt.predict(X_test)

    dt.predict(X_train)

    print('Model used for predicting')

    print(i)

    print("Confusion matrix of test dataset")

    print(confusion_matrix(y_test,dt.predict(X_test)))

    print('Accuracy Score of test dataset')

    print(accuracy_score(y_test,dt.predict(X_test)))

    print(confusion_matrix(y_train,dt.predict(X_train)))

    print(accuracy_score(y_train,dt.predict(X_train)))

    final_accuracy_scores.append([i,confusion_matrix(y_test,dt.predict(X_test)),accuracy_score(y_test,dt.predict(X_test)),confusion_matrix(y_train,dt.predict(X_train)),accuracy_score(y_train,dt.predict(X_train))])

    from sklearn.model_selection import cross_val_score

    #Crossfold Validation score for each model

    print(cross_val_score(i,X_train,y_train,cv=10))

    print('**************************************************************************************************')

    
for i in range(len(final_accuracy_scores)):

    a= final_accuracy_scores[i]

    #Sensitivity of the model

    cMatrix = a[1]

    #Sensitivity = True Positive Rate = TP/(TP+FN)--(Condition positive)

    Sensitivity = cMatrix[0][0]/(cMatrix[0][0]+cMatrix[1][0])

    #Specificity = True Negative Rate = TN/(FP+TN)-- Condition Negative

    Specificity = cMatrix[1][1]/(cMatrix[1][1]+cMatrix[0][1])

    print(a[0])

    print("Sensitivity of Model ", Sensitivity)

    print("Specificity of Model", Specificity)
#Check the test dataset 2 rows

testDf.head(2)
#Drop the column id

testx = testDf.drop(['id'],axis=1)
#Using the Bagging Classifier model as providing good a accuracy

model = BaggingClassifier()

model.fit(x,y)

#Predicting the test data

testy=model.predict(testx)
#Getting the test data in the series

Action = pd.Series(testy)
#Combine the id and action to show the results

results = pd.DataFrame({'id':testDf['id'],'Action':Action})
#Storing the results in the file

results.to_csv("Submission.csv",index=False)
results.shape
results.shape[0]-trainDf.shape[0]
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
catboost.CatBoostClassifier