

import pandas as pd



#reading the dataset

ds = pd.read_csv("..//input//bostonlabelleddataset//labelledBoston.csv")



#extracting the target variable from the dataset

target=ds["labelled"]



#now dropping the original Target variable from dataset

newds=ds.drop(columns=["labelled"],axis=1)



newds.columns

#total Observation in dataset =506

print(newds.shape)

#importing the KFold class from the sklearn.model_selection package on sklearn library

#purpose of this class is to perform kFold CrossValidation

from sklearn.model_selection import KFold



#general procedure in k-Fold Cross Validation

#The general procedure is as follows:



# 1. Shuffle the dataset randomly.

# 2. Split the dataset into k groups

# 3. For each unique group:

#    Take the group as a hold out or test data set

#    Take the remaining groups as a training data set

#    Fit a model on the training set and evaluate it on the test set





#mostly no of observations in trains across all folds will be equal,

# and no of observations in test across all folds will be equal



#After callinf kfold.split(ds)

# there will be 5 folds of same size as the  DataSet sizes

# a) each fold will be having TrainSet and testSet

# b) 80% obs in each fold would be in Training set,  20%  in test set or we can call it validation set or hold out set



#creating the instance of KFold CrossValidation Class

#by passing 

# a) n_splits=k, (default =3)

# b) shuffle=true/false; it means shuffle data before splitting into batches 

# c) random_state = any integer values

kfold=KFold(n_splits=5,shuffle=True,random_state=1234)



#now use the kfold.split(datasetwithoutTargetVariable, targetVariable) and it return the index of train and test Data

#output will be a generator object that contains 5 Training index, 5 testing index fold 

generator=kfold.split(newds,target)



#generator is a class in python which can holds the output yielded from a function

#now calling next() method on generator object 5 times and it returns trainIndex,testIndex  of each fold

#fold 1 TrainIndex testIndex

trainIndex1,testIndex1=next(generator)

#fold 2 TrainIndex testIndex

trainIndex2,testIndex2=next(generator)

#fold 3 TrainIndex testIndex

trainIndex3,testIndex3=next(generator)

#fold 4 TrainIndex testIndex

trainIndex4,testIndex4=next(generator)

#fold 5 TrainIndex testIndex

trainIndex5,testIndex5=next(generator)



#note if we call next(generator) 1 more time, it will give error

#because all the five folds are already returned by the generator alread

#so no fold are left inside it. Thats why we will get the error



# next(generator)

#output  -- >>>  Error StopIteration



# so each fold = 80% TrainData, 20% Test Data will be automatically there



#cross Verify



#shape of all test index

print(testIndex1.shape)  # 20% in Test  102/506

print(testIndex2.shape) # 20% in Test  102/506

print(testIndex3.shape) # 20% in Test  102/506

print(testIndex4.shape) # 20% in Test  102/506

print(testIndex5.shape) # 20% in Test  102/506



#to verify the percentage of observation in test data of each fold

(testIndex1.shape[0]/ds.shape[0])*100   # 20% in Test  102/506



#shape of all train index

print(trainIndex1.shape)  # 80% in Trains 

print(trainIndex2.shape)

print(trainIndex3.shape)

print(trainIndex4.shape)

print(trainIndex5.shape)





#to verify the percentage of observation in train data of each fold

(trainIndex1.shape[0]/newds.shape[0])*100   # 80% in Test  102/506



#now if we pass any of the trainIndex or testIndex to dataset we will get 

# those observation from dataset

newds.iloc[trainIndex1,]

# or

newds.iloc[testIndex1,]



# so now we need a 5 model to train and test against this data, to know whether

# our dataset will be a biased dataset i,e the model built from 

#this data set will be generalized Model or not.



#now over these 5 fold we can train and validate the Dataset





from sklearn.linear_model import LogisticRegression



logitModel1=LogisticRegression(random_state=1234)



logitModel2=LogisticRegression(random_state=1234)



logitModel3=LogisticRegression(random_state=1234)



logitModel4=LogisticRegression(random_state=1234)



logitModel5=LogisticRegression(random_state=1234)



trainIndex1.shape

len(target)





#now fitting the trainData on each model

# syntax of fit function model.fit(trainData,targetVariable)

logitModel1=logitModel1.fit(newds.iloc[trainIndex1,],target[trainIndex1])





#predicting with trainData only

predictTraining1=logitModel1.predict(newds.iloc[trainIndex1,])



#now checking the accuracy of model 1

from sklearn.metrics import confusion_matrix

#syntax of confusion_matric(yTargetActualValues,ypredictedvalues)

confusion_matrix(predictTraining1,target[trainIndex1],labels=[0, 1])



#total Training observations=405



#accuracy metric 

#TN =296

#TP =107

# FalsePositive=1

#FalseNegative=0



# so Accracy

# TN + TP /total predictions

accuracyTraining1=((296+107)/404)*100

print(accuracyTraining1)



#Test set accuracy 

#1. predict over testSet

predictTest1=logitModel1.predict(newds.iloc[testIndex1,])



#total Test Observation =102

testIndex1.shape



#calculating the accuracy of test set

confusion_matrix(predictTest1,target[testIndex1],labels=[0,1])



#accuracy metric 

#TN =76

#TP =26

# FalsePositive=0

#FalseNegative=0





# so Accracy

# TN + TP /total predictions

accuracyTest1=((76+26)/102)*100

accuracyTest1







############## Taking Fold 2  :trainSet2, testSet2 ################



#now fitting the trainData on each model

# syntax of fit function model.fit(trainData,targetVariable)

logitModel2=logitModel2.fit(newds.iloc[trainIndex2,],target[trainIndex2])





#predicting with trainData only

predictTraining2=logitModel2.predict(newds.iloc[trainIndex2,])



#now checking the accuracy of model 1

from sklearn.metrics import confusion_matrix

#syntax of confusion_matric(yTargetActualValues,ypredictedvalues)

confusion_matrix(predictTraining2,target[trainIndex2],labels=[0, 1])



#total Training observations=404



#accuracy metric 

#TN =301

#TP =103

# FalsePositive=1

#FalseNegative=0



# so Accracy

# TN + TP /total predictions

accuracyTraining2=((301+103)/405)*100

accuracyTraining2



#Test set accuracy 

#1. predict over testSet

predictTest2=logitModel2.predict(newds.iloc[testIndex2,])



#total Test Observation =102

testIndex2.shape



#calculating the accuracy of test set

confusion_matrix(predictTest2,target[testIndex2],labels=[0,1])



#accuracy metric 

#TN =71

#TP =30

# FalsePositive=0

#FalseNegative=0





# so Accracy

# TN + TP /total predictions

accuracyTest2=((76+26)/102)*100

accuracyTest2





#like this all fold Training and Testing Accuracy metrics can be done



#now for all 5 models, the Training set Accuracy is constanct and Test set Accuracy is also constant

# so we can say the model built out of this dataset will be generalized model.



#so One of the K-fold cross valiation objective is over !!



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



#Quesstion1. what if the model accuracy is not consistent accross all the training set and all hold out i.e Test set of each Folds



### i dont know what to do then ?????





#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
