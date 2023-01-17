# -*- coding: utf-8 -*-

"""

Created on Sun Mar 22 12:33:29 2020



@author: ravilocalaccount

"""



#introduction :- In this notebook, i will demo 2 things

    # 1st: i take a dataset and build a model around it, and will print the accuracy of model over Training set

    

    #Drawback : in this appraoch we never know, that whats the learning capability of the model

    #           as we directly trained it, and got a training & test accuracy of the model.

    #            So we would never know whether it is underlearn or overlearn(Overfitting) until we do not test it on 

    #            other Observation or testData set.

    

    # so thats the probelm with direct Training  of model,

    # and we donot need to jump right away on conclusion that our model is learning good or learning bad.

    

    

    # 2nd : a) In this approach , we would do cross-validation on model, and get a accuracy of the model.

    #       so this value will tell us how much a model can learn when trained on actual Traning dataset.

    #       Therefor CV helps us in knowing the skill of model on the dataset, without inactual training the model on it.



    # once cv done, then we could do actual training of model on dataset, and from the training and test accuracy we could know whether the model

    # is doing underfitting or overfitting.

    

    

################  First Process ###################

import pandas as pd

#this is a labelled data, which means each observation is classified into 0 and 1 class

boston=pd.read_csv("../input/boston/labelledBoston.csv")



print("To see shape of the data ",boston.shape)



#to see  columns of dataset

print("No of columns in dataset ",boston.columns)

print("\n")

#extract the target variable

target=boston["labelled"]



#drop the targetvariable from 

boston=boston.drop(columns=["labelled"],axis=1)

#split the dataset into traing and test



#import the LinearRegression model from sklearn library

from sklearn.linear_model import LogisticRegression

#create the instance of LinearRegression model

model=LogisticRegression(solver='lbfgs',max_iter=2000)





#now split the dataset into traing and test set

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(boston,target)



#to see the no of observation in

print("No of observation and features in Training set" ,xtrain.shape)

print("No of observation and features in Test set" ,xtest.shape)

print("No of observation and features in Target variable for Training set" ,ytrain.shape)

print("No of observation and features in Target Variable for Testing set" ,ytest.shape)





#now train the model

model=model.fit(xtrain,ytrain)



#now see the Training Accuracy of the model

# if training accuracy is poor, it means the model is underlearning (i.e Underfitting)

# Generally we predict the values on Test set to know the accuraccy of the model

#but here we will calculate training accuracy to know that the model is underlearning or not.



#it return the target value for each observation

predictTrain=model.predict(xtrain)



#now use the confusion matrix to find the 

from sklearn.metrics import confusion_matrix

# we pass the predicted values and actual values to know the confusion matrix

trainingmetric=confusion_matrix(predictTrain,ytrain,labels=[0,1])



print(trainingmetric)

#accuracy metric 

#TN =?

#TP =?

# FalsePositive=?

#FalseNegative=?





print("\n")

tnTraining=trainingmetric[0][0]

print("TrueNegative in Training ",tnTraining)

tpTraining=trainingmetric[1][1]

print("TruePositive in Training ",tpTraining)

print("\n")









# so Accracy

# TN + TP /total predictions

accuracyTraining1=((tnTraining+tpTraining)/379)*100



print("\n")

print("Training Accuracy Results ----> " ,accuracyTraining1)



#now predicting the Test set values

predictTest=model.predict(xtest)



testmetric=confusion_matrix(predictTest,ytest,labels=[0,1])



print(testmetric)

#accuracy metric 

#TN =?

#TP =?

# FalsePositive=?

#FalseNegative=?



# so Accracy

# TN + TP /total predictions



testmetric.shape

tnTest=testmetric[0][0]

print("True Negative TestMetic ", tnTest)

tpTest=testmetric[1][1]



print("\n")

print("True Positive TestMetric ", tpTest)



accuracyTest1=((tnTest+tpTest)/127)*100



print("\n")

print("Test Accuracy Results ----> " ,accuracyTest1)



print("\n")

#NOW Training Accuracy is not matching with Testing accuracy, it means the model is over learning definately



# so this is the drawback with Direct model building, which means we need to wait for entire Training process  to get over with and then find the accuracy of training

# then find the test accuracy of the model , compare both of them to know whethher model is underlearning or overlearning



#normally a Training process can span 2, 3 hours when dataset is too large, thats why we dont want to waste the resources and time 

# without knowing that how much a model shall learn.





#################### Process 2 :- Using Cross-validation to know the skill of the model.

#importing the KFold class from sklearn.model_selection module

from sklearn.model_selection import KFold

#creating the instance of the KFold class

kfold=KFold(n_splits=5,shuffle=True,random_state=1234)

#splitting the dataset and targetvariable into 5 folds

generator=kfold.split(boston,target)



#creating the model

modelCV=LogisticRegression(solver='lbfgs',max_iter=2000)



#importing cross_val_score class, to see the model score on each trained fold.

from sklearn.model_selection import cross_val_score

#it will hold all the 5 accuracy scores of each fold of the model

scores=cross_val_score(modelCV,boston,target,cv=kfold,scoring="accuracy")



#output will be an array of size 5, with test accuracy of each fold

print("scores of each fold test accuracy due to Cross Vaidation process---> ",scores)



print("\n")

#output will be single values, which shows the learning capability of the model

print("mean scores of all folds due to Cross Vaidation process---->",scores.mean())



print("\n")

#so learning capacity of the model is

print(" learning capacity of the model is intimated by Cross Vaidation process ---->",(scores.mean() *100))



learningCapacity=(scores.mean() *100)



# so if now split the dataset into train and test, 

       # 1 and find a accuracy < less learning accuracy , Then it means model is underlearning

       

       # 1 and find a accuracy > above learning accuracy , Then it means model is overlearning



#as we know from first process the testAccuracy is 

print("Test accuracy from 1st process -- >", accuracyTest1)       

#& 

print("\n")

print(" learning capacity of the model is ---->",(scores.mean() *100))



if(accuracyTest1 < learningCapacity):

    print("Model is underLearning & The reason of it due to Noisy data" )



print("\n")

if(accuracyTest1 > learningCapacity):

    print("Model is OverLearning & The reason of it due to Noisy data" )    





#Note :- Noisy Data can leads to any of the above phenomenon OVER-LEARNING and UNDER-LEARNING also
























































































