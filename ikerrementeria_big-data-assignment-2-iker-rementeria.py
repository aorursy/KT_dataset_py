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





#--------------------Beginning of assignment-----------------------------------------------



#Import the datasets



test_dataset=pd.read_csv("../input/test.csv")

train_dataset=pd.read_csv("../input/train.csv")



#Import the needed commands for the assignement



from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics


#Question: What percentage of your training set loans are in default?



print("Percentage of Training set loans in Default: ", train_dataset.default.mean())

print("Percentage of Test set loans in Default: ", test_dataset.default.mean())

# Question: Which ZIP code has the highest default rate in the training dataset?



#Mean default rate by zip code

train_grouped=train_dataset.groupby(['ZIP']).mean().default

#Sort by default rate

train_grouped=train_grouped.sort_values(ascending=False)



print('Train Data defualt rate by zip code:')

print(train_grouped)



print("The Zip code with the highest default rate in the train dataset is:")

print(train_grouped.head(n=1))

#Check which one is the first year

train_dataset.year.min()



#Deafult rate in the first year

print("Default rate on the first year on the train dataset:", train_dataset.default[train_dataset.year==0].mean())  
#Question: correlation between age and income in the traning dataset



print('Correlation age/income in train dataset:', train_dataset['income'].corr(train_dataset['age']))
#Question: What is the in-sample accuracy? That is, find the accuracy score of the fitted model for predicting the outcomes using the whole training dataset.



#Set X_train to be all the features we chose (specified on the assignment)

X_train=train_dataset[['ZIP','rent','education','income','loan_size','payment_timing','job_stability','occupation']]

#Turn categorical variables into dummies

X_train=pd.get_dummies(X_train, columns=['ZIP','occupation'])

#Set Y_train to be the predicted variable (default)

Y_train=train_dataset.default



#Build the random forest model

model=RandomForestClassifier(n_estimators=100,random_state=42,oob_score=True,n_jobs=-1)



#Fit the model on the training data

model.fit(X_train,Y_train)



#Perform the prediction

Y_pred=model.predict(X_train)



#Accuracy of the preditcion

print('In-sample accuracy: ',metrics.accuracy_score(Y_train,Y_pred))

#Question: Out of bag score of the model



print(model.oob_score_)
#Question: Out of sample accuracy? (run the model fitted above on the test data)







#Set X_test to be all the features we chose (specified on the assignment)

X_test=test_dataset[['ZIP','rent','education','income','loan_size','payment_timing','job_stability','occupation']]

#Turn categorical variables into dummies

X_test=pd.get_dummies(X_test, columns=['ZIP','occupation'])

#Set Y_test to be the predicted variable (default)

Y_test=test_dataset.default



#Perform the prediction on the test data

Y_pred=model.predict(X_test)



#Accuracy of prediction on test data

print('Out-of-sample accuracy:', metrics.accuracy_score(Y_test, Y_pred))

#Questions: What is the predicted average default probability for all non-minority members in the test set? What is the predicted average default probability for all minority members in the test set?



#Get the default predictions from the model and create an array

default_prob = pd.DataFrame(model.predict_proba(X_test), columns=["prob_no_default", "prob_default"]) #Get dataframe of the probabilities from the model

#Get the minorities and sex information from the test dataset

minorities=pd.DataFrame(test_dataset[['minority','sex']])

#Concatenate the X_test (that contains the relevant variables) with default probabilities and minorities information,and the prediction of the model

X_test_full = pd.concat([default_prob, minorities, X_test], axis=1) #Add that data frame as 2 new columns



X_test_full.head()



print('Predicted average default depending on minority:')

print(X_test_full.groupby(['minority']).mean().prob_default)

print('Predicted average default depending on sex:')

print(X_test_full.groupby(['sex']).mean().prob_default)

print('Predicted average default depending on both minority and sex:')

print(X_test_full.groupby(['sex','minority']).mean().prob_default)


#Question: Is the loan granting scheme (the cutoff, not the model) group unaware? 



#Question: Has the loan granting scheme achieved demographic parity? 



#In this version, the aproval is based upon the clasification of each person that the model achieves



X_test_full['Y_pred']= Y_pred #include the default predition of the model for each observation to the dataset



#Acceptance Rate of the model by minority and sex:



print('Share of aproved loans depending on minority:')

print(1-X_test_full.groupby(['minority']).mean().Y_pred)

print('Share of approved loans depending on sex:')

print(1-X_test_full.groupby(['sex']).mean().Y_pred)


#Question: Has the loan granting scheme achieved demographic parity? 



#create a new variable that (depending on the cutoff on the introduction) indicates if the applicant is approved of its loan

X_test_full['accept']=X_test_full.prob_no_default>.5



print('Overall share of aproved loans :')

print(X_test_full.mean().accept)



print('Share of aproved loans depending on minority:')

print(X_test_full.groupby(['minority']).mean().accept)

print('Share of approved loans depending on sex:')

print(X_test_full.groupby(['sex']).mean().accept)

#Question: Is the loan granting scheme equal opportunity? Compare the share of successful non-minority applicants that defaulted to the share of successful minority applicants that defaulted. 



#Note: im using the criteria from the Question 11 (Alternative) to determine aproved vs not aproved loans



#Include the actual default data on the dataset

X_test_full['actual_default']=test_dataset.default

#Create a variable that indicates false positives (this is, applicants that succesfully got a loan but defaulted on it)

X_test_full['false_positive']=X_test_full.accept&test_dataset.default 



print('Overall share of false positives :')

print(X_test_full.mean().false_positive)



print('Share of false positives depending on minority:')

print(X_test_full.groupby(['minority']).mean().false_positive)

print('Share of false positives depending on sex:')

print(X_test_full.groupby(['sex']).mean().false_positive)




