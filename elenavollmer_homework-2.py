# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_train = pd.read_csv("../input/modeltrap/train.csv",low_memory = False) # import training data
data_test = pd.read_csv("../input/modeltrap/test.csv",low_memory = False) # import test data
# Task 1
print("{:.2%}".format(len(data_train[(data_train.default==True)])/len(data_train)),"of the training set are in default") 
# calculate the share of loans in the training data in default
# Task 2
print("The ZIP code with the highest default rate is",data_train.groupby(by='ZIP').default.mean().idxmax())
#first, calculate the default rate per ZIP code, then return the ZIP code with the highest rate
# Task 3
print("The default rate in year 0 is ""{:.2%}".format(data_train.default[data_train.year==0].mean()))# return the default rate in year 0
# Task 4
print("The correlation between Age and Income is ""{:.4}".format(data_train.age.corr(data_train.income))) 
# measure correlation between age & income
# Task 5
from sklearn.ensemble import RandomForestClassifier # import Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state = 42, oob_score = True, n_jobs=-1) # define Gaussian Classifier
y_train = data_train.default # data to be predicted: default
x_train = data_train[['loan_size','payment_timing','education','occupation','income','job_stability','ZIP','rent']] # add variables to be used for forecasting
x_train = pd.get_dummies(data = x_train,columns=['ZIP','occupation']) # create dummies for categorical variables
clf.fit(x_train, y_train) # estimate coefficients
print("The in-sample accuracy is ", clf.score(x_train,y_train)) # test accuracy 1
# seems that all values are predicted correctly
# Task 6
print("The Out-Of-Bag score is",clf.oob_score_) # return the Out-Of-Bag-Score, also almost 1
# check further by using confusion matrix
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
conf_mat = confusion_matrix(y_true=y_train, y_pred=clf.predict(x_train)) #calculate confusion matrix
print('Confusion matrix:\n', conf_mat) # returns confusion matrix
# ALL values are predicted correctly, potentially a sign of overfitting
# Task 7
# design test data input
X_test = data_test[['loan_size','payment_timing','education','occupation','income','job_stability','ZIP','rent']] #create input for test
X_test = pd.get_dummies(data=X_test,columns=['ZIP','occupation']) # create dummies for categorical variables
Y_test = data_test.default # generate Y test values
# predict test values
Y_test_pred = clf.predict(X_test)
print("The out-of-sample accuracy is","{:.2%}".format(clf.score(X_test,Y_test))) # calculate accuracy
#the accuracy is lower than in-sample!
# preparation for next partial exercises
data_test_pred = data_test #create new dataset for next exercises
data_test_pred['prediction']=Y_test_pred #add new column "prediction" to dataset

# Task 8
print("The predicted default rate for non-minorities is","{:.2%}".format(data_test_pred.prediction[data_test_pred.minority==0].mean())) # return rejection rate for non-minorities
# Task 9
print("The predicted default rate for minorities is","{:.2%}".format(data_test_pred.prediction[data_test_pred.minority==1].mean())) # return rejection rate for minorities
# the predicted default rate for minorities is higher than for non-minorities!
# Task 11
# check for demographic parity
# demographic parity: same fractions of positives to all groups
print('Below is the share of accepted applicants by', 1-data_test_pred.groupby(by='sex').mean().prediction) 
#return the prediction of default for both genders
# only slightly differing rates. If at all, the model is biased against male applicants
print('Below is the share of accepted applicants by',1-data_test_pred.groupby(by='minority').mean().prediction) 
# return the predicted default rates for non-minorities & minorities
# some disparity can be observed. Almost all non-minority, but less minority applicants get approved. 
# Hence, this model seems biased against minority applicants.
# Task 12
# check for equal opportunity
# equal opportunity: true positives are same for each group
paid=data_test_pred[(data_test_pred.default==0)] # create subset of loans that were actually repaid
print('Below is the share of incorrectly rejected applicants by',paid.groupby(by='sex').mean().prediction) 
#calculate the proportion of incorrectly rejected applicants by gender
# almost no applicants are rejected incorrectly, of which all are male.
#However, the number is very small (<1%) and can be considered inisgnificant.
print('Below is the share of incorrectly rejected applicants by',paid.groupby(by='minority').mean().prediction) 
#calculate the proportion of incorrectly rejected applicants by minority characteristic
# almost no applicants are rejected incorrectly, of which all are minorities.
# However, the number is very small (<1%) and can be considered inisgnificant.