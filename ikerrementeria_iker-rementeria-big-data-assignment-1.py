# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import sklearn.model_selection as sklearn



# Any results you write to the current directory are saved as output.
#load the lending club data as per the assignment 1 instructions

total_data=pd.read_csv("../input/loan.csv", low_memory=False)
#checking the data

total_data.head(n=5)
#only keep data for loans whose status is fully paid or default

data=total_data[(total_data.loan_status=='Fully Paid')|(total_data.loan_status=='Default')]

data['loan_status'].head(n=7)
#create new binary target variable

data['target']=(data['loan_status']=='Fully Paid')
#checking new variable target

data['target'].head(n=7)
#return the shape of the data (observations and columns/variables)

data.shape
#---------------------------------------------------------------Task 1---------------------------------------------------

#(how many record are there?)

len(data)
#----------------------------------------Task 2_----------------------------------------------------------------------
#import matplotlib.pyplot

import matplotlib.pyplot as plt
#plot the hystogram

plt.hist(data.loan_amnt, bins=50)
#Calculate mean, median, max and std of loan amount and funded amount via a loop

loans=['loan_amnt','funded_amnt']

for x in loans:

    print('The mean '+x+' is '+str(data[loans][x].mean()))

    print('The median '+x+' is '+str(data[loans][x].median()))

    print('The maximum '+x+' is '+str(data[loans][x].max()))

    print('The standard deviation '+x+' is '+str(data[loans][x].std()))
#alternative calculation

np.mean(data['loan_amnt'])
# compute median

data.loan_amnt.median()
# compute maximum

data.loan_amnt.max()
# compute standard deviation

data.loan_amnt.std()
#------------------------------------------Task 3------------------------------------------------------------------------
#mean of short and long term loans

data.groupby(by='term').int_rate.mean() #2 different terms: 36 and 60 months
#stdv of short and long term loans 

data.groupby(by='term').int_rate.std()
#boxplot the interest rate depending on the term

data.boxplot('int_rate',by='term',figsize=(7,7))
#---------------------------------------------------TASK 4 -----------------------------------------------------------------------
#Average int_rate depending on the debt grade

data.groupby(by='grade').int_rate.mean()
#average int_rate of the grade with the highest avg interest rate

data.groupby(by='grade').int_rate.mean().max()
#(Alternative)

max(data.groupby(by='grade').int_rate.mean())
#--------------------------------------------------------TASK 5-------------------------------------------------------------------
#Proportion (in percent) of loans in default

(1-data['target'].mean())*100

#creating default rate variable

data['default']=(1-data['target'])*100
#how does the default rate differs depending on grade

data.groupby(by='grade').default.mean()

#it grows as the grade goes down (except for grade G that is 0%)
#how does the interest rate differs depending on grade

data.groupby(by='grade').int_rate.mean()

#it grows as the grade goes down
#create variable yield (I define yield as the total amount collected from interest divided by the funded amount...)

# Also, I assume the interest rate provided is annual interest rate and there is no partial capital repayments; all capital is repaid at maturity)



data['short_t']=(data['term']==' 36 months')

data['long_t']=(data['term']==' 60 months')



data['r_yield']=(data['int_rate']*data['short_t']*3+data['int_rate']*data['long_t']*5)*data['target']

#cheking

print(data['r_yield'].mean())

print(data['term'].head())

print(data['int_rate'].head())

print(data['r_yield'].head())
#highest realized yield for any debt grade

data.groupby(by='grade').r_yield.max()

#---------------------------------------------------Task 6----------------------------------------------------------------------------
#number of records per aplication type 

data.groupby(by='application_type').size()



#It does not make much sens to use this feature for prediction beacause the vast mayority of the observations are of one type
#------------------------------------------------------------Task 7-------------------------------------------------------
#construc the model subdataset of selected features with dummy variables for the categorical variables

model_var=data[['loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate','emp_length','addr_state',

             'verification_status','purpose','policy_code']]

model_var_dum=pd.get_dummies(model_var,columns=['term','emp_length','addr_state','verification_status','purpose'])
#shape of the new datasets

print(model_var.shape) # the original model has: 104+ observations, 10 variables

print(model_var_dum.shape) # the model with the dummy variables: 104+ obsevations, 86 variables

#Set the dependent variable 

target=data[['target']]

print(target.shape)

target_dum=pd.get_dummies(target)

print(target_dum.shape)
#-------------------------------------------------------------Task 8-----------------------------------------------------------------
#split data using sklearn train_test_split (test size = .33, random state= 42)

X_train, X_test, y_train, y_test= sklearn.train_test_split(model_var_dum, 

                                                           target_dum,

                                                           train_size=0.33,

                                                           random_state=42)

#shape of X.train and others

X_train.shape, y_train.shape, X_test.shape, y_test.shape
#-----------------------------------------------------Task 9---------------------------------------------------------------------------------------------------------
#import the relevant commands from sklearn library

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

from sklearn import metrics
#train the using Random Forest Classifier (n_estimators=100,max_depth=4,random_state=42)

rf_model=RandomForestClassifier(n_estimators=100,max_depth=4,random_state=42)
#Fit the model to the training data

rf_model.fit(X_train,y_train)

#Predict target and calculate the accuracy score

target_pred=rf_model.predict(X_test)



metrics.accuracy_score(y_test,target_pred)

#-----------------------------------------------Task 10---------------------------------------------------------------------------------
#repeat task 7-10 using total_data instead of data... and changing names accordingly

#(This is, remove the first step we took after importing the dataset. Where we removed all applicants with loan status other than

# fully paid and default)
#The prediction accuracy will be lower since the data in which we were working before, was overwhelmingly-representated by the Fully paid loan status

#as oposed to the other alternative (Default)... so, basically the model "could" predict that all the aplicants (in that sample) would

#fully pay and the accuracy would still be high (probably thats what is happening)

#The model prediction applied to the whole data (All loan status), would test the real prediction power!