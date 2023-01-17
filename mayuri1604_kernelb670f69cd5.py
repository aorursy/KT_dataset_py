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
#loading the csv file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train = pd.read_csv("../input/loadpred/train_AV3.csv")
train.shape
#sum of all the missing values of each column
train.isnull().sum()
# It is necesaary to impute the missing values of columns like loan amount, loan amount term, credit history.
# imputing the missing values of columns like gender may not be necessary as loan status don't depend on such factors
#statistics of each column of the train data
train.describe()
#checking the dependency of amount of loan taken by a person and it's income
plt.scatter(train['LoanAmount'],train['ApplicantIncome'])
plt.xlabel('Amount of loan taken')
plt.ylabel('Total annual income')
plt.show()
#it is observed that the most of the applicant have income less than 20000.
# Also all the loan ids with missing value of loan amount have their income less than 20000. 
#So taking all the statistics values of applicants having income less than 20000 and imputing the mean in missing value 
less_than_20000 = train[train['ApplicantIncome']<20000]
less_than_20000.describe()
#imputing the missing value of loan amount with the mean (of applicant having income less than 20000)
train['LoanAmount'].fillna(value=141.160069,inplace=True)
train.isnull().sum()
#dependency of amount of loan taken and loan amount term
plt.scatter(train['Loan_Amount_Term'],train['LoanAmount'])
plt.ylabel('Amount of loan taken')
plt.xlabel('loan amount term')
plt.show()
# Most of the applicants have loan amount term as 360. So the imputed missing value must be close to 360.  
#dependency of loan amount term and applicant's income
plt.scatter(train['Loan_Amount_Term'],train['ApplicantIncome'])
plt.ylabel('applicant income')
plt.xlabel('loan amount term')
plt.show()
train[train.Loan_Amount_Term.isnull()]
#it is observed that all the missing values of loan amount term have thier loan amount less than 200
#so imputing the missing value with the mean of only those applicants with loan amount less than 200.(idea similar to knn is used here.)
#computing statistics of applicant with loan amount less than 200 
loan_amount_less_than_200 = train[train['LoanAmount']<200]
loan_amount_less_than_200.describe()
#imputing missing values of loan amount term with mean(only of applicant with loan amount less than 200)
train['Loan_Amount_Term'].fillna(value=340.776699,inplace=True)
train.isnull().sum()
#dependency of credit history and loan amount
plt.scatter(train['Credit_History'],train['LoanAmount'])
plt.ylabel('Amount of loan taken')
plt.xlabel('credit history')
plt.show()
# no such linear dependency found between credit history and loan amount.
#dependency of credit history and applicant income
plt.scatter(train['Credit_History'],train['ApplicantIncome'])
plt.ylabel('Applicant income')
plt.xlabel('credit history')
plt.show()
train.describe()
#no factor found for dependency of credit history
# and loan status have it's strong dependency on credit history.
# So droping the row if credit history is missing
train = train.dropna(subset=['Credit_History'],how='any')
train.isnull().sum()
#converting the yes/no type of loan status into 1/0 type (as plots require float type data)
train.Loan_Status.eq('Y').mul(1)
#converting the graduate/nongraduate into 1/0 type
train.Education=train.Education.eq('Graduate').mul(1) 
#dependency of applicant's income and loan status
plt.scatter(train.ApplicantIncome,train.Loan_Status,color='g')
plt.xlabel('Income')
plt.ylabel('loan status')
plt.show()
#loan status depends on applicant's income, loan amount taken, credit history, education of applicant
# making all these factors as feature columns.
feature_cols = ['ApplicantIncome','LoanAmount','Credit_History','Education']
# training dataframe with feature columns
x = train.loc[:,feature_cols]
y= train.Loan_Status

# regression model of machine learning from skicit learn
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x,y)
#reading the testing dataframe
test = pd.read_csv('../input/loadpred/test_AV3.csv')
test.head()
#imputing missing values in test dataframe with the same as in train dataset
test['LoanAmount'].fillna(value=141.160069,inplace=True)
test = test.dropna(subset=['Credit_History'],how='any')
test.isnull().sum()
test.Education=test.Education.eq('Graduate').mul(1)
#new dataframe of test dataset that contains all the feature columns
x_new = test.loc[:,feature_cols]
#predicting the loan status with regression model
new_pred_class = logreg.predict(x_new)
new_pred_class
#loan status of test data set
pd.DataFrame({'Loan_ID':test.Loan_ID,'Loan_Status':new_pred_class}).set_index('Loan_ID')
