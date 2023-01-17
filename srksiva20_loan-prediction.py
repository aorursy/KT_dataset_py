# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import accuracy_score





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
#Reading datasets

test = pd.read_csv("../input/test.csv")

train = pd.read_csv ("../input/train.csv")



#Final prediction dataset

submission = pd.DataFrame(0, index=np.arange(0, len(test)), columns=['Loan_ID', 'Loan_Status'])



#Making copies

train_copy=train.copy() 

test_original=test.copy()
#Filling missing values

train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)

train['Married'].fillna(train['Married'].mode()[0], inplace=True)

train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)

train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)

train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)



test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)

test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)

test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)

test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
#Univariate Analysis

#Categorical Features 

plt.figure('Categorical Features')

plt.subplot(221)

train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender')

plt.subplot(222)

train['Married'].value_counts(normalize=True).plot.bar(figsize=(20,10), title='Married')

plt.subplot(223)

train['Self_Employed'].value_counts(normalize=True).plot.bar(figsize=(20,10), title='Self_Employed')

plt.subplot(224)

train['Credit_History'].value_counts(normalize=True).plot.bar(figsize=(20,10), title='Credit_History')
# Bivariate Analysis

# Loan Status based on Gender

Gender = pd.crosstab(train['Gender'],train['Loan_Status'])

Gender.div(Gender.sum(1).astype(float),axis=0).plot(kind="bar",stacked = True,figsize=(4,4),title="Loan Status based on Gender") 



# Loan Status based on Marital Status

Married = pd.crosstab(train['Married'],train['Loan_Status'])

Married.div(Married.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4),title="Loan Status based on Marital Status")



# Loan Status based on Dependents

Dependents = pd.crosstab(train['Dependents'],train['Loan_Status'])

Dependents.div(Dependents.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4),title="Loan Status based on Dependents")



#Loan Status based on Education Status

Education = pd.crosstab(train['Education'],train['Loan_Status'])

Education.div(Education.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4),title="Loan Status based on Education Status")



#Loan Status for Self employed people

Self_Employed = pd.crosstab(train['Self_Employed'],train['Loan_Status'])

Self_Employed.div(Self_Employed.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4),title="Loan Status for Self employed people")



#Loan Status based on credit history

Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])

Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4),title="Loan Status based on credit history")



#Loan Status based ownershop of property area"

Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status']) 

Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,title="Loan Status based ownershop of property area")
# Distribution of Data

train.hist(bins=10,figsize=(9,7),grid=False);
#Applicant Incomes segregated into bins vs Loan Status

bins=[0,2500,4000,6000,81000]

group=['Low','Average','High','Very high']

train_copy['Income_bin']=pd.cut(train_copy['ApplicantIncome'],bins,labels=group)

train_copy['Income_bin']=pd.cut(train_copy['ApplicantIncome'],bins,labels=group)

Income_bin=pd.crosstab(train_copy['Income_bin'],train_copy['Loan_Status'])

Income_bin.div(Income_bin.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)

plt.xlabel('ApplicationIncome')

plt.ylabel('Percentage')

plt.title('Applicant Incomes segregated into bins vs Loan Status')
#Co-Applicant Incomes segregated into bins vs Loan Status

bins=[0,1000,3000,42000]

group=['Low','Average','High']

train_copy['Coapplicant_Income_bin']=pd.cut(train_copy['CoapplicantIncome'],bins,labels=group)

Coapplicant_Income_bin=pd.crosstab(train_copy['Coapplicant_Income_bin'],train_copy['Loan_Status'])

Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)

plt.xlabel('CoapplicantIncome')

plt.ylabel('Percentage')

plt.title('CO - Applicant Incomes segregated into bins vs Loan Status')

#Combining Applicant and Co Applicant income 

train_copy['Total_Income']=train_copy['ApplicantIncome']+train_copy['CoapplicantIncome']

bins=[0,2500,4000,6000,81000]

group=['Low','Average','High','Very high']

train_copy['Total_Income_bin']=pd.cut(train_copy['Total_Income'],bins,labels=group)



Total_Income_bin=pd.crosstab(train_copy['Total_Income_bin'],train_copy['Loan_Status'])

Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.xlabel('Total_Income')

plt.ylabel('Percentage')

plt.title('Combined income into bins vs Loan status')
#dropping loan_id column 

train=train.drop('Loan_ID',axis=1)

test = test.drop('Loan_ID',axis=1)
#Assigning dependent varibale 

X = train.drop('Loan_Status',1)

y = train.Loan_Status
#Filling Dummies

X=pd.get_dummies(X) 

train=pd.get_dummies(train) 

test=pd.get_dummies(test)
#splitting the dataset for training 

x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3, random_state=42)

model = LogisticRegression(random_state=42)

model.fit(x_train, y_train)

pred_cv = model.predict(x_cv)

A = accuracy_score(y_cv,pred_cv)

print('Logistic Regression Accuracy', A)
pred_test = model.predict(test)



submission['Loan_Status']=pred_test

submission['Loan_ID']=test_original['Loan_ID']



submission['Loan_Status'].replace(0, 'N',inplace=True) 

submission['Loan_Status'].replace(1, 'Y',inplace=True)



print(submission)