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
df_train_file="../input/train_AV3.csv"
train=pd.read_csv(df_train_file)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
train.set_index('Loan_ID',inplace=True)
train

train.dtypes
#Number of null values in rows
train.isnull().sum()
#displaying rows with missing values
train[train.isnull().sum(axis=1)>0]
train.shape
train.mean()
train.median()
train.mode()
#changing strings to integer
train['gender'] = train.Gender.map({'Male':1,'Female':0})
train.drop('Gender',axis=1,inplace=True)
train['married'] = train.Married.map({'Yes':1,'No':0})
train['education'] = train.Education.map({'Graduate':1,'Not Graduate':0})
train['property_area'] = train.Property_Area.map({'Urban':2,'Semiurban':1,'Rural':0})

train.drop(['Married','Education','Property_Area'],axis=1,inplace=True)



train['loan_status'] = train.Loan_Status.map({'Y':1,'N':0})
train.drop('Loan_Status',axis=1,inplace=True)


train['self_employed'] = train.Self_Employed.map({'Yes':1,'No':0})
train.drop('Self_Employed',axis=1,inplace=True)
train

train['coapplicantincome'] = train['CoapplicantIncome'].astype('int64')
train.drop('CoapplicantIncome',axis=1,inplace=True)


train['dependents'] = train.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
train.drop('Dependents',axis=1,inplace=True)
train.dtypes
#Replacing Loan_amount_Term by median,Loan_Amount by mean and all other missing values by mode
train['gender'].fillna(1,inplace=True)


train['self_employed'].fillna(0,inplace=True)
train['Credit_History'].fillna(1,inplace=True)
train['Loan_Amount_Term'].fillna(360,inplace=True)#since median is 360 so its same as replacing with median

train['married'].fillna(1,inplace=True)


train['LoanAmount'].fillna(train['LoanAmount'].mean(),inplace=True)

train['dependents'].fillna(0,inplace=True)
train
from matplotlib import pyplot as plt

plt.scatter(train['dependents'],train['LoanAmount'],color = 'k')
plt.show()
plt.scatter(train['gender'],train['LoanAmount'],color = 'k')
plt.show()
train['ApplicantIncome'].hist(bins=20)
#This is for plotting scatter plot which is done further
Male = train.ApplicantIncome[train.gender==1]
Female = train.ApplicantIncome[train.gender==0]
LoanMale = train.LoanAmount[train.gender==1]
LoanFemale = train.LoanAmount[train.gender==0]
Educated = train.ApplicantIncome[train.education==1]
Noneducated = train.ApplicantIncome[train.education==0]
LoanEducated = train.LoanAmount[train.education==1]
LoanNoneducated = train.LoanAmount[train.education==0]
plt.scatter(LoanMale,Male,color='k')
plt.scatter(LoanFemale,Female,color='c')
plt.title('Gender Inequality')
plt.xlabel('LoanAmount')
plt.ylabel('ApplicantIncome')
plt.show()
#Where black is for educated and blue for not educated
plt.scatter(Educated,LoanEducated,color='k')
plt.scatter(Noneducated,LoanNoneducated,color='c')
plt.xlabel('Income')
plt.ylabel('LoanAmount')
plt.title('Education')
plt.show()
#As it is clear that educated people have more loan Amount
Lower_class = train[train['ApplicantIncome']<2877.50]
Lower_middleclass = train[(train['ApplicantIncome']>2877.50) & (train['ApplicantIncome']<3812.50)]
Upper_middleclass = train[(train['ApplicantIncome']>3812.50) & (train['ApplicantIncome']<5795.00)]
Upper_class = train[train['ApplicantIncome']>5795.00]
Upper_class['Features'] = "Upperclass"
Upper_middleclass['Features'] = "Upper_middleclass"
Lower_middleclass['Features'] = "Lower_midlleclass"
Lower_class['Features'] = "Lower_class" 
train_features = pd.concat([Lower_class,Lower_middleclass,Upper_middleclass,Upper_class])
train_features
#Whole dataset has new column feature
#Finding outliers
train['coapplicantincome'].describe()
train['LoanAmount'].describe()
train.ApplicantIncome.describe()
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
%matplotlib inline
rcParams['figure.figsize'] = 7,6
train[['ApplicantIncome']].boxplot(return_type='dict')
plt.plot()
train[['LoanAmount']].boxplot(return_type='dict')
plt.plot()
train[['coapplicantincome']].boxplot(return_type='dict')
plt.plot()
#these are most probable outliers
Applicant_outlier = train[train['ApplicantIncome']>14549]
Applicant_outlier.shape
Coapplicant_outlier = train[train['coapplicantincome']>9188.00]
Coapplicant_outlier.shape
Loanamount_outlier = train[train['LoanAmount']>375]
Loanamount_outlier.shape
#Any value which is far away from (3*interquartile+3rdquartile) range is most probable outlier
outlier_free = train[(train['LoanAmount']>375) | (train['coapplicantincome']>9188.0) | (train['ApplicantIncome']>14549)==0]  
outlier_free
outlier_free.shape
#Now checking on test dataset
df_test_file="../input/test_AV3.csv"
test=pd.read_csv(df_test_file)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
test.set_index('Loan_ID',inplace=True)
test
#As it is clear that the loanstatus is missing
test.dtypes
test.shape
test.isnull().sum()
#displaying rows with missing values
test[test.isnull().sum(axis=1)>0]
test.mean()
test.mode()

#Converting every string value in integers
test['gender'] = test.Gender.map({'Male':1,'Female':0})
test.drop('Gender',axis=1,inplace=True)
test['married'] = test.Married.map({'Yes':1,'No':0})
test['education'] = test.Education.map({'Graduate':1,'Not Graduate':0})
test['property_area'] = test.Property_Area.map({'Urban':2,'Semiurban':1,'Rural':0})

test.drop(['Married','Education','Property_Area'],axis=1,inplace=True)






test['self_employed'] = test.Self_Employed.map({'Yes':1,'No':0})
test.drop('Self_Employed',axis=1,inplace=True)


test['coapplicantincome'] = test['CoapplicantIncome'].astype('int64')
test.drop('CoapplicantIncome',axis=1,inplace=True)


test['dependents'] = test.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
test.drop('Dependents',axis=1,inplace=True)
test
#Giving missing values of Loan_Amount _Term median and LoanAmount mean
test['gender'].fillna(1,inplace=True)


test['self_employed'].fillna(0,inplace=True)
test['Credit_History'].fillna(1,inplace=True)
test['Loan_Amount_Term'].fillna(360,inplace=True)



test['LoanAmount'].fillna(train['LoanAmount'].mean(),inplace=True)

test['dependents'].fillna(0,inplace=True)
test
from matplotlib import pyplot as plt

plt.scatter(test['dependents'],test['LoanAmount'],color = 'k')
plt.show()
#This is for plotting scatter plot which is done further
Male = test.ApplicantIncome[test.gender==1]
Female = test.ApplicantIncome[test.gender==0]
LoanMale = test.LoanAmount[test.gender==1]
LoanFemale = test.LoanAmount[test.gender==0]
Educated = test.ApplicantIncome[test.education==1]
Noneducated = test.ApplicantIncome[test.education==0]
LoanEducated = test.LoanAmount[test.education==1]
LoanNoneducated = test.LoanAmount[test.education==0]
plt.scatter(LoanMale,Male,color='k')
plt.scatter(LoanFemale,Female,color='c')
plt.title('Gender comparision')
plt.xlabel('LoanAmount')
plt.ylabel('ApplicantIncome')
plt.show()
plt.scatter(Educated,LoanEducated,color='k')
plt.scatter(Noneducated,LoanNoneducated,color='c')
plt.xlabel('Income')
plt.ylabel('LoanAmount')
plt.title('Education')
plt.show()
test['ApplicantIncome'].hist(bins=20)
test.ApplicantIncome.describe()
test.coapplicantincome.describe()
test.LoanAmount.describe()
test[['LoanAmount']].boxplot(return_type='dict')
plt.plot()
test[['coapplicantincome']].boxplot(return_type='dict')
plt.plot()
test[['ApplicantIncome']].boxplot(return_type='dict')
plt.plot()
#Any value which is far away from 3*interquartile range is most probable outlier
outlier_free = test[(test['LoanAmount']>325) | (test['coapplicantincome']>9720.0) | (test['ApplicantIncome']>11648)==0]  
outlier_free.shape
outlier_free