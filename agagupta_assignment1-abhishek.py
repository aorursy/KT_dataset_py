# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fancyimpute import KNN # for apply KNN for missing values
from scipy import stats
import matplotlib as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/train_AV3.csv")
data.isnull().sum()
data.shape #dimensions of dataset
mean=data.mean()
mean
median=data.median()
median
data['Gender'].mode() #categorical data filled by mode of data
data['Married'].mode()

data['Self_Employed'].mode()
data['Dependents'].mode()
data['Gender'].fillna(value='Male',inplace=True) #fillimg the missing value of categorical data by mode
data['Gender'].isnull().sum()
data['Married'].fillna(value='Yes',inplace=True)
data['Self_Employed'].fillna(value='No',inplace=True)
data['Dependents'].fillna(value='0',inplace=True)
data['Married'].isnull().sum()
data['Self_Employed'].isnull().sum()
data['Dependents'].isnull().sum()
data_complete=KNN(k=3).complete(data[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']]) #filling the missing value by KNN using 3 neighbours
data_complete=pd.DataFrame(data_complete,columns=['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History'])
data_complete.isnull().sum()
mean_complete_data=data
mean_complete_data.fillna(value=mean,inplace=True)
mean_complete_data.isnull().sum()
median_complete_data=data
median_complete_data.fillna(value=mean,inplace=True)
median_complete_data.isnull().sum()
data_complete.plot(subplots=True)
mean_complete_data.plot(subplots=True)
median_complete_data.plot(subplots=True)
data_complete.plot.bar()
mean_complete_data[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']].plot.bar()
median_complete_data[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']].plot.bar()
def f1(x):
    if(x<=2875):
        return "L"
    elif(x<=3800):
        return "LM"
    elif(x<=5516):
        return "UM"
    else:
        return "U"
    

data["ApplicantStatus"]=data["ApplicantIncome"].map(f1)
data.drop(["Loan_ID"],axis=1)