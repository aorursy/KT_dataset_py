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
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/train.csv')
df.head()
df.info()
df.describe()
#high disparity in appl income and loan amount
df['ApplicantIncome'].hist(bins=40)
df.boxplot(column='ApplicantIncome') #Outliers exist
df['LoanAmount'].hist(bins=40)
df.boxplot(column='LoanAmount') #Outliers
df.boxplot(column='LoanAmount', by = 'Self_Employed')
#Not self empployed people ask have some very high loan demands but the mean is similar for both
df.boxplot(column='ApplicantIncome', by = 'Education')
#Mean is similar but some graduates have very high incomes
#calculating missing values
df.isnull().sum()
df.Married.value_counts()
#Since gender married dependents self_employed Loan_Amount_Term and credit history are categorical, we replace nan with the mode
df.Credit_History.fillna(value=1.0, inplace=True)
df.Married.fillna(value='Yes', inplace=True)
df.Gender.fillna(value='Male', inplace=True)
df.Dependents.fillna(value='0', inplace=True)
df.Self_Employed.fillna(value='No', inplace=True)
df.Loan_Amount_Term.fillna(value=360.0, inplace=True)
#LoanAmount has to be approximated
df.LoanAmount.fillna(df['LoanAmount'].mean(), inplace=True)
#Taking natural log for LoanAmount and ApplicantIncome gives us an even distributed system
df['LoanAmount_log'] = np.log(df.LoanAmount)
df.boxplot(column='LoanAmount_log')
df['ApplicantIncome_log'] = np.log(df.ApplicantIncome)
df.boxplot(column='ApplicantIncome_log')
