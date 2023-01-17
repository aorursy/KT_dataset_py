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

txt=pd.read_csv('../input/trainwax.csv')
txt.head()                ## printing first 5 rows of the data sets
txt.describe()            ## summarising the data sets
txt.shape                 ## print rows and columns in the format (rows , col )
txt.dtypes                ## data type of each columns
type(txt)                 ## type is dataframe 
## mean , median, mode of the data sets
txt.mean()
txt.mode()
txt.median()
## prints the subplots in different graphs
txt.plot(subplots=True)
cols=['LoanAmount','Credit_History','Loan_Amount_Term','CoapplicantIncome','ApplicantIncome']
ls1=['LoanAmount','Credit_History']
ls2=['Credit_History','Loan_Amount_Term']
ls3=['Loan_Amount_Term','CoapplicantIncome']
ls4=['CoapplicantIncome','ApplicantIncome']

## Histogram
## printing histogram of each column separately
txt['LoanAmount'].plot(kind='hist')
txt['Credit_History'].plot(kind='hist')
txt['Loan_Amount_Term'].plot(kind='hist')
txt['CoapplicantIncome'].plot(kind='hist')

txt[ls1].plot(kind='hist')
txt[ls2].plot(kind='hist')
txt[ls3].plot(kind='hist')
txt[ls4].plot(kind='hist')

## printing box plot 
txt[ls1].plot(kind='box')
txt[ls2].plot(kind='box')
txt[ls3].plot(kind='box')
txt[ls4].plot(kind='box')
txt[cols].plot(kind='box',subplots=True)


## printing bar plot 
txt[ls1].plot(kind='bar')
txt[ls2].plot(kind='bar')
txt[ls3].plot(kind='bar')
txt[ls4].plot(kind='bar')
txt[cols].plot(kind='bar',subplots=True)

## printing density plot 
txt[ls1].plot(kind='density')
txt[ls2].plot(kind='density')
txt[ls3].plot(kind='density')
txt[ls4].plot(kind='density')
txt[cols].plot(kind='density',subplots=True)

