# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import os

import pandas as pd

from matplotlib import pyplot as plt

import numpy as np

import math
data = pd.read_csv('../input/loan.csv', low_memory=False)

data.shape
# Define a dictionary with description for each column

LCDataDictionary = pd.ExcelFile("../input/LCDataDictionary.xlsx")

print (LCDataDictionary.sheet_names)

datadict = LCDataDictionary.parse("LoanStats")

datadict.head()

loanstatnew = datadict['LoanStatNew'].tolist()

description = datadict['Description'].tolist()

namedict = {}

for i in range(len(loanstatnew)):

    namedict[loanstatnew[i]] = description[i]

#Missing Value Treatment

##1. Check if each column contains missing value

##2. What percentage of missing value for each column

##3. Delete columns with nan% > 97%

column_naper_dict = {} # {'colname': [ missing value percentage]}

print ('columns with nan% > 97% are:')

for column in data:

   if data[column].isnull().sum()>0:

        column_naper_dict[column] = data[column].isnull().sum()/ 887379.0 

        print (column_naper_dict[column])

        if column_naper_dict[column] > 0.97:

            if column in namedict:

                print ("{0:.0f}%".format(column_naper_dict[column]  *100) )

            else:

                print (column)

            data.drop(column, axis=1,inplace='True') 

column_naper_dict
data.replace('n/a', 'nan',inplace=True)

data.fillna('')

data.head()
# Visualization

## bar-plot of loan_amnt bin / funded_amnt_bin

def label_loan_amnt(row):

    if row['loan_amnt'] <= 5000 :

      return '5K and Below'

    if row['loan_amnt'] > 5000 and row['loan_amnt'] <= 10000:

      return '5K-10K'

    if row['loan_amnt'] > 10000 and row['loan_amnt'] <= 15000:

      return '10K-15K'

    if row['loan_amnt'] > 15000 and row['loan_amnt'] <= 20000:

      return '15K-20K'

    if row['loan_amnt'] > 20000 and row['loan_amnt'] <= 25000:

      return '20K-25K'

    if row['loan_amnt'] > 25000 and row['loan_amnt'] <= 30000:

      return '25K-30K'

    if row['loan_amnt'] > 30000 :

      return '30K and Above'

    return 'Other'

data['loan_amnt_bin'] = data.apply (lambda row: label_loan_amnt(row),axis=1)

data['loan_amnt_bin'].value_counts()
import matplotlib

matplotlib.style.use('ggplot')

f = plt.figure(1)

data['loan_amnt'].hist()

plt.xlabel('Loan Amount')

plt.ylabel('Frequency')

plt.title('Histogram of Loan Amount')

plt.grid(True)

f.show()



f = plt.figure(2)

data['loan_amnt_bin'].value_counts().plot.pie(autopct='%1.0f%%',)

plt.title('Pie Chart of Loan Amount')

f.show()
def label_funded_amnt(row):

    if row['funded_amnt'] <= 5000 :

      return '5K and Below'

    if row['funded_amnt'] > 5000 and row['funded_amnt'] <= 10000:

      return '5K-10K'

    if row['funded_amnt'] > 10000 and row['funded_amnt'] <= 15000:

      return '10K-15K'

    if row['funded_amnt'] > 15000 and row['funded_amnt'] <= 20000:

      return '15K-20K'

    if row['funded_amnt'] > 20000 and row['funded_amnt'] <= 25000:

      return '20K-25K'

    if row['funded_amnt'] > 25000 and row['funded_amnt'] <= 30000:

      return '25K-30K'

    if row['funded_amnt'] > 30000 :

      return '30K and Above'

    return 'Other'

data['funded_amnt_bin'] = data.apply (lambda row: label_funded_amnt(row),axis=1)

data['funded_amnt_bin'].value_counts()



f = plt.figure(1)

data['funded_amnt'].hist()

plt.xlabel('Funded Amount')

plt.ylabel('Frequency')

plt.title('Histogram of Funded Amount')

plt.grid(True)

f.show()



f = plt.figure(2)

data['funded_amnt_bin'].value_counts().plot.pie(autopct='%1.0f%%',)

plt.title('Pie Chart of Funded Amount')

f.show()
#Counter_offer_flag: if funded amount is less than loan amount; (Note: no deals has funder amount greater than loan amount)

data['counter_offer_flag'] = data.loan_amnt-data.funded_amnt

print ('%d deals out of 887379 deals are given counter offer' %((data['counter_offer_flag']> 0).sum()))
data['grade'].value_counts().plot.pie(autopct='%1.0f%%',)

plt.title('Pie Chart of Grade')
# funded amount, term, interest rate by loan grade;

data[['funded_amnt','grade']].boxplot(by='grade')

data[['int_rate','grade']].boxplot(by='grade')

# There are two types terms: 36 months and 60 months

#data['term'] = data['term'].str.split(' ').str[1].astype('int')

#data[['term','grade']].groupby('grade').mean()
# Loan Status:

data['loan_status'].value_counts()
def label_loan_closed_flag(row):

    if row['loan_status'] in ['Fully Paid', 'Charged Off', 'Does not meet the credit policy. Status:Fully Paid',

                             'Does not meet the credit policy. Status:Charged Off']:

      return 'closed'

    return 'open'

data['loan_closed_flag'] = data.apply (lambda row: label_loan_closed_flag(row),axis=1)

data['loan_closed_flag'].value_counts()
def label_loan_bad_flag(row):

    if row['loan_status'] in [ 'Charged Off', 'Does not meet the credit policy. Status:Charged Off']:

      return 'bad'

    return 'good'

data['loan_bad_flag'] = data.apply (lambda row: label_loan_bad_flag(row),axis=1)

data['loan_bad_flag'].value_counts()