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
df =  pd.read_csv('../input/train_u6lujuX_CVtuZ9i.csv')
df.head(10)
df.describe()
df['Property_Area'].value_counts()
df['Credit_History'].value_counts()
#plotting the histogram

df['ApplicantIncome'].hist(bins=50)
#boxplot for understanding distribution

df.boxplot(column='ApplicantIncome')
#segregate the education level

df.boxplot(column='ApplicantIncome',by = 'Education')
df['LoanAmount'].hist(bins=50)
df.boxplot(column='LoanAmount')
#categorical variable analysis

temp1 = df['Credit_History'].value_counts(ascending=True)

print("temp1")
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x:x.map({'Y':1,'N':0}).mean())

temp1


temp2
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,4))

ax1 = fig.add_subplot(121)

ax1.set_xlabel('Credit_History')

ax1.set_ylabel('Count of Applicants')

ax1.set_title("Applicants by Credit_History")

temp1.plot(kind='bar')





ax2 = fig.add_subplot(122)

ax2.set_xlabel('Credit_History')

ax2.set_ylabel('probability of getting loan')

ax2.set_title("probability of getting loan by credit history")

temp1.plot(kind='bar')
#stacked bar

temp3 = pd.crosstab(df['Credit_History'],df['Loan_Status'])

temp3.plot(kind='bar', stacked=True, color=['red','blue'],grid=False)
temp4 = df['Gender'].value_counts(ascending=True)
temp4
temp5 = pd.crosstab(df['Gender'],df['Loan_Status'],)

temp5.plot(kind='bar', stacked=True, color=['red','blue'],grid=False)
#check missing values

df.apply(lambda x: sum(x.isnull()),axis=0)
df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)
df.apply(lambda x: sum(x.isnull()), axis=0)
df['Self_Employed'].value_counts()
df['Self_Employed'].fillna('No',inplace=True)
df['LoanAmount_log'] = np.log(df['LoanAmount'])

df['LoanAmount_log'].hist(bins=20)
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

df['TotalIncome_log'] = np.log(df['TotalIncome'])

df['LoanAmount_log'].hist(bins=20)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)

df['Married'].fillna(df['Married'].mode()[0], inplace=True)

df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)

df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)

df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)