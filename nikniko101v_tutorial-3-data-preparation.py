import pandas as pd

import numpy as np

from statistics import mode

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/loans-train/loans_train.csv')
# Find the number of nulls/NaNs in the dataset



df.apply(lambda x: sum(x.isnull()), axis=0)
df.boxplot(column='LoanAmount', by=['Education','Self_Employed'], rot=45)



plt.title("Boxplot of LoanAmount grouped by Education and Self_Emlpoyed")



# get rid of the automatic 'Boxplot grouped by group_by_column_name' title

plt.suptitle("")
df['Self_Employed'].value_counts()
df['Self_Employed'].fillna('No', inplace=True)
table = df.pivot_table(values='LoanAmount', 

                       index='Self_Employed', 

                       columns='Education', 

                       aggfunc=np.median)

print(table)
# Define function to return an element of the pivot table

def get_element(x):

    return table.loc[x['Self_Employed'], x['Education']]



# Replace missing values

df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(get_element, axis=1), inplace=True)
# Verify there are no missing values in column LoanAmount



df['LoanAmount'].describe()
table2 = df.pivot_table(values='Dependents', 

                       index='Property_Area', 

                       columns='Education', 

                       aggfunc=lambda x: mode(x))



# Define function to return an element of the pivot table

def get_element2(x):

    return table2.loc[x['Property_Area'], x['Education']]



# Replace missing values

df['Dependents'].fillna(df[df['Dependents'].isnull()].apply(get_element2, axis=1), inplace=True)
# Verify there are no missing values in column Dependents



df['Dependents'].describe()
# Create figure with two subplots

fig = plt.figure(figsize=(16,4))



# Plot LoanAmount

ax1 = fig.add_subplot(1, 2, 1)

ax1.set_title("Histogram of LoanAmount")

ax1.set_xlabel('LoanAmount')

ax1.set_ylabel('Number of Applicants')

df['LoanAmount'].hist(bins=20)



# Plot LoanAmount_log

ax2 = fig.add_subplot(1, 2, 2)

ax2.set_title("Histogram of ApplicantIncome")

ax2.set_xlabel('ApplicantIncome')

ax2.set_ylabel('Number of Applicants')

df['ApplicantIncome'].hist(bins=20) 
# Use a log transformation to decrease the impact of extreme values in column LoanAmount

df['LoanAmount_log'] = np.log(df['LoanAmount'])
# Create TotalIncome column and apply a log transformation

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

df['TotalIncome_log'] = np.log(df['TotalIncome'])
# Create figure with four subplots

fig = plt.figure(figsize=(16,10))



# Plot LoanAmount

ax1 = fig.add_subplot(2, 2, 1)

ax1.set_title("Histogram of LoanAmount")

ax1.set_xlabel('LoanAmount')

ax1.set_ylabel('Number of Applicants')

df['LoanAmount'].hist(bins=20)



# Plot LoanAmount_log

ax2 = fig.add_subplot(2, 2, 2)

ax2.set_title("Histogram of ApplicantIncome")

ax2.set_xlabel('ApplicantIncome')

ax2.set_ylabel('Number of Applicants')

df['ApplicantIncome'].hist(bins=20)



# Plot LoanAmount_log

ax3 = fig.add_subplot(2, 2, 3)

ax3.set_title("Histogram of LoanAmount_log")

ax3.set_xlabel('log(LoanAmount)')

ax3.set_ylabel('Number of Applicants')

df['LoanAmount_log'].hist(bins=20)



# Plot LoanAmount_log

ax4 = fig.add_subplot(2, 2, 4)

ax4.set_title("Histogram of TotalIncome_log")

ax4.set_xlabel('log(ApplicantIncome + CoapplicantIncome)')

ax4.set_ylabel('Number of Applicants')

df['TotalIncome_log'].hist(bins=20) 