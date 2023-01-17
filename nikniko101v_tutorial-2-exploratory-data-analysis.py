# The Python modules uses in this tutorial are:



# - Pandas

# - Numpy

# - Matplotlib

# - Seaborn



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/loans_train.csv')
# Print first 5 rows of the dataframe

df.head(5)
# Print last 5 rows of the dataframe

df.tail(5)
# Print statistical summary for all numerical attributes

df.describe()
df['Gender'].value_counts()
df['Gender'].count()
# Let's start by plotting a histogram of ApplicantIncome using the command:



df['ApplicantIncome'].hist(bins=50)

plt.xlabel('Applicant Income')

plt.ylabel('Number of Applicants')
df.boxplot(column='ApplicantIncome')
# Boxplots of ApplicantIncome, grouped by the categorical attribute Education

df.boxplot(column='ApplicantIncome', by='Education')

plt.title('Boxplot of ApplicantIncome grouped by Education')

plt.suptitle("") # get rid of the automatic 'Boxplot grouped by group_by_column_name' title
df['LoanAmount'].hist(bins=50)

plt.xlabel("Loan Amount")

plt.ylabel('Number of Applicants')
# Boxplot of LoanAmount

df.boxplot(column='LoanAmount')
df.plot.scatter(x='ApplicantIncome', y='LoanAmount')
sns.regplot(x='ApplicantIncome', y='LoanAmount', data=df)
sns.boxplot(x=df['LoanAmount'])
#Violin plot for LoanAmount



sns.violinplot(x=df['LoanAmount'])
#Strip plot for LoanAmount



sns.stripplot(x=df['LoanAmount'], jitter=True)
#Swarm plot for LoanAmount



sns.swarmplot(x=df['LoanAmount'])
sns.violinplot(x='Education', y='ApplicantIncome', data=df)
frequency_table = df['Credit_History'].value_counts(ascending=True)

print('Frequency Table for Credit History:') 

print(frequency_table)
pivot_table = df.pivot_table(values='Loan_Status',

                       index=['Credit_History'],

                       aggfunc=lambda x: x.map({'Y':1, 'N':0}).mean()) 
# Print pivot table

print(pivot_table)
# Plot the frequency table for Credit_History

frequency_table.plot(kind='bar')

plt.xlabel('Credit History')

plt.ylabel('Number of Applicants')

plt.title('Applicants by Credit History')
# Plot pivot table

pivot_table.plot(kind='bar')

plt.xlabel('Credit History')

plt.ylabel('Probability of Getting a Loan')

plt.title('Probability of Getting a Loan by Credit History')

plt.legend().set_visible(False) # we don't need the default legend
stacked_chart = pd.crosstab(df['Credit_History'], df['Loan_Status'])

stacked_chart.plot(kind='bar', stacked=True, color=['red', 'blue'])

plt.ylabel('Number of Applicants')
stacked_chart_gender = pd.crosstab([df['Credit_History'], df['Gender']], df['Loan_Status'])

stacked_chart_gender.plot(kind='bar', stacked=True, color=['red', 'blue'], grid=False)

plt.ylabel('Number of Applicants')