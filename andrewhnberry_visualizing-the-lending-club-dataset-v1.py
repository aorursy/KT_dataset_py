# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing the holy trinity of data science plugins
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import numpy as np
import matplotlib.pyplot as plt 

#other visualzation packages
import seaborn as sns
df = pd.read_csv('/kaggle/input/lending-club-loan-data/loan.csv', 
                 low_memory = False)
df.head(3)

df.shape
#Check percentage of data missing for each feature/column
round(df.isna().sum()/len(df),3)
df.dropna(thresh = 0.7*(len(df)), axis = 1, inplace = True)
round(df.isna().sum()/len(df),3)
#Current shape
#df.shape
#Lets beging my looking at the loan status. 
plt.figure(figsize = (14,7))
sns.countplot(y = df.loan_status, data = df, order = df.loan_status.value_counts().index)
plt.title("Loan Status Count")
plt.show()

#Let's look at the purpose of these loans
plt.figure(figsize = (14,7))
sns.countplot(y = df.purpose, data = df, order = df.purpose.value_counts().index)
plt.title("Loan Status Count")
plt.show()
#Let's look at the distribution of loan amounts
plt.figure(figsize = (18,6))

#fig 1
plt.subplot(1,2,1)
plt.hist(x = df.loan_amnt, bins = 30)
plt.title("Histogram of the loan amount for all loans in dataset")
plt.xlabel("Loan Amount")
plt.ylabel("Distribution Count")
plt.axvline(df.loan_amnt.median(), color='red', linestyle='dashed', linewidth=2.5)


#fig 2
plt.subplot(1,2,2)
sns.boxplot(x = df['loan_amnt'])
plt.title("Box Plot of the loan amount for all loans in dataset")
plt.xlabel("Interest Rates %")
plt.ylabel("Distribution Count")

plt.show()
print(f'The biggest loan given on comes to an ammount of ${max(df.loan_amnt)}.')
print(f'The smallest loan given on comes to an ammount of ${min(df.loan_amnt)}.')
#Let's look at the Interest Rates 
plt.figure(figsize = (18,6))

#fig 1
plt.subplot(1,2,1)
plt.hist(x = df['int_rate'], bins = 30)
plt.title("Histogram of the interest for all loans in dataset")
plt.xlabel("Interest Rates %")
plt.ylabel("Distribution Count")
plt.axvline(df.int_rate.median(), color='red', linestyle='dashed', linewidth=2.5)

#fig 2
plt.subplot(1,2,2)
sns.boxplot(x = df['int_rate'])
plt.title("Box Plot of the interest for all loans in dataset")
plt.xlabel("Interest Rates %")
plt.ylabel("Distribution Count")

plt.show()
#Let's look at the grade of the loans
plt.figure(figsize = (14,7))
ax = sns.countplot(x = df.grade, data = df, order = df.grade.value_counts().index)

total = len(df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height+5,
           '{:1.2f}%'.format(height/total*100),ha = 'center')

plt.title("Grade Status Count")
plt.show()
grade_int = df.groupby(['grade'])['int_rate'].mean()
plt.figure(figsize = (12,6))
ax = grade_int.plot(kind = 'bar', color = 'red')

total = 100
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height+1,
           '{:1.2f}%'.format(height/total*100),ha = 'center', fontsize = 12)

ax.spines['top'].set_visible(False) 
ax.spines['right'].set_visible(False)
    
plt.title('Average Interest Rates For Each Loan Grade.', fontsize = 14)
plt.ylabel('Interest Rates (%)')
plt.xlabel('Loan Grade')
plt.xticks(rotation = 0)
plt.show()
# Lets measure how many loans were created each year. 
# First we need to create a new column year, and transform the issue_date
# feature to year
df['year'] = pd.to_datetime(df.issue_d).dt.year
plt.figure(figsize = (14,7))
sns.countplot(x = df.year, data = df)
plt.title("Amount of Loans Given Per Year")
plt.show()