# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
kiva_loans = pd.read_csv("../input/kiva_loans.csv")
kiva_loans.head()
kiva_loans.info()
kiva_loans['sector'].unique()
plt.subplots(figsize=(16,12))
plt.pie(kiva_loans['sector'].value_counts(),labels=kiva_loans['sector'].unique(),autopct='%1.1f%%')
plt.axis('equal')
kiva_loans_by_activity = kiva_loans.groupby(['sector'])['loan_amount'].sum()
kiva_loans_by_activity.reset_index()
kiva_loans_by_activity = pd.DataFrame({'sector':kiva_loans_by_activity.index,'loan_amount':kiva_loans_by_activity.values})
kiva_loans_by_activity = kiva_loans_by_activity.sort_values("loan_amount",ascending=False)
kiva_loans_by_activity.head()
plt.subplots(figsize=(16,12))
sns.barplot(x="sector",y="loan_amount",data=kiva_loans_by_activity)
plt.xlabel('Sector')
plt.ylabel('Loan amount (USD)')
plt.xticks(rotation="vertical")
plt.subplots(figsize=(16,12))
plt.pie(kiva_loans_by_activity['loan_amount'],labels=kiva_loans_by_activity['sector'],autopct='%1.1f%%')
plt.axis('equal')
plt.tight_layout
plt.subplots(figsize=(16,12))
plt.pie(kiva_loans['currency'].value_counts(),labels=kiva_loans['currency'].unique(),autopct='%1.1f%%')
plt.axis('equal')
plt.tight_layout
kiva_loans['borrower_genders'][0:5]
kiva_loans['num_borrowers'] = kiva_loans['borrower_genders'].str.split().str.len()
kiva_loans['num_borrowers'][0:5]
plt.subplots(figsize=(16,12))
kiva_loans['num_borrowers'].value_counts().plot.bar()
plt.tight_layout()
plt.xlabel('No. of borrowers')
plt.ylabel('Total no. of loans')
