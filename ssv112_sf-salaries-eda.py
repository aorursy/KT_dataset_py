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
df = pd.read_csv('/kaggle/input/sf-salaries/Salaries.csv')

df.head()

df.info()
import seaborn as sns
sns.heatmap(df.isnull(), yticklabels=False ,)
df.drop(columns='Notes',inplace=True)
df.head()
#Different Job Titles in SF



df['JobTitle'].nunique()
#Number of Jobs reported based on job titles



df['JobTitle'].value_counts()



#HENCE 3 MOST COMMON JOBS ARE Transit Operator ,Special Nurse and Registered Nurse 
#Average Base and Totalpay of dataset

#while  finding , i got 4 entries with base pay not provided hence remvong from calculation



df.drop([148646,148650,148651,148652],inplace=True)
#Average BasePay of dataset

df['BasePay'].astype(float).mean()
#Average Totalpay of dataset

df['TotalPayBenefits'].astype(float).mean()
df.head(1)
#Employees with Max and Min Salary



df[df['TotalPayBenefits']==df['TotalPayBenefits'].max()]['EmployeeName'][0]
df[df['TotalPayBenefits']==df['TotalPayBenefits'].min()]['EmployeeName'].values[0]
#AVERAGE SALARIES BASED ON YEAR AND EMPLOYMENT TITLE



#AVERAGE ON YEAR

df.groupby('Year').mean()['TotalPayBenefits']



#AVERAGE ON EMPLOYMENT TITLE

df.groupby('JobTitle').mean()['TotalPayBenefits'].sort_values(ascending=False)



# MEANS ON AN AVERAGE, A CHIEF INVESTMENT OFFICER EARNS MOST IN SAN FRANCISCO
#CHECKING CORRELATION OF DATA



df.corr()



#SHOWS TOTAL PAY AND TOTAL PAY BENEFIRS ARE HIGHLY CORRELATED WHICH IS CORRECT