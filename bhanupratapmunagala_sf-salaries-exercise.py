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
#Import necessary packages
import numpy as np
import pandas as pd
sal_df = pd.read_csv('../input/Salaries.csv')   
#Head of the DataFrame

sal_df.head(5)
#Analyze Data Distribution
#Method 1.  This will not show string data types
sal_df.describe().transpose()
#Analyze Data Distribution
#Method 2
sal_df.info()
#Average BasePay
sal_df['BasePay'].mean()
#highest amount of OvertimePay

sal_df['OvertimePay'].max()
#Identify JobTitle of a particular person from dataset
sal_df[sal_df['EmployeeName'] == 'GARY JIMENEZ']['JobTitle']
#Identify TotalPayBenefits of a particular person from dataset
sal_df[sal_df['EmployeeName'] == 'PATRICK GARDNER']['TotalPayBenefits']
#Highest paid person 
#Method 1

sal_df.loc[sal_df['TotalPayBenefits'].idxmax()]['EmployeeName']
#Highest Paid Person
#Method 2

sal_df[sal_df['TotalPayBenefits'] == sal_df['TotalPayBenefits'].max()]['EmployeeName']
#Lowest Paid Person

sal_df.loc[sal_df['TotalPayBenefits'].idxmin()]['EmployeeName']

# sal_df[sal_df['TotalPayBenefits'] == sal_df['TotalPayBenefits'].min()]['EmployeeName']
#average  BasePay of all employees per year? (2011-2014)

sal_df.groupby('Year').mean()['BasePay']
# Unique job titles 
# len(sal_df['JobTitle'].unique())

sal_df['JobTitle'].nunique()

#Top 5 most common jobs

sal_df['JobTitle'].value_counts().head(5)
#Job Titles with only one occurence in 2013
#Break-Up: sal_df['Year'] == 2013 & ['JobTitle'].value_counts() == 1 & Sum

sum(sal_df[sal_df['Year'] == 2013]['JobTitle'].value_counts() == 1)

# Count of Job Title with specific word in the data set

def word_check_jobtitle(Title):
    if 'police' in Title.lower().split():
        return True
    else:
        return False
    

sum(sal['JobTitle'].apply(word_check_jobtitle))
#Correlation between two elements

sal_df[['Year','TotalPayBenefits']].corr()