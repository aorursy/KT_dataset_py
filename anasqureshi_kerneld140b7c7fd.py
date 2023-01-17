import numpy as np 

import pandas as pd 

sal = pd.read_csv("../input/SF_salary_data_gender.csv")

sal['BasePay'] = pd.to_numeric(sal['BasePay'], errors = 'coerce')

sal['OvertimePay'] = pd.to_numeric(sal['OvertimePay'], errors = 'coerce')

sal['OtherPay'] = pd.to_numeric(sal['OtherPay'], errors = 'coerce')



c = sal['BasePay'].mean()

print(c)

import os

print(os.listdir("../input"))



sal['OvertimePay'].max()
#what is the job title of joseph driscoll ?

sal[sal['Employee Name'] == 'joseph driscoll']['JobTitle']
#what is the name of highest paid person ?

sal[sal['TotalPay']==sal['TotalPay'].max()]['Employee Name']
#lowest paid person

sal[sal['TotalPay']==sal['TotalPay'].min()]['Employee Name']
#average base pay of all employees per year(2011-2014)

sal.groupby('Year').BasePay.mean()

#or

#sal.groupby('year').mean()['BasePay']
#number of unique jobTitles

sal['JobTitle'].nunique()
#top 5 most common jobs

sal['JobTitle'].value_counts().head(5)
#how many job titles were represented by only one person in 2013?

sum(sal[sal['Year']==2013]['JobTitle'].value_counts()==1)
#how many people have chief in their jobtitle?

def chief_string(title):

    if 'chief' in title.lower().split():

        return True

    else:

        return False

  

sum(sal['JobTitle'].apply(lambda x : chief_string(x)))