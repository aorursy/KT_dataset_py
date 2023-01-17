import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
sal = pd.read_csv('../input/Salaries.csv')
sal.head()
# Use the .info() method to find out how many entries there are



sal.info()#148654 Entries
# Average Base Pay



sal['BasePay'].mean()
# highest amount of OvertimePay in the dataset



sal['OvertimePay'].max()
# job title of JOSEPH DRISCOLL



sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle' ]
# how much JOSEPH DRISCOLL make (including benefits);



sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits']
#  name of highest paid person (including benefits)



x = sal['TotalPayBenefits'].max()



sal[sal['TotalPayBenefits'] == x]
#  name of lowest paid person (including benefits)



x = sal['TotalPayBenefits'].min()



sal[sal['TotalPayBenefits'] == x]
#  average (mean) BasePay of all employees per year? (2011-2014)



sal.groupby('Year').mean()['BasePay']

# unique job titles



sal['JobTitle'].nunique()
# top 5 most common jobs



sal['JobTitle'].value_counts().head(5)
# Job Titles were represented by only one person in 2013



sum(sal[sal['Year']==2013]['JobTitle'].value_counts() == 1)
#  people have the word Chief in their job title



p = []



for i in sal['JobTitle']:

    

    if 'chief' in i.lower():

        

        p.append(i)

        

len(p)