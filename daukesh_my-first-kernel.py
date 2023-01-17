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
!pwd
sal = pd.read_csv('../input/Salaries.csv')
sal.head()
sal.info()
sal['BasePay'].head()
sal['BasePay'].values
sal.dtypes
sal[sal['EmployeeName'] == 'Joseph Driscoll']
sal[sal['EmployeeName'] == 'Joseph Driscoll']['JobTitle']
sal[sal['EmployeeName'] == 'Joseph Driscoll']['TotalPayBenefits']
sal.iloc[sal['TotalPayBenefits'].idxmax()]
highestPaid = sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()

sal[highestPaid]
lowestPaid = sal['TotalPayBenefits'] == sal['TotalPayBenefits'].min()

sal[lowestPaid]
sal['TotalPayBenefits'].dropna().mean()
sal['BasePay'].head()
salBasePay = sal.dropna(axis = 0, how='any', subset=['BasePay'])

salBasePay = salBasePay[salBasePay['BasePay'] != 'Not Provided']

salBasePay['BasePay'] = salBasePay['BasePay'].astype(float)

salBasePay.dtypes

                                                     
df = salBasePay.groupby('Year').mean()['BasePay']

df
sal['JobTitle'].nunique()
sal['JobTitle'].value_counts()[0:6]
sal.head()
state = sal[sal['Year']==2013]['JobTitle'].value_counts() == 1

state.sum()
def chief_string(title):

    if 'chief' in title.split():

        return True

    else:

        return False

    

sal['JobTitle'].apply(lambda x : chief_string(x))
sum(sal['JobTitle'].apply(lambda x : chief_string(x)))