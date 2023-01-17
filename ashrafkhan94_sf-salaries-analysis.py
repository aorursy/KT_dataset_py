from pandas import Series,DataFrame

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



sal = pd.read_csv('/kaggle/input/sf-salaries/Salaries.csv', low_memory=False)
sal.head(2)
sal.info()
avg_basepay = Series(sal[sal['BasePay']!='Not Provided']['BasePay'], dtype='float64').mean()

avg_basepay
sal.iloc[sal['TotalPayBenefits'].idxmax()][['EmployeeName','TotalPayBenefits']]
sal.groupby('Year').mean()['TotalPayBenefits']
sal['JobTitle'].value_counts().head(5)
def find_chief(job_title):

    count=0

    for title in job_title:

        if('CHIEF' in title.upper().split()):

            count+=1

    return count



find_chief(sal['JobTitle'])
sal.groupby('JobTitle').mean()['TotalPayBenefits'].sort_values(ascending=False).head(5)