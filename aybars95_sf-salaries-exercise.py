import numpy as np

import pandas as pd
sal = pd.read_csv('../input/Salaries.csv', na_values='Not Provided')
sal.head()
sal.head() # 148654 Entries
sal['BasePay'].mean()
sal['OvertimePay'].max()
sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle']
sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits']
sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()] #['EmployeeName']
sal[sal['TotalPayBenefits']== sal['TotalPayBenefits'].min()] #['EmployeeName']
sal.groupby('Year').mean()['BasePay']
sal['JobTitle'].nunique()
sal['JobTitle'].value_counts().head(5)
sum(sal[sal['Year']==2013]['JobTitle'].value_counts() == 1)
def chief_string(title):

    if 'chief' in title.lower():

        return True

    else:

        return False
sum(sal['JobTitle'].apply(lambda x: chief_string(x)))
sal['title_len'] = sal['JobTitle'].apply(len)
sal[['title_len','TotalPayBenefits']].corr() # No correlation