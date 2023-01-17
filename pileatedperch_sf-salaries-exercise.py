import numpy as np
import pandas as pd
sal = pd.read_csv('../input/Salaries.csv', na_values='Not Provided')
sal.head()
sal.info()
sal['BasePay'].mean()
sal['OvertimePay'].max()
sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['JobTitle']
sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['TotalPayBenefits']
ind = sal['TotalPayBenefits'].idxmax()
sal.loc[ind]['EmployeeName']
ind = sal['TotalPayBenefits'].idxmin()
sal.iloc[ind]
sal.groupby('Year').mean()['BasePay']
sal['JobTitle'].nunique()
sal['JobTitle'].value_counts().head()
(sal[sal['Year']==2013]['JobTitle'].value_counts()==1).sum()
sal['JobTitle'].apply(lambda str:('chief' in str.lower())).sum()