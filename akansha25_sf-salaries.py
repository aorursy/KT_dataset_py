import pandas as pd

import numpy as np

salary = pd.read_csv('../input/Salaries.csv')
salary.head()
salary.info()
# Notes seems to be missing a lot of values

salary.drop(['Notes'],axis=1,inplace=True)
salary.head()
salary.set_index('Id')

# convert the pay columns to numeric

for col in ['BasePay', 'OvertimePay', 'OtherPay', 'Benefits']:

    salary[col] = pd.to_numeric(salary[col], errors='coerce')

salary.describe()
salary.info()
salary['BasePay'].mean()

salary['OvertimePay'].max()

salary[salary['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle']

salary[salary['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits']

salary[salary['TotalPayBenefits'] == salary['TotalPayBenefits'].max()]

salary[salary['TotalPayBenefits'] == salary['TotalPayBenefits'].min()]

salary.groupby('Year').mean() ['BasePay']

salary['JobTitle'].nunique()

salary['JobTitle'].value_counts().head(5)

salary['JobTitle'].value_counts()

salary['JobTitle'].apply(lambda str:('chief' in str.lower())).sum()
