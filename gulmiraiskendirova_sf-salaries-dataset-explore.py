import numpy as np

import pandas as pd
df = pd.read_csv('../input/sf-salaries/Salaries.csv', low_memory=False)
df.head()
df.info()
objlist = ['BasePay', 'OvertimePay', 'OtherPay', 'Benefits']



for obj in objlist:

    df[obj] = pd.to_numeric(df[obj], errors = 'coerce')
df['BasePay'].mean()
df['OvertimePay'].max()
df[df['TotalPayBenefits'] == df['TotalPayBenefits'].max()]
df[df['TotalPayBenefits'] == df['TotalPayBenefits'].min()]
df['Year'].unique()
df.groupby('Year')['BasePay'].mean()
df['Status'].unique()
df[df['Status'] == 'PT']['BasePay'].max() #highest BasePay for PT employees

df[df['Status'] == 'PT']['BasePay'].min() #lowest BasePay for PT employees
df[df['Status'] == 'FT']['BasePay'].max() #highest BasePay for FT employess
df[df['Status'] == 'FT']['BasePay'].min() #lowest BasePay for FT employees
df.groupby('Status')['BasePay'].mean()
commonJob = df['JobTitle'].value_counts().head()

commonJob

df[['BasePay', 'TotalPayBenefits']].corr()
df['JobTitle'].nunique()