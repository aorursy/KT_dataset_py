import pandas as pd

import numpy as np
data=pd.read_csv('../input/Salaries.csv')
data.info()
data.drop(['Notes'],axis=1,inplace=True)
data.head()
data.set_index('Id')

data['JobTitle'].nunique()
data['JobTitle'].value_counts()
data.info()
for col in ['BasePay', 'OvertimePay', 'OtherPay', 'Benefits']:

    data[col] = pd.to_numeric(data[col], errors='coerce')
data.info()
data['BasePay'].mean()
data['OvertimePay'].max()
data[data['EmployeeName'] == 'JOSEPH DRISCOLL'][['JobTitle','TotalPayBenefits']]
data.iloc[data['TotalPayBenefits'].idxmax()]
data.groupby('Year')['BasePay'].mean()

sum(data[data['Year'] == 2013]['JobTitle'].value_counts() == 1)
sum(data['JobTitle'].str.lower().str.contains('chief', na=False))