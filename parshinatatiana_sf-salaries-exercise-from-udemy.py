import pandas as pd
import numpy as np
sal = pd.read_csv('../input/Salaries.csv')
sal.head()
sal.info()
# Let's try to execute mean() function directly
sal['BasePay'].mean()
sal['BasePay']
# Change value Not Provided to Nan
sal[sal['BasePay'].str.contains('Not Provided', na=False)] = np.NaN
sal['BasePay'].mean()
sal['BasePay'] = sal['BasePay'].apply(lambda x : float(x))
sal['BasePay'].mean()
sal[~sal['OvertimePay'].str.contains('Not Provided', na=False)]['OvertimePay'].apply(lambda x : float(x)).max()
sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle']
sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits']

sal.iloc[sal['TotalPayBenefits'].idxmax()]
sal.iloc[sal['TotalPayBenefits'].idxmin()]
sal.groupby('Year')['BasePay'].mean()
sal['JobTitle'].nunique()
sal['JobTitle'].value_counts().head()
sum(sal[sal['Year'] == 2013]['JobTitle'].value_counts() == 1)
sum(sal['JobTitle'].str.lower().str.contains('chief', na=False))
sal['title_len'] = sal['JobTitle'].str.len()
sal[['title_len','TotalPayBenefits']].corr()