import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Read in the data
data = pd.read_csv('../input/Salaries.csv', index_col='Id', na_values=['Not Provided', 'Not provided'])
## Pre-processing

# Change NaNs in the EmployeeName and JobTitle columns back to 'Not Provided'
data.loc[:, ['EmployeeName', 'JobTitle']] = data.loc[:, ['EmployeeName', 'JobTitle']].apply(lambda x: x.fillna('Not provided'))

# Normalize EmployeeName and JobTitle
data.loc[:, 'EmployeeName'] = data.loc[:, 'EmployeeName'].apply(lambda x: x.lower())
data.loc[:, 'JobTitle'] = data.loc[:, 'JobTitle'].apply(lambda x: x.lower())
bPay = data[pd.notnull(data.BasePay)]

sns.distplot(bPay.BasePay, kde=False)
plt.title('Distribution of Base Pay')
PT = bPay[bPay.Status == 'PT']['BasePay']
FT = bPay[bPay.Status == 'FT']['BasePay']

PT.name = 'Part Time'
FT.name = 'Full Time'

plt.figure(figsize=(8,6))
sns.distplot(PT, kde=False)
sns.distplot(FT, kde=False)
plt.title('Distribution of Full-Time and Part-Time Employee Base Pay')
print('Median firefighter salary: $' + str(np.nanmedian(bPay[bPay.JobTitle == 'firefighter']['BasePay'])))
print('Median police salary: $' + str(np.nanmedian(bPay[bPay.JobTitle == 'police officer']['BasePay'])))
polFir = bPay[np.logical_or(bPay.JobTitle == 'firefighter', bPay.JobTitle == 'police officer')]

g = sns.FacetGrid(polFir, col='Year')
g.map(sns.violinplot, 'JobTitle', 'BasePay')
sns.distplot(bPay[bPay.JobTitle == 'firefighter']['BasePay'], kde=False, color='red')
sns.distplot(bPay[bPay.JobTitle == 'police officer']['BasePay'], kde=False)
plt.title('Distribution of Police and Firefighter Base Pay')
byYear = bPay.groupby('Year').aggregate(np.median)
byYear['Year'] = byYear.index

sns.barplot(data=byYear, x='Year', y='BasePay')