import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import numpy as np
from sklearn import preprocessing
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
bank_data = pd.read_csv('../input/bank-marketing-analysis/bank-additional-full.csv', sep=';')
print(bank_data.shape)
print(list(bank_data.columns))
bank_data.head(3)
bank_data.info()
## Grouping 'basic.xy' in the education column to be 'basic'

bank_data['education'] = np.where(bank_data['education'] == 'basic.4y', 'Basic', bank_data['education'])
bank_data['education'] = np.where(bank_data['education'] == 'basic.6y', 'Basic', bank_data['education'])
bank_data['education'] = np.where(bank_data['education'] == 'basic.9y', 'Basic', bank_data['education'])
## Checking the values of y, which is the target variable

bank_data['y'].value_counts()
sns.countplot(x='y', data=bank_data, palette='hls')
plt.show()
## In order to simplify the conversion rate computations, encode output variable as 1 for 'yes' and 0 for 'no'

bank_data['conversion'] = bank_data['y'].apply(lambda x: 1 if x == 'yes' else 0)
## Calculate the conversion rate

conversions = bank_data['conversion'].sum()
total_clients = len(bank_data)
conversion_rate = conversions / total_clients * 100
conversion_rate
## Analyze how the conversion rate varies by different age groups

age_conversions = bank_data.groupby('age')['conversion'].sum()
age_total_clients = bank_data.groupby('age')['conversion'].count()
conversions_by_age = age_conversions / age_total_clients * 100.0
## Plotting this

ax = conversions_by_age.plot(grid=True, figsize=(10, 7), title='Conversion Rates by Age')
ax.set_xlabel('Age')
ax.set_ylabel('Conversion rate (%)')
plt.show()
ax = age_total_clients.plot(grid=True, figsize=(10, 7), title='Total Clients by Age')
ax.set_xlabel('Age')
ax.set_ylabel('Total Clients')
plt.show()
bank_data['age_group'] = pd.cut(bank_data['age'], [18, 30, 40, 50, 60, 100])
## Re-calculate conversion rate by age group and plot

conversions_by_age_group = bank_data.groupby('age_group')['conversion'].sum()
total_clients_age_group = bank_data.groupby('age_group')['conversion'].count()
conversions_by_age_group = conversions_by_age_group / total_clients_age_group * 100.0
ax = conversions_by_age_group.plot(kind='bar', color='skyblue', grid=True, figsize=(10,7), title = 'Conversion Rates by Age Group')

ax.set_xlabel('Age Group')
ax.set_ylabel('Conversion Rate (%)')

plt.show()
## Comparing the distributions of marital status amongst the converted and non-converted groups

conversions_by_marital_status = pd.pivot_table(bank_data, values='y', index='marital', columns='conversion', aggfunc=len)
conversions_by_marital_status
conversions_by_marital_status.plot(kind='pie', figsize=(15, 7), startangle=90, subplots=True, autopct=lambda x: '%0.1f%%' % x)

plt.show()
age_marital = bank_data.groupby(['age_group', 'marital'])['conversion'].sum().unstack('marital').fillna(0)
age_marital
age_marital = age_marital.divide(bank_data.groupby('age_group')['conversion'].count(), axis=0)
age_marital
ax = age_marital.plot(kind='bar', stacked=True, grid=True, figsize=(10,7))

ax.set_title('Conversion Rates by Age & Marital Status')
ax.set_xlabel('Age Group')
ax.set_ylabel('Conversion Rate (%)')

plt.show()
bank_data.groupby('job').mean()
pd.crosstab(bank_data['job'], bank_data['y']).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
bank_data.groupby('marital').mean()
table=pd.crosstab(bank_data['marital'], bank_data['y'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
bank_data.groupby('education').mean()
table=pd.crosstab(bank_data['education'], bank_data['y'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Education vs Purchase')
plt.xlabel('Education')
plt.ylabel('Proportion of Customers')
pd.crosstab(bank_data['month'], bank_data['y']).plot(kind='bar')
plt.title('Purchase Frequency for Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')
pd.crosstab(bank_data['day_of_week'], bank_data['y']).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')
pd.crosstab(bank_data['poutcome'], bank_data['y']).plot(kind='bar')
plt.title('Purchase Frequency for Poutcome')
plt.xlabel('Poutcome')
plt.ylabel('Frequency of Purchase')



















