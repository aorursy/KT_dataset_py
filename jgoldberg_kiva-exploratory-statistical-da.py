# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import re
import statsmodels.api as sm
%matplotlib inline
pd.options.display.max_columns = 500

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Read kiva_loans.csv
kiva_df = pd.read_csv("../input/kiva_loans.csv")

#Convert dates to pd.datetime
kiva_df['date'] = pd.to_datetime(kiva_df['date']).dt.year
kiva_df['funded_time'] = pd.to_datetime(kiva_df['funded_time'])
kiva_df['posted_time'] = pd.to_datetime(kiva_df['posted_time'])
kiva_df['disbursed_time'] = pd.to_datetime(kiva_df['disbursed_time'])
print('#Unique Activities:',len(kiva_df['activity'].unique()), '#Unique Sectors:',len(kiva_df['sector'].unique()))
print('Shape:',kiva_df.shape)
kiva_df.head()
loan_df = kiva_df['funded_amount'].describe()[['min','50%','max']].to_frame(name='$funded')
loan_df['$loaned'] = kiva_df['loan_amount'].describe()[['min','50%','max']]
loan_df['term_months'] = kiva_df['term_in_months'].describe()[['min','50%','max']]
loan_df
#Missing value analysis
missing_data = kiva_df.isnull().sum()
missing_data = missing_data.iloc[missing_data.nonzero()]
print(missing_data)
fig, ax = plt.subplots(figsize=(10,6))
ax.barh(missing_data.index, missing_data, alpha=0.5)
ax.set_title('Number of Missing Values')
ax.set_ylabel('Kiva_df Column')
ax.grid(color='gray', linestyle='--', alpha=0.5)
sector_df = kiva_df.groupby('sector')['loan_amount'].count().to_frame(name='Loan Count')
sector_df['$Median Loan'] = kiva_df.groupby('sector')['loan_amount'].median()
sector_df['$Min Loan'] = kiva_df.groupby('sector')['loan_amount'].min()
sector_df['$Max Loan'] = kiva_df.groupby('sector')['loan_amount'].max()
print(sector_df)

plt.figure(figsize=(10,4))
plt.rc('font', size=16)
sector_df['Loan Count'].plot(kind='barh', title='Loan Counts by Sector')

plt.figure(figsize=(10,4))
sector_df[['$Min Loan','$Median Loan']].plot(kind='barh', stacked=True, title='Loan Size by Sector')
plt.figure(figsize=(10,4))
plt.bar(kiva_df['date'],kiva_df['loan_amount'])
plt.show()
kiva_df['#female'] = kiva_df['borrower_genders'].str.count('female')
kiva_df['#male'] = kiva_df['borrower_genders'].str.count('male')
kiva_df['#borrowers'] = kiva_df['#female'] + kiva_df['#male']
print(kiva_df['#borrowers'].describe()[['min','50%','max']])
print(kiva_df['#male'].describe()[['min','50%','max']])
print(kiva_df['#female'].describe()[['min','50%','max']])
counts = kiva_df.groupby('#borrowers')['#borrowers'].count()

plt.figure(figsize=(10,4))
plt.rc('font', size=16)
counts.hist(title='#Borrowers')
#plt.figure()
fig1, ax = plt.subplots(1,1, figsize=(10,6))
sns.distplot(kiva_df['#borrowers'].dropna(), color='g', label='All Borrowers')
ax.legend()
ax.set(xlim=(0,50), xlabel='Number of Borrowers')

fig2, ax = plt.subplots(1,1, figsize=(10,8), sharex=True)
sns.distplot(kiva_df['#male'].dropna(), color='b', label='Males')
sns.distplot(kiva_df['#female'].dropna(), color='red', label='Females')
ax.legend()
ax.set(xlim=(0,10))

#b.set(xlim=(0,50))

#plt.setp(axe)
#plt.tight_layout()

#plt.hist([kiva_df['#borrowers'].dropna(),
#          kiva_df['#male'].dropna()
#          ,kiva_df['#female'].dropna()],
#            bins=20, range=[0,20], stacked=True, 
#              color = ['g','b','r'], alpha=0.3)
#ax = plt.gca()
#ax.xaxis.set_major_locator(MaxNLocator(10))
#plt.legend(loc='best')
#plt.xlabel('Total #Borrowers', fontsize=16)
#plt.ylabel('Number', fontsize=16)
#plt.xticks(fontsize=16)
#plt.yticks(fontsize=16)
#plt.show()
#Relate loaned amount to number of borrowers; limit y<10k to see and trend
fig = plt.figure(figsize=(10,8))
ax = plt.scatter(kiva_df['#borrowers'], kiva_df['loan_amount'], c='r')
plt.xlabel('Number of Borrowers')
plt.ylabel('Size of Loan (USD)')
plt.ylim(0,10000)
plt.show()
#Using statsmodel.api
kiva_df = kiva_df.replace([np.inf, -np.inf], np.nan)
kiva_df = kiva_df.dropna()
model = sm.OLS(kiva_df['loan_amount'].dropna(), sm.add_constant(kiva_df['#borrowers'])).fit()
predictions = model.predict(sm.add_constant(kiva_df['#borrowers']))
model.summary()
#Using sklearn
from sklearn import linear_model
lm = linear_model.LinearRegression()
model = lm.fit(kiva_df['#borrowers'], kiva_df['loan_amount'])
predictions = lm.predict(kiva_df['#borrowers'])
lm.score(kiva_df['#borrowers'], kiva_df['loan_amount'])
#Read Loan_themes_by_region
loans_by_region_df = pd.read_csv("../input/loan_themes_by_region.csv")
print(loans_by_region_df.shape, 
      len(loans_by_region_df['country'].unique()),
      len(loans_by_region_df['region'].unique()),
      len(loans_by_region_df['Partner ID'].unique()),
      len(loans_by_region_df['sector'].unique()),
      len(loans_by_region_df['Loan Theme Type'].unique()),
      len(loans_by_region_df['number'].unique()),
      loans_by_region_df['number'].describe()[['min','50%','max']])

loans_by_region_df.head()
#Read loan_theme_ids
loan_ids_df = pd.read_csv("../input/loan_theme_ids.csv")
print(loan_ids_df.shape)
loan_ids_df.head()
#Read kiva_mpi_region_locations.csv
kiva_locations_df = pd.read_csv("../input/kiva_mpi_region_locations.csv")
print(kiva_locations_df.shape)
print(kiva_locations_df.head())

fig, ax = plt.subplots(1,1, figsize=(10,6))
plt.hist(kiva_locations_df['MPI'].dropna())
#kiva_locations_df.hist('MPI')

