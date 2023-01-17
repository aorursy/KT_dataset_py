import numpy as np

import pandas as pd

import scipy

from scipy import stats

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',25)
data=pd.read_csv("../input/world-billionaires/billionaires.csv")
data.head(10)
data.rename(columns = {'natinality':'nationality'}, inplace = True)
data.head(20)
data.info()
data.describe()
data.describe(include='object')
#int64 --> Numerical discrete

#float64 --> Numerical continuous

#object --> String Categories
data['net_worth']=data['net_worth'].astype('float')
data['year']=data['year'].astype('object')

data['rank']=data['rank'].astype('object')
data.info()
data.describe()
data.describe(include='object')
type(data.columns[0])
data.columns = ['year', 'rank', 'name', 'net_worth', 'age', 'nationality','source_wealth']
data.head()
data['NewCol'] = "Something"
data.head()
data.drop(['NewCol'],inplace=True, axis=1) #drop the column, which is an operation to be performed on whole column



# axis = 1, do operations column wise

# axis = 0, do operation row wise
data['year_new'] = data['year'].astype('object')

data['new_worth'] = data['net_worth'].astype('object')

data['rank'] = data['rank'].astype('object')
data.info()
data['rank'].value_counts()
data['name']
data[['year','rank','net_worth']].head(20)
data.columns
data[(data['nationality'] == 'United States') & (data['year'] == 2019) & (data['net_worth'] > 50 )][['name','rank']]
data.groupby('year')
data['name'].value_counts()
data['source_wealth'].value_counts()
data[(data['name'] == 'Mukesh Ambani')]
data_by_year=data.groupby('year')

data_by_year.describe()
data.groupby(by=["year","age"]).describe()
data_by_year[['net_worth','age']].mean()
data_by_year[['net_worth','age']].median()
data.describe(include='object')
data['net_worth'].mean()
data['nationality'].mode()
plt.figure(figsize=[16,6])

y=data.groupby('year').mean()['age']

xi = list(range(2002,2020))

plt.plot(xi,y, marker ='o', linestyle='--', color='r', label='Square')

plt.xticks(xi,xi)

plt.xlabel('Year')

plt.ylabel('Mean Age for that year')
plt.figure(figsize=[16,6])

y=data.groupby('year').mean()['net_worth']

xi = list(range(2002,2020))

plt.plot(xi,y, marker ='o', linestyle='--', color='r', label='Square')

plt.xticks(xi,xi)



plt.ylabel('Mean Age for that year')

plt.xlabel('Year')
!pip install seaborn 

import seaborn as sns 
filter= ['Bill Gates','Jeff Bezos','Mark Zuckerberg','Paul Allen','Larry Ellison']

data[data['name'].isin(filter)][['name','year','net_worth']]
plt.figure(figsize=(16,6))

filter = ['Bill Gates','Jeff Bezos','Mark Zuckerberg','Paul Allen','Larry Ellison', "Mukesh Ambani"]

comparison = data[data['name'].isin(filter)][['name','year','net_worth']]

sns.lineplot(data=comparison,x='year',y='net_worth',hue='name')

plt.xticks(xi,xi)

plt.xlabel('Year')

plt.ylabel('Net Worth')

plt.title('Wealth of Major Tech Companies Owners over years')

!pip install seaborn 

import seaborn as sns 
plt.figure(figsize=(10,5))

filter = ['Microsoft','Facebook','Bershire Hathway','Wal-Mart','Amazon','LVMH']

comparision = data[data['source_wealth'].isin(filter)][['source_wealth','year','net_worth']]

sns.lineplot(data=comparision,x='year',y='net_worth',hue='source_wealth')

plt.xticks(xi,xi)

plt.xlabel('Year')

plt.ylabel('Net Worth')

plt.title('Wealth of Major Tech Companies Owner over Years')

plt.show()
data.describe()
data_by_year['net_worth'].max()
data_by_year['net_worth'].max() - data_by_year['net_worth'].min()
data_by_year['net_worth'].var()
data_by_year['net_worth'].std()
Q1 = data_by_year['net_worth'].quantile(0.25)

Q3 = data_by_year['net_worth'].quantile(0.75)

IQR = Q3 - Q1

print(IQR)
data['age_std'] = ((data['age'] - data['age'].mean())/ data['age'].std())

data.sort_values(by='age_std').head(10)
# we take difference of values with mean and then divide by SD --> This give you average Devaition w.r.t mean/sd
data['age_std']=((data['age'] - data['age'].mean())/data['age'].std())

data.sort_values(by='age_std').head(10)
data['net_worth_std']=((data['net_worth'] - data['net_worth'].mean())/data['net_worth'].std())

data.sort_values(by='net_worth_std', ascending=False).head(10)