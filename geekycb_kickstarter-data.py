import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv  = '../input/ks-projects-201801.csv'

#import csv and set index column to 'ID'
df = pd.read_csv(csv, index_col = 'ID')
data = pd.read_csv(csv,index_col = 7,parse_dates=[5, 7])
df['launched'] = pd.to_datetime(df['launched'], format='%Y-%m-%d %H:%M:%S')
df['deadline'] = pd.to_datetime(df['deadline'], format='%Y-%m-%d')
data['usd_pledged'] = data['usd pledged']
data = data.drop('usd pledged', axis=1)
#Plot number of pojects by category
%matplotlib inline
count_data = df['main_category'].value_counts()
ax = count_data.plot(kind = 'bar', figsize = [10, 10], width=0.6, alpha=0.6, grid=False)
ax.set_xticklabels(count_data.index,rotation=45, rotation_mode='anchor', ha='right')
ax.yaxis.grid(True)
ax.set_title('Kickstarter Categories by the most Projects created')
ax.set_xlabel('Main Categories of Projets')
ax.set_ylabel('Number of Projects Funded')
plt.show()
# What is the statistical breakdown of pledge amounts in USD?
df['usd_pledged_real'].max()
df.usd_pledged_real.describe().apply(lambda x: format(x, 'f'))
#The breakdown of state of the various Kickstarter Projects
s_f = df['state'] 
totals = s_f.value_counts()
totals

# Calculate the difference between the amount of USD pledged and the final amount pledged in USD
df['diff'] = (df['usd_pledged_real'] / df['usd pledged']  ) * 100

#Only those projects that were funded with less than 100% difference
funded_differnece_lt_100 = df[df['diff']  < 100]
#Select only the US
usd_pledged_in_us = funded_differnece_lt_100[df['country'] == 'US']
# Only the succesful projects in the US
success = usd_pledged_in_us[usd_pledged_in_us['state'] == 'successful']
success.head()
year17 = data.loc['2017-01-01':'2017-12-01']
#Total amount pledged at the end of the month in 2017
year17.pledged.loc['2017-01-01':'2017-12-01'].resample('1M').sum()
# Find the mean given by in 2017
year17.pledged.loc['2017-01-01':'2017-12-01'].resample('1M').mean()
# Find the mean given by day of the week
year17.drop(['ID'], axis=1).groupby(year17.index.weekday_name).mean()
#remove data from 1970
no_outliers = data.loc['2008-12-01': '2018-01-31']
no_outliers.head()
import matplotlib.pyplot as plt
%matplotlib inline
x = no_outliers.index  
y = no_outliers.pledged


fig, ax = plt.subplots( figsize = (15,15))
fig.suptitle('Trend from 2009- 2018', fontsize=16)
fig.text(0.5, 0.04, 'Year', ha='center', va='center')
fig.text(0.06, 0.5, 'Amount Pledged in Billions', ha='center', va='center', rotation='vertical')
no_outliers.pledged.plot(ax=ax)
plt.show()


%matplotlib inline
#Resample the mean pledged every 12 months
no_outliers.usd_pledged.resample('12M').mean().plot( style=':')
no_outliers.usd_goal_real.resample('12M').mean().plot( style='--')
plt.legend(['usd pledged', 'usd_goal_real'], loc ='upper left')
plt.title('Comparsion of Amount pledged Vs Goal reached')
plt.xlabel('Year Launched')
plt.ylabel('Mean amount raised over year')
import matplotlib.pyplot as plt
%matplotlib inline
#Resample the mean pledged every 3 months
no_outliers.usd_pledged.resample('3M').mean().plot( style=':')
no_outliers.usd_goal_real.resample('3M').mean().plot( style='--')
plt.legend(['usd pledged', 'usd goal real'], loc ='upper left')
plt.title('Comparsion of Amount pledged Vs Goal reached')
plt.xlabel('Year Launched')
plt.ylabel('Mean amount raised over year')
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(10,10))

wd =  no_outliers.usd_pledged.groupby(no_outliers.index.dayofweek).mean()
wd = no_outliers.usd_pledged.groupby(no_outliers.index.dayofweek).mean()

wd.index =['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']

wd.plot(style=':')
