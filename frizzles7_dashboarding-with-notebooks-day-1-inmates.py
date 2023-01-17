# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
% matplotlib inline
import seaborn as sns
sns.set_style('white')

import os
print(os.listdir("../input"))
# read in the data
inmates = pd.read_csv('../input/daily-inmates-in-custody.csv')
inmates.head()
# descriptive statistics
inmates.describe(include='all')
# info
inmates.info()
# let's drop DISCHARGED_DT, as it has no data and TOP_CHARGE, since it's missing a bunch of values
inmates = inmates.drop(['DISCHARGED_DT', 'TOP_CHARGE'], axis=1)
# let's drop the remaining rows missing values
inmates = inmates.dropna()
# updated info
inmates.info()
# visualization for age
plt.figure(figsize=(15,6))
sns.distplot(inmates['AGE'])
plt.show()
# visualization for race
plt.figure(figsize=(15,6))
sns.countplot(inmates['RACE'])
plt.show()
# convert to datetime
inmates['ADMITTED_DT'] = pd.to_datetime(inmates['ADMITTED_DT'])
# calculate difference betweeen admitted and today in years
inmates['YEARS_IN'] = (pd.to_datetime('today') - inmates['ADMITTED_DT']) / pd.Timedelta('365.25 days')
inmates[['ADMITTED_DT','YEARS_IN']].head()
# visualization for YEARS_IN
plt.figure(figsize=(15,6))
sns.distplot(inmates['YEARS_IN'], kde=False)
plt.show()
