# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df_list = []

for i in os.listdir("../input"):

    df = pd.read_csv('../input/'+i, header = 0, index_col = None)

    df_list.append(df)



bikes = pd.concat(df_list, axis = 0, ignore_index = True)    
bikes.head()
for i in bikes['month'].unique():

    print('Month {}: {}'.format(i, len(bikes[bikes['month'] == i])))
bikes.info()
bikes.isnull().any()
bikes[pd.isna(bikes).any(axis = 1)]
print('Missing value in each month:')

for i in bikes['month'].unique():

    print('Month {}: {}'.format(i,len(bikes[(bikes['month'] == i) & (pd.isna(bikes).any(axis = 1))])))

    
bikes.dropna(inplace = True)
bikes.info()
bikes.describe()
plt.figure(figsize = (12, 8))

sns.set_style('darkgrid')

sns.distplot(bikes['member_birth_year'])
def group_birth(year):

    if year >= 2005 and year <= 2019 :

        return 'Children'

    elif year >= 1995 and year <= 2004:

        return 'Youth'

    elif year >= 1955 and year <= 1994:

        return 'Adult'

    else:

        return 'Senior'

    

bikes['age_group'] = bikes['member_birth_year'].apply(group_birth)        
sns.countplot(bikes['age_group'])
print('Number of Services by:')

for i in bikes['age_group'].unique():

    print('{} used {} times for baywheel services'.format(i, len(bikes[bikes['age_group'] == i])))
sns.countplot(data = bikes, x= 'age_group', hue = 'user_type')
len(bikes[(bikes['age_group'] == 'Senior') & (bikes['user_type'] == 'Customer')])
sns.pairplot(bikes, hue = 'month')
sns.pairplot(bikes, hue = 'age_group')
plt.figure(figsize = (12,8))

sns.distplot(bikes['trip_duration_sec'])
from sklearn.preprocessing import StandardScaler

standard = StandardScaler()

bikes['trip_duration_sec_standard'] = standard.fit_transform(np.array(bikes['trip_duration_sec']).reshape(-1,1))

bikes = bikes[(bikes['trip_duration_sec_standard'] > -3) & (bikes['trip_duration_sec_standard'] < 3)]
len(bikes)
plt.figure(figsize = (12,8))

sns.kdeplot(bikes[bikes['age_group'] == 'Youth']['trip_duration_sec'], color = 'darkred', label ='Youth')

sns.kdeplot(bikes[bikes['age_group'] == 'Adult']['trip_duration_sec'], color = 'blue', label ='Adult')

sns.kdeplot(bikes[bikes['age_group'] == 'Senior']['trip_duration_sec'], color = 'green', label ='Senior')

plt.legend(frameon = False)
plt.figure(figsize = (12,8))

sns.kdeplot(bikes[bikes['age_group'] == 'Youth']['start_station_id'], color = 'darkred', label ='Youth')

sns.kdeplot(bikes[bikes['age_group'] == 'Adult']['start_station_id'], color = 'blue', label ='Adult')

sns.kdeplot(bikes[bikes['age_group'] == 'Senior']['start_station_id'], color = 'green', label ='Senior')

plt.title('Start Station Based on the Age Group')

plt.legend(frameon = False)
plt.figure(figsize = (12,8))

sns.kdeplot(bikes[bikes['age_group'] == 'Youth']['end_station_id'], color = 'darkred', label ='Youth')

sns.kdeplot(bikes[bikes['age_group'] == 'Adult']['end_station_id'], color = 'blue', label ='Adult')

sns.kdeplot(bikes[bikes['age_group'] == 'Senior']['end_station_id'], color = 'green', label ='Senior')

plt.title('End Station Based on the Age Group')

plt.legend(frameon = False)
sns.pairplot(bikes, hue='user_type')