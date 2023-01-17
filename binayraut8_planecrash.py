# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline



import matplotlib

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#plane_cresh_file = '../input/planecrashinfo_20181121001952.csv'



df= pd.read_csv('../input/planecrashinfo_20181121001952.csv')
df.head(5)


print(f'There are {df.shape[0]} rows in this dataset')

print(f'There are {df.shape[1]} columns in this dataset')
df.info()
df['time'] = df['time'].map(lambda x: x.lstrip('c'))

df['aboard']= df['aboard'].str[0:4]

df['fatalities']= df['fatalities'].str[0:4]

df['aboard'] = df['aboard'].map(lambda x: x.rstrip())

df['fatalities'] = df['fatalities'].map(lambda x: x.rstrip())

df.replace('?',np.nan, inplace = True)

df.head(5)
df['date'] = pd.to_datetime(df['date'])

df['aboard'] = pd.to_numeric(df['aboard'])

df['fatalities'] = pd.to_numeric(df['fatalities'])

df['ground'] = pd.to_numeric(df['ground'])

df['total_death'] = df['fatalities'] + df['ground']
df['year'] =  pd.DatetimeIndex(df['date']).year

df['month'] = pd.DatetimeIndex(df['date']).month

df['day'] =  pd.DatetimeIndex(df['date']).day
df.info()
print(f'Total fatalities {int(df.fatalities.sum())} ')

print(f'Total aboard {int(df.aboard.sum())} ')

print(f'Total ground fatalities {int(df.ground.sum())} ')
plt.hist(df.year,bins= 50, histtype='bar', color='g')
df1= df.groupby(df['year']).sum()

df1 = df1.drop(['month'], axis = 1)

df1 = df1.drop(['day'], axis= 1)

df1.plot(figsize= (20,8))
#df.loc[df['year'] == 2001]

#df.loc[df['ground'] == df['ground'].max()]

df.sort_values('ground', ascending= False).head()



df.loc[df['ground'] == df['ground'].max()]

df.loc[5017,'ground'] = 0

df.loc[5017,'total_death'] = df.loc[5017,'fatalities'] + df.loc[5017,'ground']

df1= df.groupby(df['year']).sum()

df1 = df1.drop(['month'], axis = 1)

df1 = df1.drop(['day'], axis= 1)

df1.plot(figsize= (20,8))
#df1['total_death'].plot(figsize = (20,8))

average_total_death= int(df1.total_death.mean())

colors = tuple(np.where(df1['total_death'] < average_total_death,'g','r'))

a = df1['total_death'].plot(kind = 'bar',alpha =0.7, title = 'Total Deaths',figsize=(25,10), fontsize = 9, color = colors)

plt.axhline(y=average_total_death, color='r', linestyle='--')

plt.show()
fatalities_colors = tuple(np.where(df1['fatalities'] < int(df1['fatalities'].mean()) ,'g','r'))

b = df1['fatalities'].plot(kind = 'bar',alpha =0.7, title = 'Total yearly fatalities',figsize=(25,10), fontsize = 9, color = fatalities_colors)

plt.axhline(y=int(df1['fatalities'].mean()), color='r', linestyle='--')

plt.show()
ground_colors = tuple(np.where(df1['ground'] < int(df1['ground'].mean()) ,'g','r'))

c = df1['ground'].plot(kind = 'bar',alpha =0.7, title = 'Total yearly ground fatalities',figsize=(25,10), fontsize = 9, color = ground_colors)

plt.axhline(y=int(df1['ground'].mean()), color='r', linestyle='--')

plt.show()
df['death_pct'] = df['fatalities']/df['aboard']

a = df['death_pct'].value_counts()

b= a[1]/a.sum()

print(f'{"{:.2%}".format(b)} of crashes have no survivors')
df['ac_type'].value_counts().head(10).plot(kind = 'bar')
df['operator'].value_counts().head(10).plot(kind ='bar')