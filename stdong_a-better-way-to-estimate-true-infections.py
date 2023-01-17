from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import matplotlib.ticker as plticker  #ticker control

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df1 = pd.read_csv('/kaggle/input/covid19testing/tested_worldwide.csv', delimiter=',') 

df1.dataframeName = 'tested_worldwide.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
df1[df1.Country_Region=='Canada']
df1_on=df1[df1.Province_State=='Ontario'].dropna(subset=['daily_tested'])

df1_on=df1_on.assign(daily_rate=df1_on.daily_positive/df1_on.daily_tested*100)

#df1_on['daily_pos_rate']=df1_on.daily_positive/df1_on.daily_tested*100

df1_on.daily_rate.replace(np.inf, 0, inplace=True)

df1_on.daily_rate.fillna(0, inplace=True)

df1_on.reset_index(drop=True, inplace=True)

df1_on.head()
fig, ax = plt.subplots()

ax.plot(df1_on.Date, df1_on.daily_tested,'x-')

plt.title('Daily Number of Tests')

plt.xticks(rotation=90)

loc = plticker.MultipleLocator(base=5.0) 

ax.xaxis.set_major_locator(loc)

plt.show()
fig, ax = plt.subplots()

ax.plot(df1_on.Date, df1_on.daily_rate,'x-')

plt.title('Daily Positive Test Rate')

plt.xticks(rotation=90)

loc = plticker.MultipleLocator(base=5.0) 

ax.xaxis.set_major_locator(loc)

plt.show()
base_pct=df1_on['daily_rate'].iloc[19:30].mean() # calculate baseline

base_pct
df1_on=df1_on.assign(scale_factor=df1_on.daily_rate/base_pct) # define scale factor

df1_on.drop(df1_on.index[range(19)],inplace=True)  #drop the first 19 rows, where no significant testing took place

df1_on.head()
df1_on=df1_on.assign(daily_positive_exp=round(df1_on.daily_positive*df1_on.scale_factor), positive_exp=np.nan) #expected daily positive cases

df1_on.loc[19,'positive_exp']=df1_on.loc[19,'positive']

for i  in range(1,len(df1_on)):

    df1_on.loc[19+i,'positive_exp']=df1_on.loc[18+i,'positive_exp']+df1_on.loc[19+i, 'daily_positive_exp'] #expected total positive cases

df1_on

df1_on=df1_on.assign(positive_exp2=df1_on.death*100,daily_positive_exp2=np.nan) #expected total positive cases based on deaths

df1_on.loc[19,'daily_positive_exp2']=0

for i  in range(1,len(df1_on)):

    df1_on.loc[19+i,'daily_positive_exp2']=100*(df1_on.loc[19+i,'death']-df1_on.loc[18+i, 'death']) #expected daily positive cases
fig, ax = plt.subplots()

ax.plot(df1_on.Date, df1_on.daily_positive,'x-', label='reported')

ax.plot(df1_on.Date, df1_on.daily_positive_exp,'x-', label='estimate from tests')

plt.title('Daily New Infections')

plt.xticks(rotation=90)

loc = plticker.MultipleLocator(base=5.0) 

ax.xaxis.set_major_locator(loc)

plt.legend()

plt.show()
fig, ax = plt.subplots()

ax.plot(df1_on.Date, df1_on.positive,'x-', label='reported')

ax.plot(df1_on.Date, df1_on.positive_exp,'x-', label='estimate from tests')

plt.title('Total Infections')

plt.xticks(rotation=90)

loc = plticker.MultipleLocator(base=5.0) 

ax.xaxis.set_major_locator(loc)

plt.legend()

plt.show()
fig, ax = plt.subplots()

ax.plot(df1_on.Date, df1_on.daily_positive_exp2,'x-', label='estimate from deaths')

ax.plot(df1_on.Date, df1_on.daily_positive_exp,'x-', label='estimate from tests')

plt.title('Daily New Infections')

plt.xticks(rotation=90)

loc = plticker.MultipleLocator(base=5.0) 

ax.xaxis.set_major_locator(loc)

plt.legend()

plt.show()
fig, ax = plt.subplots()

ax.plot(df1_on.Date, df1_on.positive_exp2,'x-', label='estimate from deaths')

ax.plot(df1_on.Date, df1_on.positive_exp,'x-', label='estimate from tests')

plt.title('Total Infections')

plt.xticks(rotation=90)

loc = plticker.MultipleLocator(base=5.0) 

ax.xaxis.set_major_locator(loc)

plt.legend()

plt.show()
fig, ax = plt.subplots()

ax.plot(df1_on.index,df1_on.positive_exp2,'x-', label='estimate from deaths')

ax.plot(df1_on.index+3,df1_on.positive_exp,'x-', label='estimate from tests') #delay the results by 3 days

plt.title('Total Infections')

plt.xticks(rotation=90)

loc = plticker.MultipleLocator(base=5.0) 

ax.xaxis.set_major_locator(loc)

plt.legend()

plt.show()