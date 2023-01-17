# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/world-bank-wdi-212-health-systems/2.12_Health_systems.csv')

data.head()
data.info()
data.Health_exp_per_capita_USD_2016.describe()
data['health_development_level']=["high" if i>1037 else "low" for i in data.Health_exp_per_capita_USD_2016]

print(data['health_development_level'].value_counts(dropna=False))
#Visual Exploratory Data Analysis



data.boxplot(column='Health_exp_per_capita_USD_2016',by='health_development_level',figsize=(9,15))
data['category']=["best" if i>8000 else "good" if 4000<i<8000 else "average" if 2000<i<4000 else "bad" if 1000<i<2000 else "worst" for i in data.Health_exp_per_capita_USD_2016]

print(data['category'].value_counts(dropna=False))
filtr = data.Health_exp_per_capita_USD_2016 > 8000

f_data = data[filtr]

print(f_data)
#Tidy Data(melting)

data_new = (data[(data['Health_exp_per_capita_USD_2016']>4000)])

melted = pd.melt(frame = data_new, id_vars = 'Country_Region',value_vars = ['Health_exp_pct_GDP_2016','Health_exp_per_capita_USD_2016'])

print(melted)

#pivoting data(reverse of melted)



print(melted.pivot(index='Country_Region',columns='variable',values='value'))
#concatenating

data1 = (data[(data['Health_exp_per_capita_USD_2016']>6000)])

data2 = (data[(data['Health_exp_pct_GDP_2016']<3)])

#vertical concatenate

conc_data_row = pd.concat([data1,data2],axis=0,ignore_index=True)

conc_data_row
#horizontal concatenate

data1 = (data[(data['Health_exp_per_capita_USD_2016']>6000)])

data2 = (data[(data['Health_exp_pct_GDP_2016']<3)])

conc_data_col = pd.concat([data1,data2],axis=1)

conc_data_col
#Data Types

data.dtypes
#Missing Data and Testing with Assert

#find missing values

data.info()
data['Province_State'].value_counts(dropna=False)
#drop non-values

data11 =data

data11['Province_State'].dropna(inplace=True)

assert data11['Province_State'].notnull().all()



#Building Data Frames from Scratch

 # data frames from dictionary

    

country = data.World_Bank_Name

Health_exp = data.Health_exp_per_capita_USD_2016

list_label = ["country","Health_exp"]

list_col =[country, Health_exp]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
#add new column

df['region'] = data.Country_Region



#Broadcasting

df['income'] = 0

df
data.info()
#Visual Exploratory Data Analysis



   #PLOT

data1 = data.loc[:,['Health_exp_pct_GDP_2016','Health_exp_out_of_pocket_pct_2016','Health_exp_per_capita_USD_2016']]

data1.plot(figsize=(15,15))
#SUBPLOTS

data1.plot(subplots=True, figsize=(15,15))

plt.show()
f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True,linewidth=.5,fmt='.1f',ax=ax)

plt.show()
data.plot(kind='scatter',x="Health_exp_per_capita_USD_2016", y="per_capita_exp_PPP_2016",figsize=(10,10))
#hist

data1.plot(kind='hist',y='Health_exp_per_capita_USD_2016',bins=50,range=(0,250),figsize=(10,10))
#histogram subplot with non cumulative and cumulative

fig,axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind='hist',y='Health_exp_per_capita_USD_2016',bins=50,range=(0,250),ax=axes[0])

data1.plot(kind='hist',y='Health_exp_per_capita_USD_2016',bins=50,range=(0,250),ax=axes[1],cumulative=True)

plt.savefig('graph.png')

plt
#Statistical Exploratory Data Analysis



data.describe()
#Indexing Pandas Time Series

data2=data.head()

date_list=['1996-01-10','1996-02-10','1996-03-10','1996-04-11','1996-05-12']

datetime_object=pd.to_datetime(date_list)

data2['date']=datetime_object

data2=data2.set_index('date')

data2
#Select according to date index

print(data2.loc['1996-03-10'])
print(data2.loc['1996-03-10':'1996-05-12'])
#calculate mean according to years

data2.resample('A').mean()
#calculate mean according to months

data2.resample('M').mean()
#As you seen, some values are NaN, to interpolate as linear method

data2.resample('M').first().interpolate('linear')

#and interpolate, do not change real mean

data2.resample('M').mean().interpolate('linear')