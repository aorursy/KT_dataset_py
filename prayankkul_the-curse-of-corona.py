# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read the file and drop

df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')

df.drop('Sno',axis=1,inplace=True)
# basic Statstics for data 

df.info()
# Data missing information

data_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})

data_info=data_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))

data_info=data_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.

                         rename(index={0:'null values (%)'}))

display(data_info)
# Total Number of cases confirmed by dates From 22-JAN-2020 to 30-JAN-2020



df['Last Update'] = pd.to_datetime(df['Last Update'])

df['Last Update'] = df['Last Update'].dt.date

confirmed_case = df.groupby('Last Update')['Confirmed'].agg('sum')

plt.figure(figsize=(15, 7))

confirmed_case.plot.barh()

plt.xlabel('Total Number of Confirmed Cases Day-wise')

plt.ylabel('Date (Jan 2020)')

plt.title('Total Confirmed Cased of Corona Virus worldwide')

for i,v in enumerate(confirmed_case):

    plt.text(v+2, i, str(v), color='black', fontweight='bold')

plt.grid(True)

plt.show()
# Total Number of cases Deaths by dates From 22-JAN-2020 to 30-JAN-2020

confirmed_death = df.groupby('Last Update')['Deaths'].agg('sum')

plt.figure(figsize=(15, 7))

confirmed_death.plot.barh()

plt.xlabel('Total Number of Death Cases Day-wise')

plt.ylabel('Date (Jan 2020)')

plt.title('Total Death Cases of Corona Virus worldwide')

for i,v in enumerate(confirmed_death):

    plt.text(v, i, str(v), color='black', fontweight='bold')

plt.grid(True)

plt.show()
death_rate = confirmed_death/confirmed_case*100

plt.figure(figsize=(15, 7))

death_rate.plot.line()

plt.xlabel('Death Rate Day-wise')

plt.ylabel('Date (Jan 2020)')

plt.title('Death Rate of Corona Virus worldwide')

plt.grid(True)

plt.show()
Countrywise = df.groupby('Country').agg('sum')

display(Countrywise)
plt.figure(1, figsize=(15, 7))

Countrywise['Confirmed'].plot.bar()

plt.xlabel('Country')

plt.ylabel('Total Number of confirmed Cases')

plt.title('Total number confirmed cases of Corona Virus worldwide(Country-wise Analysis)')

plt.grid(False)

plt.show()
plt.figure(1, figsize=(15, 7))

Countrywise['Deaths'].plot.bar()

plt.xlabel('Country')

plt.ylabel('Total Number of Death Case Cases')

plt.title('Total number Death cases of Corona Virus worldwide(Country-wise Analysis)')

plt.grid(False)

plt.show()
statewise = df.groupby(['Country','Province/State']).agg('sum')

display(statewise)
China =statewise.iloc[statewise.index.get_level_values('Country') == 'China']

display(China)
Mainland_China = statewise.iloc[statewise.index.get_level_values('Country') == 'Mainland China']

Mainland_China=Mainland_China.reset_index('Country')

Mainland_China.drop('Country',axis=1,inplace=True)

display(Mainland_China)
plt.figure(1, figsize=(20, 10))

Mainland_China['Confirmed'].plot.barh()

plt.xlabel('Confirmed Cases')

plt.ylabel('States')

plt.title('Total number Confirmed cases of Corona Virus Statewise(Mainland China Statewise Analysis)')

for i,v in enumerate(Mainland_China['Confirmed']):

    plt.text(v, i, str(v), color='black', fontweight='bold')

plt.grid(False)

plt.show()
# Death case Analysis Statewise

plt.figure(1, figsize=(20, 10))

Mainland_China['Deaths'].plot.barh()

plt.xlabel('Death Cases')

plt.ylabel('States')

plt.title('Total number Death cases of Corona Virus Statewise(Mainland China Statewise Analysis)')

for i,v in enumerate(Mainland_China['Deaths']):

    plt.text(v, i, str(v), color='black', fontweight='bold')

plt.grid(False)

plt.show()
# Mean Confirmed Cases and Death Cases in Mainland China

print("Mean Confirmed Cases in Mainland China  {:.1f}".format(Mainland_China['Confirmed'].mean()))

print("Mean Confirmed Cases in Mainland China  {:.1f}".format(Mainland_China['Deaths'].mean()))

print("Total Death Rate in Mainland China is {:.1f}%".format((Mainland_China['Deaths'].mean()/Mainland_China['Confirmed'].mean())*100))
# Mean Confirmed Cases and Death Cases in China

print("Mean Confirmed Cases in China  {:.1f}".format(China['Confirmed'].mean()))

print("Mean Confirmed Cases in China  {:.1f}".format(China['Deaths'].mean()))

print("Total Death Rate in China is {:.1f}%".format((China['Deaths'].mean()/China['Confirmed'].mean())*100))
Date_Statewise = df.groupby(['Last Update','Country','Province/State']).agg('sum')

display(Date_Statewise)
# Each Date wise Analysis

Date_Statewise.iloc[Date_Statewise.index.get_level_values('Country') == 'Mainland China']