# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Macro

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



jpunemployment = pd.read_csv("../input/fdsadczgdsatewedvcxz/jpunyear.csv",names = ['year','unemployment rate'])

jpunemployment = jpunemployment.drop(labels=range(0,44),axis=0)

jpunemployment = jpunemployment.reset_index(drop=True)

#jpunemployment.index = range(len(jpunemployment))

#print(jpunemployment)



jpunemployment['unemployment change'] = jpunemployment['unemployment rate'] / jpunemployment['unemployment rate'][0]



interest= pd.read_csv('../input/fdsadczgdsatewedvcxz/interest.csv',names = ['year','Interest Rate'])

#print(interest)



GDP = pd.read_csv('../input/fdsadczgdsatewedvcxz/GDP.csv',names = ['year','GDP'])

#GDP = GDP.astype(float)

GDP['GDP change'] = GDP['GDP']/GDP['GDP'][0]

#print(GDP)



CPI = pd.read_csv('../input/fdsadczgdsatewedvcxz/CPI.csv',names = ['year','CPI'])#Base on 2015

CPI['CPI change'] = CPI['CPI']/CPI['CPI'][0]

#print(CPI)



Export = pd.read_csv('../input/fdsadczgdsatewedvcxz/export.csv',names = ['year','Export'])

Export['Export change'] = Export['Export']/Export['Export'][0]

#print(Export)



foreign_currency_reserve = pd.read_csv('../input/fdsadczgdsatewedvcxz/foreign currency reserve.csv',names = ['year','foreign currency reserve'])

foreign_currency_reserve['foreign currency reserve change'] = foreign_currency_reserve['foreign currency reserve']/foreign_currency_reserve['foreign currency reserve'][0]

#print(foreign_currency_reserve)



df=pd.DataFrame() #empty dataframe  

df['unemployment change'] = jpunemployment['unemployment change']

df['interest'] = interest['Interest Rate']

df['GDP_change'] = GDP['GDP change']

df['CPI change'] = CPI['CPI change']

df['Export change'] = Export['Export change']

df['foreign currency reserve change'] = foreign_currency_reserve['foreign currency reserve change']

#print(df)

#df[['unemployment change','interest','foreign currency reserve change','Export change','CPI change']].plot(figsize=(12,8)) 

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(1, 1, 1)

#ax.plot(df[['unemployment change','interest','foreign currency reserve change','Export change','CPI change']])

ax.plot(df['unemployment change'],label='unemployment change')

ax.plot(df['interest'],label='interest')

ax.plot(df['CPI change'],label='CPI change')

ax.plot(df['Export change'],label='Export change')

ax.plot(df['foreign currency reserve change'],label='foreign currency reserve change')

ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])

ax.set_xticklabels(['2013','2014','2015','2016(Negative Interest)','2017','2018','2019'], rotation=30, fontsize='small')

plt.legend(loc='best')

plt.show
#ANA



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



ANA0 = pd.read_csv('../input/ana-data/ana (1).csv')

ANA0 = ANA0.drop(labels=range(0,5),axis=0)

ANA0 = ANA0.reset_index(drop=True)

interest = pd.read_csv('../input/fdsadczgdsatewedvcxz/interest.csv',names = ['year','Interest Rate'])

interest = interest.drop(labels=range(6,8),axis=0)

print(interest)

#print(ANA0)



ANA0['Short-term Debt Change'] = ANA0['Short-term Debt']/ANA0['Short-term Debt'][0]

ANA0['Long-term Debt Change'] = ANA0['Long-term Debt']/ANA0['Long-term Debt'][0]

ANA0['Operating Expenses Change'] = ANA0['Operating Expenses']/ANA0['Operating Expenses'][0]

ANA0['Current Assets Change'] = ANA0['Current Assets']/ANA0['Current Assets'][0]



ANA = pd.DataFrame() 

ANA['interest'] = interest['Interest Rate']

ANA['Short-term Debt Change'] = ANA0['Short-term Debt Change']

ANA['Long-term Debt Change'] = ANA0['Long-term Debt Change']

ANA['Operating Expenses Change'] = ANA0['Operating Expenses Change']

ANA['Current Assets Change'] = ANA0['Current Assets Change']

ANA = ANA.astype(float)

#print(ANA)



#ANA[['interest','Short-term Debt Change','Long-term Debt Change','Operating Expenses Change','Current Assets Change']].plot(figsize=(12,8)) 

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(1, 1, 1)

ax.plot(ANA['interest'],label='interest')

ax.plot(ANA['Short-term Debt Change'],label='Short-term Debt Change')

ax.plot(ANA['Long-term Debt Change'],label='Long-term Debt Change')

ax.plot(ANA['Operating Expenses Change'],label='Operating Expenses Change')

ax.plot(ANA['Current Assets Change'],label='Current Assets Change')

ax.set_xticks([0, 1, 2, 3, 4, 5, 6])

ax.set_xticklabels(['2013','2014','2015','2016(Negative Interest)','2017','2018'], rotation=30, fontsize='small')

plt.legend(loc='best')

plt.show
#Honda



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



Honda = pd.read_csv('../input/honda-data/hondadataset(5).csv', encoding= 'unicode_escape',header=None)

Honda = Honda.drop(labels=range(0,1),axis=0)

Honda = Honda.reset_index(drop=True)

Honda = Honda.astype(float)

interest= pd.read_csv('../input/fdsadczgdsatewedvcxz/interest.csv',names = ['year','Interest Rate'])



Honda['interest'] = interest['Interest Rate']

Honda['Short-term Debt Change'] = Honda[25]/Honda[25][0]

Honda['Net Assets Change'] = Honda[24]/Honda[24][0]

Honda['Accrued expenses Change'] = Honda[21]/Honda[21][0]

#print(Honda)



#Honda[['interest','Short-term Debt Change','Net Assets Change','Accrued expenses Change']].plot(figsize=(12,8)) 

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(1, 1, 1)

ax.plot(Honda['interest'],label='interest')

ax.plot(Honda['Short-term Debt Change'],label='Short-term Debt Change')

ax.plot(Honda['Net Assets Change'],label='Net Assets Change')

ax.plot(Honda['Accrued expenses Change'],label='Accrued expenses Change')

ax.set_xticks([0, 1, 2, 3, 4, 5, 6])

ax.set_xticklabels(['2013','2014','2015','2016(Negative Interest)','2017','2018'], rotation=30, fontsize='small')

plt.legend(loc='best')

plt.show
#Meiji



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



interest= pd.read_csv('../input/fdsadczgdsatewedvcxz/interest.csv',names = ['year','Interest Rate'])



Meiji = pd.read_csv('../input/meiji-1/Meiji 1.csv', encoding= 'unicode_escape',header=None)



Meiji = Meiji.drop(index=[0,1,2,9],axis=0)

Meiji = Meiji.drop([68],axis=1)

Meiji = Meiji.reset_index(drop=True)

Meiji = Meiji.astype(float)

print(Meiji)



Meiji['interest'] = interest['Interest Rate']

Meiji['Total Assets Change'] = Meiji[8]/Meiji[8][0]

Meiji['Current Liabilities Change'] = Meiji[11]/Meiji[11][0]



#Meiji[['interest','Total Assets Change','Current Liabilities Change']].plot(figsize=(12,8)) 

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(1, 1, 1)

ax.plot(Meiji['interest'],label='interest')

ax.plot(Meiji['Total Assets Change'],label='Total Assets Change')

ax.plot(Meiji['Current Liabilities Change'],label='Current Liabilities Change')

ax.set_xticks([0, 1, 2, 3, 4, 5, 6])

ax.set_xticklabels(['2013','2014','2015','2016(Negative Interest)','2017','2018','2019'], rotation=30, fontsize='small')

plt.legend(loc='best')

plt.show