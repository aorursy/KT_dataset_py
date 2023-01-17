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

plt.style.use('fivethirtyeight')


# Any results you write to the current directory are saved as output.
#!pip install urllib2
df = pd.read_csv('../input/corona-cases-in-pakistan/Corona Pakistan.csv',index_col='SNo')
df.shape
df.info()
df["Reporting Date"] =  pd.to_datetime(df['Reporting Date'])
df.sort_values("Reporting Date", axis = 0, ascending = True, 
                 inplace = True, na_position ='last') 
df.head()
df.describe()
df
fig, ax = plt.subplots(figsize=(12,7))
df[['Reporting Date','Cumulative Travellers Screened']].plot(x='Reporting Date',kind='line',ax=ax, title="Cumulative Travellers Screened")
fig, ax = plt.subplots(figsize=(12,7))
df[['Reporting Date','Returnees from China']].plot(x='Reporting Date',kind='line',ax=ax, title="Cumulative Chinese Travellers Screened")
fig, ax = plt.subplots(figsize=(12,7))
df[['Reporting Date','Returnees from Iran']].plot(x='Reporting Date',kind='line',ax=ax, title="Cumulative Iranian Travellers Screened")
import matplotlib.pyplot as plt

ax.set_xlabel("Reporting Date")
ax.set_ylabel("Count of Confirmed Positive Cases, Tests Performed, and Admitted")
fig, ax = plt.subplots(figsize=(12,7))
df[['Reporting Date','Cumulative Tests Performed','Cumulative Test Positive Cases','Still Admitted']].plot(x='Reporting Date',kind='line', ax=ax, title="Cumulative Tests Performed Versus Positive Cases and Still Admitted")
fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlabel("Reporting Date")
ax.set_ylabel("Pakistan | Comparing Count of Suspected, Positive, Admitted and Expired Cases")
df[['Reporting Date','Cumulative - Suspected Cases','Cumulative Test Positive Cases','Still Admitted','Expired']].plot(x='Reporting Date',kind='line',ax=ax, title="Pakistan | Positive Cases versus Still Admitted and Expired")
df['Positive Rate'] = (df['Cumulative Test Positive Cases']/df['Cumulative - Suspected Cases'])*100
df['Death Rate in Pakistan'] = (df['Expired']/df['Cumulative Test Positive Cases'])*100
#df[['Positive Rate','Admission Rate','Death Rate']]
fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlabel("Reporting Date")
ax.set_ylabel("Positve Rate versus Death Rate")
df[['Reporting Date','Positive Rate','Death Rate in Pakistan']].plot(x='Reporting Date',kind='line', ax=ax, title="Pakistan's Corona Effort(%)")

df['Death Rate in Italy']= (df['Total Deaths in Italy']/df['Total Cases in Italy'])*100


df['Death Rate in World']= (df['Global Deaths']/df['Global Cases'])*100
fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlabel("Reporting Date")
ax.set_ylabel("Date Rate(%) in Pakistan versus World")
df[['Reporting Date','Death Rate in World','Death Rate in Pakistan','Death Rate in Italy']].plot(x='Reporting Date',kind='line',ax=ax, title="Corona Related Death Rate(%) in Pakistan versus World and Italy")
#df[['Reporting Date','Death Rate in Italy','Death Rate in Pakistan']].plot(x='Reporting Date',kind='line', ax=ax, title="Death Rate(%) in Pakistan versus Italy")
fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlabel("Reporting Date")
ax.set_ylabel("Count of Positive Cases")
df[['Reporting Date','Cumulative Test Positive Cases','Global Cases']].plot(x='Reporting Date',kind='line',ax=ax, title="Global Positive Cases versus Pakistan's Positive Cases")
fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlabel("Reporting Date")
ax.set_ylabel("Log of Positive Cases in Pakistan versus World")

df['log of Cumulative Pakistani Positive Cases']= np.log(df['Cumulative Test Positive Cases'])
df['log of Cumulative Global Positive Cases']= np.log(df['Global Cases'])
df[['Reporting Date','log of Cumulative Pakistani Positive Cases','log of Cumulative Global Positive Cases']].plot(x='Reporting Date',kind='line',ax=ax, title="Comparison on Lograthmic Scale | Global Positive Cases versus Pakistan's Positive Cases")
fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlabel("Reporting Date")
ax.set_ylabel("Log of Cumulative deaths in Pakistan versus World")

df['log of Cumulative Pakistani Deaths']= np.log(df['Expired'])
df['log of Cumulative Global Deaths']= np.log(df['Global Deaths'])
df[['Reporting Date','log of Cumulative Pakistani Deaths','log of Cumulative Global Deaths']].plot(x='Reporting Date',kind='line', ax=ax ,title="Comparison on Lograthmic Scale | Fatalities in Pakistan versus Rest of the World")
fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlabel("Reporting Date")
ax.set_ylabel("Number of Calls")

df[['Reporting Date','New Calls','Cumulative Calls']].plot(x='Reporting Date',kind='line',ax=ax, title="Tracking Potential Sources of Infection")

df['log of Travellers Count(Iran + China)']= np.log(df['Returnees from China'] + df['Returnees from Iran'] )

df['log of Cumulative Pakistani Positive Cases']= np.log(df['Cumulative Test Positive Cases'])

df['log of Cumulative Global Positive Cases']= np.log(df['Global Cases'])

fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlabel("Reporting Date")
ax.set_ylabel("Log of Cumulative Positive Cases in Pakistan versus World")

df[['Reporting Date','log of Cumulative Pakistani Positive Cases','log of Cumulative Global Positive Cases','log of Travellers Count(Iran + China)']].plot(x='Reporting Date',kind='line',ax=ax, title="Comparison on Lograthmic Scale | Global Positive Cases versus Pakistan's Positive Cases and Impact of Travelling Restrictions")
fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlabel("Reporting Date")
ax.set_ylabel("Log of Counts")

df['log of Cumulative Positive Cases in Pakistan']= np.log(df['Cumulative Test Positive Cases'])
df['log of Cumulative Global Positive Cases']= np.log(df['Global Cases'])
df['log of Cumulative Deaths in Pakistan']= np.log(df['Expired'])
df['log of Cumulative Global Deaths']= np.log(df['Global Deaths'])
df['log of Cumulative Positive Cases in Italy ']= np.log(df['Total Cases in Italy'])
df['log of Cumulative Deaths in Italy']= np.log(df['Total Deaths in Italy'])
df[['Reporting Date','log of Cumulative Positive Cases in Italy ','log of Cumulative Global Positive Cases','log of Cumulative Positive Cases in Pakistan','log of Cumulative Deaths in Pakistan','log of Cumulative Global Deaths','log of Cumulative Deaths in Italy']].plot(x='Reporting Date',kind='line',ax=ax, title="Comparison on Lograthmic Scale | Fatalities in Pakistan versus Rest of the World and Italy")
## As Per Media Reports quoting Zafar Mirza

import matplotlib.pyplot as plt
classes = ['Misc - Mostly Secondary Cases','Umrah','Tableghis', 'Iranian Origin(Zaireen) Positive Cases','Positive Cases from Other Countries']
pop = [220,100,100,857,191]

plt.pie(pop,labels=classes,autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Breakdown of Pakistans Corona Virus Cases')

#Show
plt.show()