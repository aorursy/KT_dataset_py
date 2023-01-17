# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')

df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)



df['Datum'] = pd.to_datetime(df['Datum'])

df['Date'] = df['Datum'].apply(lambda row: row.date())



df['Country'] = df.Location.apply(lambda row: row.split(', ')[-1])



df.head()
total_companies = len(df['Company Name'].unique())

total_rows = len(df)

print("Total number of Companies: {}".format(total_companies))

print("Total Data Rows: {}".format(total_rows))
count = df.groupby('Company Name')

count = count['Company Name'].count()

count = count.sort_values(ascending=False)



top = 15



top15 = list(count.index[:top])



plt.figure(figsize=(12, 6))

plt.title('Top 15 Companies with maximum Launches',fontsize=20)

plt.xlabel("Company Name",fontsize=12)

plt.ylabel("#Launches",fontsize=12)



ax = sns.barplot(x = count.index[:top], y = count.values[:top])

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



plt.show()
rvsn = df[df['Company Name']=='RVSN USSR'].sort_values(by="Date")



plt.figure(figsize=(12, 6))



rvsn['Date'] = rvsn["Date"].astype("datetime64")

rr = rvsn.groupby(rvsn['Date'].dt.year)["Date"].count().plot(kind='bar')



plt.title('Launch Distribustion of RBSN USSR each Year', fontsize=16)

plt.ylabel('#Launches',fontsize=14)

plt.xlabel('Year',fontsize=14)



plt.show()
age = df.groupby('Company Name').agg({"Date":["min",'max']})

age.columns = ['First Launch','Last Launch']



age['Years of service'] =age['Last Launch'] - age['First Launch']

age['Years of service'] = age['Years of service'].apply(lambda row: float("{:.1f}".format(row.days/365)))

age = age.sort_values(by='Years of service', ascending=False)



age = age.reset_index()



age
def highlights(s):

    

    if str(s['First Mission']) == "Failure":

        return ['background-color: red']*2

    elif str(s['First Mission']) == "Partial Failure":

        return ['background-color: Salmon']*2

    else:

        return ['background-color: green']*2

    



first_mission = df.groupby('Company Name')['Status Mission','Date'].apply(min)

first_mission = first_mission.reset_index()[['Company Name','Status Mission']]

first_mission.columns = ['Company Name','First Mission']

first_mission_styled = first_mission.style.apply(highlights,axis=1)



first_mission_styled
active = df[df['Status Rocket']=='StatusActive']

active = active.groupby('Company Name').apply(lambda dd: dd['Company Name'].count())

active = active.sort_values(ascending=False)



top = 15



plt.figure(figsize=(12, 6))

plt.title('Top 15 Companies with Active Rockets',fontsize=20)

plt.xlabel("Company Name",fontsize=12)

plt.ylabel("# Active Rockets",fontsize=12)



ax = sns.barplot(x=active.index[:top], y=active.values[:top])

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)



plt.show()
## Ranking Companies on success rate



## Formula: (successfull Launches/ total Launches)



success_rate = df.groupby('Company Name')

success_rate = success_rate.apply(lambda dd: 

            len(dd[dd['Status Mission']=='Success'])/len(dd)) 



success_rate_top = success_rate[top15]

success_rate_top = success_rate_top.sort_values(ascending=False)



top = 56



plt.figure(figsize=(16, 6))

ax = sns.barplot(x=success_rate_top.index[:top], y=success_rate_top.values[:top])

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



plt.title('Success rate of 15 companies with maximum Launches',fontsize=20)

plt.ylabel("",fontsize=12)

plt.xlabel("",fontsize=12)



plt.show()

x = df.groupby('Status Mission')

x = x['Status Mission'].count()

x = x.sort_values(ascending=False)



## Pie Chart

plt.figure(figsize=(9, 9))

explode = (0, 0.1, 0, 0)



plt.pie(x.values, labels=x.index, autopct='%1.1f%%',

       explode=explode)

plt.show()
x = pd.DataFrame()



## concatinating

x[['First Launch','Last Launch','Year of Service']] = age.set_index('Company Name')[['First Launch','Last Launch','Years of service']]

x['Launch Count'] = count

x['Success Rate'] = success_rate

x['Total Active'] = active

x = x.fillna(0)

x['Total Active'] = x['Total Active'].astype('int')

x['First Mission status'] = first_mission.set_index('Company Name')['First Mission']

x

