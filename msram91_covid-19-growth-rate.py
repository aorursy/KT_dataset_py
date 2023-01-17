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
df_india = pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv")

df_india.head()

# Converting Date columns into Date type

df_india['date_C'] =pd.to_datetime(df_india['Date'],format='%d/%m/%y')

# Converting Incorrect state name into right ones

df_india['State/UnionTerritory']=df_india['State/UnionTerritory'].str.replace('Nagaland#','Nagaland')

df_india['State/UnionTerritory']=df_india['State/UnionTerritory'].str.replace('Jharkhand#','Jharkhand')
# getting max(date) and state

df_date=df_india.groupby('State/UnionTerritory')['date_C'].max().to_frame().reset_index()

# Merge to get only latest data of each state

df_merge_latest = df_india.merge(df_date,how='inner', on =['State/UnionTerritory','date_C'] )

df_merge_latest = df_merge_latest.sort_values(by=['Confirmed'],ascending=True)

df_merge_latest.sort_values(by='Confirmed',ascending=False).reset_index()

df_merge_latest=df_merge_latest.drop(['Time','Sno','Date','ConfirmedIndianNational','ConfirmedForeignNational'],axis=1)



# df_merge_latest.sort_values(by='Confirmed',ascending=False).reset_index()

df_merge_latest['ActiveCases'] = df_merge_latest['Confirmed'] - (df_merge_latest['Cured']+df_merge_latest['Deaths'])

df_merge_latest=df_merge_latest.sort_values(by='Confirmed',ascending=False).reset_index()
df_merge_latest["RecoveryRate"] = round(df_merge_latest['Cured']/df_merge_latest['Confirmed']*100,2)

df_merge_latest["DeathRate"] = round(df_merge_latest['Deaths']/df_merge_latest['Confirmed']*100,2)

df_merge_latest['InfectedRate'] = round((df_merge_latest['ActiveCases']/df_merge_latest['Confirmed'])*100,2)

df_merge_latest=df_merge_latest.sort_values(by='Confirmed',ascending=False).reset_index()

df_merge_latest=df_merge_latest.loc[df_merge_latest['Confirmed']>10,:]  #Considering states contains more than 10 confirmed cases

df_merge_latest=df_merge_latest.loc[df_merge_latest['State/UnionTerritory'] != 'Unassigned',:]

df_merge_latest

import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize=(60,20))

sns.barplot(x='State/UnionTerritory',y='RecoveryRate',data=df_merge_latest.sort_values(by='RecoveryRate',ascending=False))

plt.ylabel("Rate")

plt.xlabel("States")

plt.title("Analysis- Covid 19 - India")

plt.show()
plt.figure(figsize=(60,20))

sns.barplot(x='State/UnionTerritory',y='DeathRate',data=df_merge_latest.sort_values(by='DeathRate',ascending=False))

plt.ylabel("DeathRate")

plt.xlabel("States")

plt.title("Analysis- Covid 19 - India")

plt.show()
plt.figure(figsize=(60,20))

sns.barplot(x='State/UnionTerritory',y='ActiveCases',data=df_merge_latest.sort_values(by='ActiveCases',ascending=False))

plt.ylabel("UnderTreatment")

plt.xlabel("States")

plt.title("Analysis- Covid 19 - India")

plt.show()
df_grp_date = df_india.groupby('date_C')['Confirmed'].sum().to_frame().reset_index().sort_values(by='date_C')

df_grp_date.tail(10)
# df_grp_state_by_date = 

df_grp_state_by_date= df_india.groupby(['date_C','State/UnionTerritory'])['Confirmed','Cured','Deaths'].sum().reset_index()

df_grp_state_by_date.head()
# getting recovery rate and death rate

df_grp_state_by_date["RecoveryRate"] = round(df_grp_state_by_date['Cured']/df_grp_state_by_date['Confirmed']*100,2)

df_grp_state_by_date["DeathRate"] = round(df_grp_state_by_date['Deaths']/df_grp_state_by_date['Confirmed']*100,2)

df_grp_state_by_date["ActiveCases"] = df_grp_state_by_date['Confirmed'] - (df_grp_state_by_date['Cured']+df_grp_state_by_date['Deaths'])

df_grp_state_by_date['Infected_Rate'] = round((df_grp_state_by_date['ActiveCases']/df_grp_state_by_date['Confirmed'])*100,2)

df_grp_state_by_date['month_2020'] = pd.DatetimeIndex(df_india['date_C']).month

df_grp_state_by_date[df_grp_state_by_date['State/UnionTerritory']=='Tamil Nadu'].sort_values(by='date_C',ascending=False)

df_grp_state_by_date.head()



df_grp_by_date= df_india.groupby(['date_C'])['Confirmed','Cured','Deaths'].sum().reset_index()

df_grp_by_date["RecoveryRate"] = round(df_grp_by_date['Cured']/df_grp_by_date['Confirmed']*100,2)

df_grp_by_date["DeathRate"] = round(df_grp_by_date['Deaths']/df_grp_by_date['Confirmed']*100,2)

df_grp_by_date["ActiveCases"] = df_grp_by_date['Confirmed'] - (df_grp_by_date['Cured']+df_grp_by_date['Deaths'])

df_grp_by_date['UnderTreatment_Rate'] = round((df_grp_by_date['ActiveCases']/df_grp_by_date['Confirmed'])*100,2)

df_grp_by_date.head()

plt.figure(figsize=(50,5))

sns.lineplot(x='date_C',y='RecoveryRate',data=df_grp_by_date)

sns.lineplot(x='date_C',y='DeathRate',data=df_grp_by_date)

# sns.lineplot(x='date_C',y='UnderTreatment_Rate',data=df_grp_by_date)

plt.ylabel("Rate")

plt.xlabel("Date")

plt.title("Analysis- Covid 19 - India")

plt.show()
import seaborn as sns

import matplotlib.pyplot as plt
df_grp_TTN = df_grp_state_by_date[df_grp_state_by_date['State/UnionTerritory']== 'Tamil Nadu']

plt.figure(figsize=(30,5))

sns.lineplot(x='date_C',y='RecoveryRate',data=df_grp_TTN)

sns.lineplot(x='date_C',y='DeathRate',data=df_grp_TTN)

sns.lineplot(x='date_C',y='Infected_Rate',data=df_grp_TTN)

plt.ylabel("Rate")

plt.xlabel("Date")

plt.title("Analysis- Covid 19 - Tamil Nadu")

plt.show()
# df_i=pd.read_csv("/kaggle/input/covid19-in-india/IndividualDetails.csv")

# df_i[df_i['detected_state']=='Tamil Nadu']

# # df_i.head()

df_testing =pd.read_csv("/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv")

df_testing_TN=df_testing[df_testing['State']=='Tamil Nadu'].reset_index()

df_testing_TN['Infected_Rate']= round(df_testing_TN['Positive']/df_testing_TN['TotalSamples'],2)*100

# df_testing_TN['Average_Rate_India']=round(df_testing['Positive'].mean(),2)

df_testing_TN



c=pd.read_csv("/kaggle/input/covid19-in-india/IndividualDetails.csv")

# statewise_df=

statewise_df=c.groupby('detected_state')['id'].count().to_frame().reset_index()

statewise_df['TotalCases']=statewise_df['id']

statewise_df=statewise_df.drop('id',axis=1)

statewise_df




