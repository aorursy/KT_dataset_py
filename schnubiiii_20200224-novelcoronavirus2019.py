# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#df1 = pd.read_csv('../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')  #older Data......
df2 = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df_conf = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
df_deaths = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
df_recov = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
df2.head()
df_conf.head()
df_deaths.head()
df_recov.head()
df2.describe()
df_conf.describe()
#df_deaths.describe()
#df_recov.describe()
df2.info()
#df_conf.info()
#df_deaths.info()
#df_recov.info()
df_conf2 = df_conf.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1)
df_deaths2 = df_deaths.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1)
df_recov2 = df_recov.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1)
patnum = [] 
for columns in df_conf2.columns:           #calculation of the total cases
    patnum.append(sum(df_conf2[columns])) 

plt.figure(figsize=(25,10))
ax = sns.barplot(x=df_conf2.columns, y = patnum)
ax.set(xlabel='date', ylabel='number of infected people')
ax.set_title('Number of infections from 22nd of January 2020 on worldwide')
plt.xticks(rotation=80)
plt.show()
#-------------------------------------------------------------------------------------------------------------------
deathnum = [] 
for columns in df_deaths2.columns: 
    deathnum.append(sum(df_deaths2[columns]))

plt.figure(figsize=(25,10))
ax = sns.barplot(x=df_deaths2.columns, y = deathnum)
ax.set(xlabel='date', ylabel='number of patients died')
ax.set_title('Number of deaths from 22nd of January 2020 on worldwide')
plt.xticks(rotation=80)
plt.show()
#---------------------------------------------------------------------------------------------------------------------
recovnum = [] 
for columns in df_recov2.columns: 
    recovnum.append(sum(df_recov2[columns]))

plt.figure(figsize=(25,10))
ax = sns.barplot(x=df_recov2.columns, y = recovnum)
ax.set(xlabel='date', ylabel='number of patients recovered')
ax.set_title('Number of recoveries during 22nd of January 2020 on worldwide from Cov19-2 infection')
plt.xticks(rotation=80)
plt.show()
df3 = df2.groupby(['Country/Region']).sum()
df3 = df3.iloc[:, 1:4]
df3 = df3.reset_index()
#---------------------------------------------------
fig, axs = plt.subplots(ncols=3, sharey = True, figsize=(30,40))
ax1 = sns.barplot(x = 'Confirmed', y ='Country/Region', data=df3 , ci = None, ax = axs[0])
ax1.set(xlabel='Infections', ylabel='Country/Region')
ax1.set_xscale("log")
ax1.set_title('Total Infections per Country')
#-------------------------------------------------
ax2 = sns.barplot(x = 'Deaths', y ='Country/Region', data=df3 , ci = None, ax = axs[1])
ax2.set(xlabel='Deaths', ylabel='Country/Region')
ax2.set_title('Total Deaths per Country')
ax2.set_xscale("log")
#-------------------------------------------------------
ax3 = sns.barplot(x = 'Recovered', y ='Country/Region', data=df3 , ci = None, ax = axs[2])
ax3.set(xlabel='Recoveries', ylabel='Country/Region')
ax3.set_title('Total Recoveries per Country')
ax3.set_xscale("log")
plt.show()

x = df2.groupby("Country/Region").sum().reset_index()
print("The SARS-CoV-2 Virus hast infected " + str(x["Country/Region"].count())+ " countries up to now.")
df_china = df2.loc[df2['Country/Region']== 'Mainland China']
df_china2 = df_china.groupby("ObservationDate").sum()
df_china2 = df_china2.reset_index()
fig, axs = plt.subplots(nrows=3,sharex=True ,figsize=(30,15))
ax1 =sns.barplot(x = 'ObservationDate', y='Confirmed',  data = df_china2, ax = axs[0], ci = None)
ax1.set_title('Number of Cov19-2 infections in China')
#------------------------------------------------------------------------------------
ax2 = sns.barplot(x = 'ObservationDate', y='Deaths',  data = df_china2, ax = axs[1],ci = None)
ax2.set_title('Number of Cov19-2 related deaths in China')
#----------------------------------------------------------------------------------------
ax3 = sns.barplot(x = 'ObservationDate', y='Recovered',  data = df_china2, ax = axs[2],ci = None)
ax3.set_title('Number of Cov19-2 recoveries in China')
plt.xticks(rotation=80)

plt.show()
df_china = df2.loc[df2['Country/Region']== 'Mainland China'] 

plt.figure(figsize=(30,15))
ax = sns.barplot(x = 'Province/State', y='Confirmed',  data = df_china, hue = 'ObservationDate')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('Number of Cov19-2 infections per Chinese Province and date') 
plt.show()
df_ger = df2.loc[df2['Country/Region']== 'Germany'] 

fig, axs = plt.subplots(nrows=3,sharex=True ,figsize=(30,15))
ax1 =sns.barplot(x = 'ObservationDate', y='Confirmed',  data = df_ger, ax = axs[0])
ax1.set_title('Number of Cov19-2 infections in Germany')
#------------------------------------------------------------------------------------
ax2 = sns.barplot(x = 'ObservationDate', y='Deaths',  data = df_ger, ax = axs[1])
ax2.set_title('Number of Cov19-2 related deaths in Germany')
#----------------------------------------------------------------------------------------
ax3 = sns.barplot(x = 'ObservationDate', y='Recovered',  data = df_ger, ax = axs[2])
ax3.set_title('Number of Cov19-2 recoveries in Germany')
plt.xticks(rotation=80)

plt.show()
df_ger2 = df_ger
df_ger2 = df_ger2.drop(["SNo","Province/State","Last Update","Country/Region"], axis = 1)
#df_ger2.head()
ax1 = sns.pairplot(df_ger2,  kind="reg")
ax1.fig.suptitle('Pairplot for COVID19 Cases in Germany over time', y=1.08)
plt.show()
df_ger2["Ratio"] = (df_ger2["Deaths"]/df_ger2["Confirmed"]*100)
df_ger2.head()
plt.figure(figsize=(15,8))
sns.stripplot(x = 'ObservationDate', y='Ratio',  data = df_ger2)
plt.xticks(rotation=80)
plt.title('Development of the death to confirmed infection ratio over time in percent')
plt.show()
x = df_ger2.iloc[-1,1] 
y = df_ger2.iloc[-1,2]
z = round((y/x)*100, 3)
print("Percentage of fatal cases per confirmed COVID19 infection in Germany:" + " " + str(z) + "%")
