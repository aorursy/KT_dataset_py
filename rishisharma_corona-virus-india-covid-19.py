import os 

import pandas as pd 

import numpy as np

import glob    

import matplotlib.pyplot as plt 

import seaborn as sns

import warnings 

warnings.filterwarnings('ignore')

os.chdir('/kaggle/input/coronavirus-in-italy/dati-province/')
covid_italy = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', "dpc-covid19-ita-province*.csv"))))

covid_italy = covid_italy.drop_duplicates()

covid_italy.head()
covid_india = pd.read_csv('/kaggle/input/covid19-corona-virus-india-dataset/complete.csv')

covid_india.head()
covid_italy['date'] = covid_italy['data'].str[:10]

covid_daily_cases_italy = pd.DataFrame(covid_italy.groupby('date',as_index=True).sum()['totale_casi'])

covid_daily_cases_italy.reset_index(inplace=True)

covid_daily_cases_italy.columns= ['date','total_cases']

covid_daily_cases_italy.head()
covid_india['total_cases'] = covid_india['Total Confirmed cases (Indian National)'] + covid_india['Total Confirmed cases ( Foreign National )']

covid_daily_cases_india = pd.DataFrame(covid_india.groupby('Date',as_index=True).sum()['total_cases'])

covid_daily_cases_india.reset_index(inplace=True)

covid_daily_cases_india.columns= ['date','total_cases']

covid_daily_cases_india.head()
covid_daily_cases_combi=pd.merge(covid_daily_cases_italy,covid_daily_cases_india, how = 'inner',on='date')

covid_daily_cases_combi.columns = ['date','total_cases_italy','total_cases_india']

covid_daily_cases_combi.head()
#fig, ax1 = plt.subplots(figsize=(20, 15))

df = pd.melt(covid_daily_cases_combi, id_vars="date", var_name="country", value_name="total_cases")

# covid_daily_cases_combi.plot(x="date", y="total_cases_italy")

# ax2 = plt.twinx()

# covid_daily_cases_combi.plot(x="date", y="total_cases_india",ax=ax2)



import matplotlib.style as style 



style.use('seaborn-poster') 

style.use('ggplot')



fig, ax1 = plt.subplots(figsize=(17, 8))

g=sns.barplot(x="date", y="total_cases",data = covid_daily_cases_india)

plt.xticks(rotation=75)

plt.ylabel('Cumulatve Corona Cases In India')

plt.xlabel('Date')

plt.title('Cumulative Corona Cases in Inida',fontsize = 25, weight = 'bold')

plt.show()



import matplotlib.style as style 



style.use('seaborn-poster') 

style.use('ggplot')



fig, ax1 = plt.subplots(figsize=(17, 8))

g=sns.barplot(x="date", y="total_cases",data = covid_daily_cases_italy)

plt.xticks(rotation=75)

plt.ylabel('Cumulatve Corona Cases In Italy')

plt.xlabel('Date')

plt.title('Cumulative Corona Cases in Italy',fontsize = 25, weight = 'bold')

plt.show()



## Correlation of Corona Cases between India and Italy 



covid_india.columns = ['date','state','confirmed_indian','confirmed_fgn','cured','lat','lng','death','total_cases']
covid_state_india=pd.DataFrame(covid_india.groupby(['state']).sum()[['total_cases','death']])

covid_state_india.reset_index(inplace=True)

covid_state_india_current=covid_india[covid_india['date']=='2020-03-20']
covid_state_india_current['death_rate'] = covid_state_india_current['death']*100/ covid_state_india_current['total_cases']

covid_state_india_current['death_rate'] = covid_state_india_current['death_rate'].round(2)





#df.value1 = df.value1.round()
style.use('seaborn-poster') 

style.use('ggplot')



fig, ax1 = plt.subplots(figsize=(17, 8))

g=sns.barplot(x="state", y="total_cases",data = covid_state_india_current.sort_values('total_cases',ascending=False))

plt.xticks(rotation=75)

plt.ylabel('Total Corona Cases In States of India')

plt.xlabel('State')

plt.title('Total Corona Cases In States of India',fontsize = 25, weight = 'bold')

plt.show()

#Create combo chart

fig, ax1 = plt.subplots(figsize=(18,8))

color = 'tab:green'

#bar plot creation

ax1.set_title('COVID Death Rate(Across Indian States)', fontsize = 25, weight = 'bold')

ax1 = sns.barplot(x='state', y='total_cases', data = covid_state_india_current, palette='summer')

ax1.tick_params(axis='y')

ax1.set_xlabel('State', fontsize=16)

ax1.set_ylabel('Number of Cases of Corona in India', fontsize=16)



plt.xticks(rotation=60)

#specify we want to share the same x-axis

ax2 = ax1.twinx()

color = 'tab:red'

#line plot creation

ax2 = sns.lineplot(x='state', y='death_rate', data = covid_state_india_current, palette='summer')

ax2.tick_params(axis='y', color=color)

ax2.set_ylabel('Death Rate', fontsize=16)

ax1.grid(False)



#show plot

plt.show()
