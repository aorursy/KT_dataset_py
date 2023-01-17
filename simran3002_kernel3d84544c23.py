# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Ignoring all warnings

import warnings

warnings.filterwarnings("ignore")

# Setting color for dark mode

COLOR = 'white'



# Setting color for light mode

#COLOR = 'black'



plt.rcParams['text.color'] = 'black'

plt.rcParams['axes.labelcolor'] = COLOR

plt.rcParams['xtick.color'] = COLOR

plt.rcParams['ytick.color'] = COLOR
co_data=pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

co_data.head()
co_data.tail()
co_data.shape
co_data.info()
co_data.describe()
del_col_list=['Time']

co_data=co_data.drop(del_col_list,axis=1)

co_data.head()
total = co_data.groupby('State/UnionTerritory')['Deaths'].sum()
total = total.sort_values()

plt.figure(figsize=(20,10))

plt.xlabel('State/UnionTerritory', fontsize=20)

plt.xticks(rotation=90, fontsize=15)

plt.yticks(fontsize = 15)

plt.ylabel('No. of Deaths',fontsize=20)

#plt.yscale('log')

#plt.plot(total)

plt.scatter(total.index,total.values)

ax = plt.axes()

# Setting the background color

ax.set_facecolor("black")

plt.title('Total Deaths in each state', color=COLOR,fontsize = 25)

plt.show()

# What is the per day increase in Confirmed Cases and deaths ?

cnf_data = co_data[['Date','Confirmed']]

cnf_deaths = co_data[['Date','Deaths']]



# What is the per month increase in Confirmed cases? (To know the average number of hospital beds )

## We can use the cnf_data dataframe for this



# What is the percentage of people affected in every age group ?(to Know which age group is most likely to get the virus)

age_grp_path = "../input/covid19-in-india/AgeGroupDetails.csv"

age_grp_data = pd.read_csv(age_grp_path)

age_grp_percentage = age_grp_data[['AgeGroup','Percentage']]



# What is the percentage of people affected in each state (Population VS Confirmed Cases)?

population_path = "../input/covid19-in-india/population_india_census2011.csv"

pop_data = pd.read_csv(population_path)

state_pop_data = pop_data[['State / Union Territory','Population']]

state_cnf_cases = co_data[['State/UnionTerritory','Confirmed']]



# States with most and least testing happening?

state_testing_path = "../input/covid19-in-india/StatewiseTestingDetails.csv"

state_testing_data_raw = pd.read_csv(state_testing_path)

state_testing_data = state_testing_data_raw[['State','TotalSamples']]



# What is the No. of cured people every day in each State?

state_cured_cases = co_data[['State/UnionTerritory','Date','Cured']]



# What is the Number of active cases in each state ?

state_active_cases = co_data[['State/UnionTerritory','Date','Cured','Deaths','Confirmed']]
grpd_data_cnf = cnf_data.groupby('Date')['Confirmed'].sum().reset_index()

grpd_data_cnf['Date'] = pd.to_datetime(grpd_data_cnf.Date, dayfirst=True)

grpd_data_cnf = grpd_data_cnf.set_index('Date')

grpd_data_cnf = grpd_data_cnf.sort_index()



grpd_data_deaths = cnf_deaths.groupby('Date')['Deaths'].sum().reset_index()

grpd_data_deaths['Date'] = pd.to_datetime(grpd_data_deaths.Date, dayfirst=True)

grpd_data_deaths = grpd_data_deaths.set_index('Date')

grpd_data_deaths = grpd_data_deaths.sort_index()
day_increase_cases = grpd_data_cnf['Confirmed'].diff()

day_increase_deaths = grpd_data_deaths['Deaths'].diff()



plt.figure(figsize=(15,10))

ax = plt.axes()

# Setting the background color

ax.set_facecolor("black")



plt.plot(day_increase_cases)

plt.plot(day_increase_deaths)



plt.xlabel('Date', fontsize=20)

#plt.xticks(rotation=90, fontsize=15)

#plt.yticks(fontsize = 15)

plt.ylabel('Amount of increase',fontsize=20)

plt.legend(['Daily increase in cases','Daily increase in deaths'], loc='upper left')

plt.title('Increase in Cases and Deaths', color=COLOR,fontsize = 25)

plt.show()
grpd_data_cnf_month = grpd_data_cnf.groupby(pd.Grouper(freq="M")).sum()
plt.figure(figsize=(15,10))

plt.plot(grpd_data_cnf_month)



ax = plt.axes()

# Setting the background color

ax.set_facecolor("black")



plt.xlabel('Month', fontsize=20)

#plt.xticks(rotation=90, fontsize=15)

#plt.yticks(fontsize = 15)

plt.ylabel('Cases',fontsize=20)

#plt.yscale('log')

plt.legend(['Confirmed Cases'], loc='upper left')

plt.title('Total cases per month', color=COLOR,fontsize = 25)

plt.show()
age_grp_percentage['Percentage'] = age_grp_percentage['Percentage'].str.rstrip('%')

age_grp_percentage['Percentage'] = age_grp_percentage['Percentage'].astype(float)
plt.figure(figsize=(15,10))

plt.bar(age_grp_percentage['AgeGroup'],age_grp_percentage['Percentage'], color='orange')



plt.xlabel('Age Group', fontsize=20)

#plt.xticks(rotation=90, fontsize=15)

#plt.yticks(fontsize = 15)

plt.ylabel('Percentage',fontsize=20)

#plt.yscale('log')



ax = plt.axes()

# Setting the background color

ax.set_facecolor("black")



plt.title('Distribution of cases (on basis of Age group)', color=COLOR,fontsize = 25)

plt.show()
state_pop_data

state_cnf_cases = state_cnf_cases.groupby('State/UnionTerritory')['Confirmed'].sum().reset_index()

state_pop_data.columns = ['State/UnionTerritory','Population']
common_data = state_pop_data.merge(state_cnf_cases,  how='inner')

common_data['Percentage'] = common_data['Confirmed']/common_data['Population']*100
plt.figure(figsize=(20,10))

plt.bar(common_data['State/UnionTerritory'],common_data['Percentage'], color='orange')

ax = plt.axes()

# Setting the background color

ax.set_facecolor("black")

plt.xlabel('State/UnionTerritory', fontsize=20)

plt.xticks(rotation=90, fontsize=15)

plt.yticks(fontsize = 15)

plt.ylabel('Percentage',fontsize=20)

plt.title('Percentage of people affected in each state', color=COLOR,fontsize = 25)

plt.show()

state_testing_data = state_testing_data.groupby('State')['TotalSamples'].sum().reset_index()
plt.figure(figsize=(20,10))

plt.bar(state_testing_data['State'],state_testing_data['TotalSamples'], color='orange')

plt.xlabel('State/UnionTerritory', fontsize=20)

plt.xticks(rotation=90, fontsize=15)

plt.yticks(fontsize = 15)

plt.ylabel('No. of Tests',fontsize=20)



ax = plt.axes()

# Setting the background color

ax.set_facecolor("black")



plt.title('Tests performed in each state', color=COLOR,fontsize = 25)
state_cured_cases = state_cured_cases.groupby(['State/UnionTerritory'])
plt.figure(figsize=(20,15),dpi=400)

plt.xticks(rotation=270, fontsize=5)

states = []

for i in state_cured_cases:

    i[1]['Date'] = pd.to_datetime(i[1].Date, dayfirst=True)

    #i[1] = i[1].set_index('Date')

    #i[1] = i[1].sort_index()

    plt.plot(i[1]['Date'], i[1]['Cured'])

    states.append(i[0])

plt.legend(states, loc='upper left')

ax = plt.axes()

# Setting the background color

ax.set_facecolor("black")

plt.title('No. of people cured every day in each State?', color=COLOR,fontsize = 25)

plt.xlabel('Date', fontsize=20)

plt.yticks(fontsize = 15)

plt.ylabel('No. of People',fontsize=20)

plt.show()
state_active_cases

state_active_cases['Date'] = pd.to_datetime(state_active_cases.Date, dayfirst=True)

state_cases = state_active_cases.set_index('Date')

state_cases = state_cases.sort_index()
last_updated_cases = state_cases.loc['2020-07-17']

last_updated_cases['Active'] = last_updated_cases['Confirmed'] - last_updated_cases['Cured'] - last_updated_cases['Deaths']

last_updated_cases = last_updated_cases[['State/UnionTerritory','Active']]

last_updated_cases.set_index('State/UnionTerritory')
plt.figure(figsize=(20,10))

plt.bar(last_updated_cases['State/UnionTerritory'],last_updated_cases['Active'], color='orange')

plt.xlabel('State/UnionTerritory', fontsize=20)

plt.xticks(rotation=90, fontsize=15)

plt.yticks(fontsize = 15)

plt.ylabel('Active Cases',fontsize=20)

ax = plt.axes()

# Setting the background color

ax.set_facecolor("black")

plt.title('No. of active cases in each State', color=COLOR,fontsize = 25 )