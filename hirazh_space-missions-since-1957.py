# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

%matplotlib inline
# Loading the data
space_missions = pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')
space_missions
space_missions.info()
# The column 'Rocket' has the whitespace in the begining therefore remove all the unnecessary whitespaces
# from the all the columns if available
space_missions.columns = space_missions.columns.str.lstrip()
space_missions['Company Name'].nunique()
space_companies = space_missions.groupby('Company Name')['Company Name'].count().to_frame('Mission Counts').reset_index()
space_companies = space_companies.sort_values(by='Mission Counts', ascending=False)

# filtering the space comapnies which has a launch count more than 10
space_companies = space_companies.loc[space_companies['Mission Counts'] >= 10]

# Ploting the space_companies data
fig, ax = plt.subplots(figsize=(16,8))
sns.set_style('dark')
ax = sns.barplot(x=space_companies['Company Name'], y=space_companies['Mission Counts'], palette=("deep"))
ax.set_xticklabels(space_companies['Company Name'], rotation=90)
ax.set_xlabel('Company Name', fontsize=16)
ax.set_ylabel('Mission Counts', fontsize=16)
ax.set_title('Total Missions (Since 1957)', fontsize=20)
plt.show()
# Create the Mision Year column from Datum column
space_missions['Datum'] = pd.to_datetime(space_missions['Datum'], utc=True)
space_missions['Mission Year'] = space_missions['Datum'].dt.year

old_missions = space_missions.loc[space_missions['Mission Year'] < 2000]
new_missions = space_missions.loc[space_missions['Mission Year'] >= 2000]
# the missions after year 2000 
space_missions_since_2000 = new_missions[['Company Name', 'Datum', 'Mission Year']]

mission_count_since_2000 = space_missions_since_2000.groupby('Company Name')['Company Name'].count().to_frame('Mission Count').reset_index()
mission_count_since_2000 = mission_count_since_2000.sort_values(by='Mission Count', ascending=False)
top_mission_count_since_2000 = mission_count_since_2000.loc[mission_count_since_2000['Mission Count'] >= 10]

# plotting the data for the missions after 2000
fig,ax = plt.subplots(figsize=(16,8))
sns.set_style('dark')
ax = sns.barplot(x=top_mission_count_since_2000['Company Name'], y=top_mission_count_since_2000['Mission Count'], palette=("deep"))
ax.set_xticklabels(top_mission_count_since_2000['Company Name'], rotation=90)
ax.set_xlabel('Company Name', fontsize=16)
ax.set_ylabel('Mission Counts', fontsize=16)
ax.set_title('Total Missions (Since 2000)', fontsize=20)

plt.show()
less_than_10_mission = mission_count_since_2000.loc[mission_count_since_2000['Mission Count'] < 10].reset_index(drop=True)
less_than_10_mission
mission_status = space_missions.groupby('Status Mission')['Status Mission'].count().to_frame('Status Count').reset_index()
mission_status
def mission_success(data_frame):
    result_df = data_frame.groupby('Status Mission')['Status Mission'].count().to_frame('Status Count').reset_index()
    # In the Status Mission column we get the SUM of Failure, Partial Failure and Prelaunch Failure in the "Status Mission"
    # as a single status "Failure"
    failed_count = result_df.iloc[0:3]['Status Count'].sum()
    success_count = result_df.iloc[-1]['Status Count']
    
    data = [['Failure', failed_count], ['Success', success_count]]
    # creating the Data Frame using the Failure and Success counts
    result_df = pd.DataFrame(data, columns = ['Status', 'Count'])
    
    return result_df

# Mission status since 1957
success_rate_all = mission_success(space_missions)
# Mission status before 2000
success_rate_before_2000 = mission_success(old_missions)
# Mission status After 2000
success_rate_after_2000 = mission_success(new_missions)

# ploting the data
fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

explode = (0.0, 0.1)
colors = ['#FC9D9A', '#2196F3']

ax1.pie(success_rate_all['Count'], labels=success_rate_all['Status'], autopct='%1.1f%%', 
        explode=explode, colors=colors)
ax1.set_title('Success rate since 1957', fontsize=16)
ax2.pie(success_rate_before_2000['Count'], labels=success_rate_before_2000['Status'], autopct='%1.1f%%',
       explode=explode, colors=colors)
ax2.set_title('Success rate before 2000', fontsize=16)
ax3.pie(success_rate_after_2000['Count'], labels=success_rate_after_2000['Status'], autopct='%1.1f%%',
       explode=explode, colors=colors)
ax3.set_title('Success rate After 2000', fontsize=16)

plt.show()
space_missions['Launch Site'] = space_missions['Location'].str.split(', ').str[1]
space_missions['Country'] = space_missions['Location'].str.split(', ').str[-1]
space_missions.head()
space_missions['Country'].value_counts()
space_missions.loc[space_missions['Country'] == 'Russian Federation', 'Country'] = 'Russia'
space_missions.loc[space_missions['Country'] == 'Russian', 'Country'] = 'Russia'
space_missions.loc[space_missions['Country'] == 'Shahrud Missile Test Site', 'Country'] = 'Iran'
space_missions.loc[space_missions['Country'] == 'New Mexico', 'Country'] = 'USA'
space_missions.loc[space_missions['Country'] == 'Pacific Missile Range Facility', 'Country'] = 'USA'
space_missions.loc[space_missions['Country'] == 'Barents Sea', 'Country'] = 'Russia'

space_missions['Country'].unique()
launch_sites = space_missions.groupby('Country')['Country'].count().to_frame('counts').reset_index()

fig,ax = plt.subplots(figsize=(10,8))
sns.set_style('dark')
ax = sns.barplot(x=launch_sites['Country'], y=launch_sites['counts'], palette=("deep"))
ax.set_xticklabels(launch_sites['Country'], rotation=90)
ax.set_xlabel('Country', fontsize=16)
ax.set_ylabel('Launch Counts', fontsize=16)
ax.set_title('Lauch Location since 1957', fontsize=20)

plt.show()
# removing the empty columns and removing the whitespaces 
launch_cost = space_missions['Rocket'].dropna().str.lstrip().str.rstrip()
launch_cost = launch_cost.str.replace(',','')
launch_cost = launch_cost.astype('float')

fig, ax = plt.subplots(figsize=(10,5))
ax = sns.distplot(launch_cost, bins=100)
ax.set_title('Cost distribution ($ Million)', fontsize=16)
plt.show()
space_x_data = space_missions[space_missions['Company Name'] == 'SpaceX']
space_x_data.head()
yearly_mission_count = space_x_data.groupby('Mission Year')['Mission Year'].count().to_frame('Yearly_Count').reset_index()

fig, ax = plt.subplots(figsize=(10,6))
ax = sns.barplot(x=yearly_mission_count['Mission Year'], y=yearly_mission_count['Yearly_Count'], color="#5E7CE2")
ax.set_yticks(np.arange(0, 24, step=2))
ax.set_ylabel('Launch Count', fontsize=16)
ax.set_xlabel('Year', fontsize=16)
ax.set_title('SpaceX Launches', fontsize=20)
plt.show()
spacex_launch_sites = space_x_data.groupby('Launch Site')['Launch Site'].count().to_frame('Launch_Count').reset_index()

fig, ax = plt.subplots(figsize=(8,6))
ax = sns.barplot(x=spacex_launch_sites['Launch Site'], y=spacex_launch_sites['Launch_Count'], color="#5E7CE2")
ax.set_ylabel('Launch Count', fontsize=16)
ax.set_xlabel('Launche Sites', fontsize=16)
ax.set_xticklabels(spacex_launch_sites['Launch Site'], rotation=90)
ax.set_title('SpaceX Launche Sites', fontsize=20)
plt.show()
spacex_success_rate = mission_success(space_x_data)
spacex_success_rate

fig, ax = plt.subplots()
ax.pie(spacex_success_rate['Count'], labels=spacex_success_rate['Status'], autopct='%1.1f%%', 
        explode=explode, colors=colors)
ax.set_title('SpaceX Success Rate', fontsize=16)

plt.show()
launch_cost = space_x_data['Rocket'].dropna().str.lstrip().str.rstrip()
launch_cost = launch_cost.str.replace(',','')
launch_cost = launch_cost.astype('float')

fig, ax = plt.subplots(figsize=(10,5))

ax = sns.distplot(launch_cost, bins=20)
ax.set_title('Cost distribution ($ Million)', fontsize=16)
plt.show()