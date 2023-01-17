import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from bs4 import BeautifulSoup

import requests

from scipy.stats import norm

from scipy.stats import t

from scipy import stats

from sklearn import preprocessing

from IPython.display import Image

pd.set_option('display.max_columns', None)
data = pd.read_csv("../input/us-accidents/US_Accidents_Dec19.csv")
data.head()
data['Start_Time'].dtypes
data['Start_Time'] = pd.to_datetime(data['Start_Time'])

data['Start_Time'] = data['Start_Time'].dt.normalize()

data['Start_Time'].dtypes
data.columns
data_clean = data.drop(columns = ['End_Time', 'Temperature(F)', 'Wind_Chill(F)', 

                                  'End_Lat', 'End_Lng', 

                                  'Humidity(%)', 'Pressure(in)',

                                   'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)',

                                   'Precipitation(in)','Amenity', 'Bump', 'Crossing', 'Number',

                                  'Street', 'Side', 'City', 'County', 'State',

                                   'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',

                                   'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop',

                                   'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight',

                                   'Astronomical_Twilight', 'Timezone','Airport_Code','Weather_Timestamp' ])
data_clean.isnull().sum()
data_year = data[(data['Start_Time'] > '2016-08-23') &

                (data['Start_Time'] <= '2019-08-23')]
traffic_codes = data['TMC'].value_counts()

traffic_codes = pd.DataFrame(traffic_codes)

traffic_codes.reset_index()

traffic_codes.index.names = ['Code']

traffic_codes.head()
event_codes = pd.read_html('https://wiki.openstreetmap.org/wiki/TMC/Event_Code_List')
codes = event_codes[0].drop(columns = ['N','Q','T','D','U','C','R'])
codes.head()
df_codes = codes.merge(traffic_codes, left_on='Code', right_on='Code')

df_codes = df_codes.rename(columns = {'TMC':'# of Accidents'})
df_codes.sort_values(by = '# of Accidents',ascending = False)
severity_time = data_year[['Severity','Start_Time']]

severity_time['Start_Time'] = severity_time['Start_Time'].dt.date

sev_count = severity_time.groupby(severity_time['Start_Time'])['Severity'].value_counts()

sev_count = severity_time.groupby(severity_time['Start_Time'])['Severity'].value_counts().unstack(level = 1)

sev_count.reset_index(level=0, inplace=True)

sev_count.head()
accidents_by_state = [data_year[data_year['State'] == i].count()['ID'] 

                      for i in data_year.State.unique()]

accidents_by_state.sort(reverse = True)

states = data_year.State.unique()

state_severity = data_year[['Severity', 'ID', 'State']]

grouped = state_severity.groupby(['State']).sum().reset_index()

sort_grouped = grouped.sort_values(by = 'Severity', ascending = False)

sort_grouped.head()
stateabbr = pd.read_html('https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States')[0]

stateabbr.head()
state_list = stateabbr.values

state_list = pd.DataFrame(state_list)

state_list = state_list[[0,1]]

state_list = state_list.rename(columns = {0:'Long',1:'State'})

state_list.head()
df = pd.read_html('https://en.wikipedia.org/wiki/Seat_belt_laws_in_the_United_States')[0]

df = df.drop(['Type of law','Date of first law','Who is covered','Base fine before fees'], axis = 1)

df = df.rename(columns = {'State, federal district, or territory':'State','Seat Belt Usage (2017)[7][note 2]':'Seatbelt'})

df = df.drop(52) ### Wake Island has no data.

df.sort_values(by = 'Seatbelt', ascending = False).head()
df_states = sort_grouped.merge(state_list, left_on='State', right_on='State')

df_states = df_states.rename(columns = {'State':'Abbv','Long':'State'})

df_states.head()
df_states_full = df_states.merge(df, left_on = 'State', right_on='State')

df_states_full = df_states_full[['Abbv','State','Severity','Seatbelt']]

df_states_full['Seatbelt'] = df_states_full['Seatbelt'].str.replace('%','')

df_states_full.sort_values(by = 'Severity', ascending = False).head()
df_states_full.Seatbelt = df_states_full.Seatbelt.astype(str).astype(float)

df_states_full.dtypes
top_grouped = state_list.merge(sort_grouped, left_on='State', right_on='State')

top_grouped = top_grouped.sort_values(by = 'Severity', ascending = False)

top_grouped.Long = top_grouped.Long.str.replace(("[['E']]"),"")

top_grouped.head()
Image("../input/tableauimages/download.png")
sns.set(style="white", context="talk")



fig, ax = plt.subplots(figsize=(16,10)) 

sns.barplot(x = top_grouped['Severity'], y = top_grouped['Long'].head(10))

plt.xticks(rotation=90)

plt.yticks(fontsize = 20)

plt.title('Top 10 # of Accidents by State Aug 2016 - Aug 2019', fontsize = 30)

plt.xlabel('# of Accidents', fontsize = 20)

plt.ylabel('State', fontsize = 20)
severities4_state = data[data['Severity'] == 4]['State'].value_counts()

severities3_state = data[data['Severity'] == 3]['State'].value_counts()

severities2_state = data[data['Severity'] == 2]['State'].value_counts()

severities1_state = data[data['Severity'] == 1]['State'].value_counts()

names = ['Severity 4', 'Severity 3', 'Severity 2', 'Severity 1']
sever_df = pd.concat([severities4_state, severities3_state,

                     severities2_state, severities1_state],

                    axis = 1)

sever_df.columns = names

sever_df.head(10)

sever_df.fillna(0)

sever_df.reset_index(inplace = True)

sever_df = sever_df.rename(columns = {'index':'State'})

sever_df = sever_df.fillna(0)

sever_df_sort = sever_df.sort_values(by = 'Severity 4', ascending = False)
plt.figure(figsize=(16, 10))

ax = sever_df_sort.plot(x = 'State', y = 'Severity 4', kind = "bar", figsize = (16,10))

plt.show
sns.set(style="white", context="talk")



fig, ax = plt.subplots(figsize=(16,10)) 

data_year['Weather_Condition'].value_counts().head(10).plot.bar(color = ['coral', 

                                                                         'darkorange',

                                                                         'limegreen', 

                                                                         'navy', 

                                                                         'orchid',

                                                                         'slateblue',

                                                                        'teal',

                                                                        'royalblue',

                                                                        'lightsteelblue',

                                                                        'darkblue'],

                                                               alpha = 0.75)

plt.xlabel('Weather Condition', fontsize = 20)

plt.ylabel('# of Accidents', fontsize = 20)

ax.tick_params(labelsize= 15)

plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.title('Most Common Weather Conditions at the time of Accident', fontsize = 25)
plt.figure(figsize=(16, 10))

plt.plot(sev_count.Start_Time, sev_count[1], label = 'Severity 1', markevery = 5)

plt.plot(sev_count.Start_Time, sev_count[2], label = 'Severity 2', markevery = 5)

plt.plot(sev_count.Start_Time, sev_count[3], label = 'Severity 3', markevery = 5)

plt.plot(sev_count.Start_Time, sev_count[4], label = 'Severity 4', markevery = 5)



plt.xlabel('Date')

plt.ylabel('# of Accidents')

plt.title('# of Accidents by Severity against Time')

plt.legend()

plt.show()
data_year['Weather_Condition 1'] = data_year['Weather_Condition'].str.split('/').str[0].str.strip()

data_year['Weather_Condition 2'] = data_year['Weather_Condition'].str.split('/').str[1].str.strip()

data_year['Weather_Condition 3'] = data_year['Weather_Condition'].str.split('/').str[2].str.strip()
data_year.head()
data_year.loc[data_year['Weather_Condition'] == 'Sand / Dust Whirlwinds / Windy']
df_unmerged = data_year[['Severity',

                         'Weather_Condition 1',

                         'Weather_Condition 2',

                         'Weather_Condition 3']]

df_unmerged = df_unmerged.loc[df_unmerged['Weather_Condition 1'].notna()]

df_unmerged.drop(columns = ['Weather_Condition 2', 'Weather_Condition 3'], inplace = True)

dummies = pd.get_dummies(df_unmerged)

dummies.head()
columns = list(dummies.columns)

columns.remove('Severity')



correlation_values = []

weather_types = []



for column in columns:

    corr = dummies['Severity'].corr(dummies[column])

    if corr >= 0.01 or corr <= -0.01:

        print(f'The correlation for {column} = {corr}')

        print('------------------------------------------------------')

        weather_types.append(column)

        correlation_values.append(corr)
weather_types = [item.replace('Weather_Condition 1_','') for item in weather_types]

sns.set(style="white", context="talk")

fig, ax = plt.subplots(figsize=(16,10)) 

sns.barplot(x = weather_types, y = correlation_values)

plt.xticks(rotation=90)

plt.title('Correlation: Weather Conditions and Accident Severity\n', fontsize = 30)

plt.xlabel('\nWeather Condition', fontsize = 30)

plt.xticks(fontsize = 20)

plt.ylabel('Correlation', fontsize = 30)
sns.set(style="white", context="talk")

fig, ax = plt.subplots(figsize=(16,10)) 

x = df_states_full['Seatbelt']

y = df_states_full['Severity']

ax = sns.regplot(x, y, 

                  data = df_states_full, scatter_kws = {"s": 250},

                  marker = "+", color = 'r')

ax.set(xlabel = "Seatbelt Useage (%)", ylabel = "Number of Accidents")

result = stats.linregress(x, y)

print("Slope: ", result.slope)

print("Intercept: ", result.intercept)

print("rvalue: ", result.rvalue)

print("pvalue: ", result.pvalue)

print("stderr: ", result.stderr)
df_states_nooutlier = df_states_full.drop([0, 1, 2])

df_states_nooutlier.head()

df_states_nooutlier.Seatbelt = df_states_full.Seatbelt.astype(str).astype(float)

sns.set(style="white", context="talk")

fig, ax = plt.subplots(figsize=(16,10)) 

x = df_states_nooutlier['Seatbelt']

y = df_states_nooutlier['Severity']

ax = sns.regplot(x, y, 

                  data = df_states_nooutlier,

                  scatter_kws = {"s": 250},

                  marker = "+", color = 'r')

ax.set(xlabel = "Seatbelt Useage (%)", ylabel = "Number of Accidents")

result = stats.linregress(x, y)

print("Slope: ", result.slope)

print("Intercept: ", result.intercept)

print("rvalue: ", result.rvalue)

print("pvalue: ", result.pvalue)

print("stderr: ", result.stderr)
st, p_value =  stats.ttest_1samp(data['Severity'], 2.5)
print(st, p_value)
data['Severity'].mean()
st, p_value =  stats.ttest_1samp(data['Distance(mi)'], 1)
print(st, p_value)
data['Distance(mi)'].mean()
data.loc[data.Severity == 4]['Distance(mi)'].mean()

st, p_value =  stats.ttest_1samp(data.loc[data.Severity == 4]['Distance(mi)'], 1.4)

print(st, p_value)
DUI = pd.read_html('https://backgroundchecks.org/which-states-have-the-worst-dui-problems.html')

DUI[0].head(5)