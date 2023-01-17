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
df = pd.read_csv("../input/us-accidents/US_Accidents_Dec19.csv")
df.head()
df.columns
df_california = df.loc[df['State'] == 'CA']
df_california = df_california.drop(['Source','TMC','Distance(mi)','Description','Number','Street','Amenity','Civil_Twilight','Nautical_Twilight','Astronomical_Twilight'],1)
# df_california.head(1000)
df_california.head(10)
df_california.info()
df_california.City[df_california['City'] == 'Tracy'].value_counts()
pd.set_option('display.max_rows', 500)
import seaborn as sns
import matplotlib.pyplot as plt

city_count  = df_california['City'].value_counts()
city_count = city_count[:20,]
plt.figure(figsize=(25,10))
sns.barplot(city_count.index, city_count.values, alpha=0.8)
plt.title('Accidents as per city in CA')
plt.ylabel('Number of Accidents', fontsize=12)
plt.xlabel('city', fontsize=12)
plt.show()
# df_california.shape
df_california.isnull().sum() > 100000
df_california = df_california.drop(['End_Lat','End_Lng','Wind_Chill(F)','Wind_Speed(mph)','Precipitation(in)'],1)
df_california.shape
city_count_CA = city_count[:10]
city_count_CA
df_california.columns
df_california.to_csv('CaliforniaAccidents.csv',index=False)
city_count_CA.to_csv('Top1city_count_CA0CA.csv',index=False)
total_states = df['State'].unique()
print(total_states)
state_count = {}
for state in total_states:
    state_count[state] = df[df['State'] == state].shape[0]
    print(state_count[state])
print(state_count)
# import json
# with open('state_count.json', 'w') as fp:
#     json.dump(state_count, fp)
df_US = df.drop(['Source','TMC','Distance(mi)','Description','Number','Street','Amenity','Civil_Twilight','Nautical_Twilight','Astronomical_Twilight'],1)
# df_US.head(100)
len(df_US[df_US['State'] == "CA"]['City'].unique())
pd.set_option('display.max_columns', None)
i, US_counties = 0, {}
for index, row in df_US.iterrows():
#     print(row['State'])
    state, city, county = row['State'], row['City'], row['County']
    if state not in US_counties:
        US_counties[state] = {}
    if county not in US_counties[state]:
        US_counties[state][county] = {}
    if city not in US_counties[state][county]:
        US_counties[state][county][city] = 1
    else:
        US_counties[state][county][city] += 1
print(US_counties)
import json
with open('cities_count.json', 'w') as fp:
    json.dump(US_counties, fp)

filter1 = df_US['Crossing'] == True
filter2 = df_US['Bump'] == True
filter3 = df_US['Give_Way'] == True
filter4 = df_US['Junction'] == True
filter5 = df_US['No_Exit'] == True
filter6 = df_US['Railway'] == True
filter7 = df_US['Roundabout'] == True
filter8 = df_US['Station'] == True
filter9 = df_US['Stop'] == True
filter10 = df_US['Traffic_Calming'] == True
filter11 = df_US['Traffic_Signal'] == True
filter12 = df_US['Turning_Loop'] == True

df_US_fil = df_US.where(filter1 | filter2 | filter3 | filter4 | filter5 | filter6 |
                           filter7 | filter8 | filter9 | filter10 | filter11 | filter12)

# df_US_fil.head(100)
# df_US[df_US['Crossing'] == True].head(100)
# df_US_fil[df_US_fil['State'] == nan]
# df_US_fil['State'].isnull().values
df_US_fil = df_US_fil[df_US_fil['State'].notna()]
df_US_fil.head(100)
# df_US_fil.head(100)
df_US_fil[df_US_fil['State'] == "CA"][df_US_fil['Traffic_Signal'] == True]['State'].count()
i, US_gender = 0, {}
gender = {'female' : 0, 'male': 0}
# df_US['State'].unique()
for row in df_US['State'].unique():
#     print(row)
    US_gender[row] = gender
    
print(US_gender)

with open('gender_count.json', 'w') as fp:
    json.dump(US_gender, fp)
    

# df_US.groupby('Weather_Condition').count()
states = df.State.unique()
# val = df_US[df_US['State'] == "TX"]['Weather_Condition'].value_counts().to_dict()
# print(val)
res = {}
for item in states:
    val = df_US[df_US['State'] == item]['Weather_Condition'].value_counts().to_dict()
    res[item] = []
    for k,v in val.items():
        temp = {}
        temp['text'] = k
        temp['value'] = v
        res[item].append(temp)
# print(res)
US_weather_condition = res
causes = ['Crossing', 'Bump', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop','Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
# wc2 = df_US['Bump'] == True
# wc3 = df_US['Give_Way'] == True
# wc4 = df_US['Junction'] == True
# wc5 = df_US['No_Exit'] == True
# wc6 = df_US['Railway'] == True
# wc7 = df_US['Roundabout'] == True
# wc8 = df_US['Station'] == True
# wc9 = df_US['Stop'] == True
# wc10 = df_US['Traffic_Calming'] == True
# wc11 = df_US['Traffic_Signal'] == True
wc12 = df_US['Turning_Loop'] == True
res = {}
for item in states:
    res[item] = []
    for cause in causes:
        temp = {}
        count = df_US[df_US['State'] == item][df_US[cause] == True]['State'].count()
        temp['text'] = cause
        temp['value'] = int(count)
        
        res[item].append(temp)
        
# print(res)
#         print("state: ",item, " cause: ", cause, " count: ",df_US[df_US['State'] == item][df_US[cause] == True]['State'].count())
    
# with open('US_States_Weather_Condition.json', 'w') as fp:
#     json.dump(US_weather_condition, fp)
print(res)
# US_all_causes = res
# with open('US_States_All_Causes.json', 'w') as fp:
#     json.dump(US_all_causes, fp)
US_causes = res
with open('US_States_Causes.json', 'w') as fp:
    json.dump(US_causes, fp)
# print(type(df_US['Weather_Timestamp'][0].to_datetime()))
# df_US['Start_Time'].isna().sum()
df_US['timestamp'] = df_US['Start_Time'].astype('datetime64[ns]')
# df_US.head(10)
# print(type(df_US['timestamp'][0]))

# df_US_TS = df_US.set_index('timestamp').head(10)
# df_US_TS.head(10)
# time = df_US['timestamp'][0]
# df_US['date'] = df_US['timestamp'].dt.date
import datetime
df_US.head(10)
# print( df_US_TS.loc['2019-01-04':'2019-10-06'])
start_date, end_date = datetime.date(2017, 1, 17), datetime.date(2017, 2, 17)

st, et = '2018-01-04 05:46:00', '2018-01-06 05:46:00'
df_US_state = df_US.groupby('State')
mask = (df_US_state.get_group('OH')['date'] > start_date) & (df_US_state.get_group('OH')['date'] <= end_date)

print(df_US.loc[mask])

# print(time.dt.date())
# df_US_TS['2018-01-04 05:46:00':'2018-10-06 05:46:00']
Y, M1, D = 2016, 4, 1
M2 = M1 + 2
D2 = {}
D2[1], D2[2], D2[3], D2[4], D2[5], D2[6] = 31, 28, 31, 30, 31, 30
D2[7], D2[8], D2[9], D2[10], D2[11], D2[12] = 31, 31, 30, 31, 30, 31

states = df_US.State.unique()
res_state = {}
for state in states:
    val = df_US[df_US['State'] == item]['Weather_Condition'].value_counts()
    i = 0
    causes = []
    for k,v in val.items():
        causes.append(k)
        i += 1
        if i >= 5:
            break

    
    res_cause = []
    for cause in causes:
        res = []
        start_date, end_date = datetime.date(2016, 2, 1), datetime.date(2016, 4, 30)
        mask = (df_US[df_US['State'] == state][df_US['Weather_Condition'] == cause]['date'] > start_date) & (df_US[df_US['State'] == state][df_US['Weather_Condition'] == cause]['date'] <= end_date)
        count = df_US[df_US['State'] == state][df_US['Weather_Condition'] == cause].loc[mask].shape[0]
        res.append({"x": "Q1", "y":count})
        print("State: ",state, "Cause: ",cause, " Start Date: ",start_date, " End Date: ", end_date, " count: ",count)
        
        Y, M1, D = 2016, 4, 1
        M2 = M1 + 2
        q = 2
        while Y < 2020:
            start_date, end_date = datetime.date(Y, M1, D), datetime.date(Y, M2, D2[M2])
            mask = (df_US[df_US['State'] == state][df_US['Weather_Condition'] == cause]['date'] > start_date) & (df_US[df_US['State'] == state][df_US['Weather_Condition'] == cause]['date'] <= end_date)
            count = df_US[df_US['State'] == state][df_US['Weather_Condition'] == cause].loc[mask].shape[0]
            res.append({"x" : "Q" + str(i), "y" : count})
            print("State: ",state, "Cause: ",cause, " Start Date: ",start_date, " End Date: ", end_date, " count: ",count)
            i += 1
            M1 += 3
            if M1 > 12:
                Y += 1
                M1 = 1
            M2 = M1 + 2
        res_cause.append({"id": cause, "data":res})
    res_state[state] = res_cause        
    
states = df_US.State.unique()
i=0
# for index, row in df_US.iterrows():
#     i+=1
print(states)
df_US_states = df_US.groupby('State')
states = df_US.State.unique()
# df_US_states['ID'].size()
States = [
  { "label": "Alaska", "value": "AK"},
  { "label": "Alabama", "value": "AL"},
  { "label": "Arkansas", "value": "AR"},
  { "label": "Arizona", "value": "AZ"},
  { "label": "California", "value": "CA" },
  { "label": "Colorado", "value": "CO" },
  { "label": "Connecticut", "value": "CT" },
  { "label": "Washington D.C.", "value": "DC" },
  { "label": "Delaware", "value": "DE" },
  { "label": "Florida", "value": "FL" },
  { "label": "Georgia", "value": "GA" },
  { "label": "Guam", "value": "GU" },
  { "label": "Hawaii", "value": "HA" },
  { "label": "Iowa", "value": "IA" },
  { "label": "Idaho", "value": "ID" },
  { "label": "Illinois", "value": "IL" },
  { "label": "Indiana", "value": "IN" },
  { "label": "Kansas", "value": "KS" },
  { "label": "Kentucky", "value": "KY" },
  { "label": "Louisiana", "value": "LA" },
  { "label": "Massachusetts", "value": "MA" },
  { "label": "Maryland", "value": "MD" },
  { "label": "Maine", "value": "ME" },
  { "label": "Michigan", "value": "MI" },
  { "label": "Minnesota", "value": "MN" },
  { "label": "Missouri", "value": "MO" },
  { "label": "Mississippi", "value": "MS" },
  { "label": "Montana", "value": "MT" },
  { "label": "North Carolina", "value": "NC" },
  { "label": "North Dakota", "value": "ND" },
  { "label": "Nebraska", "value": "NE" },
  { "label": "New Hampshire", "value": "NH" },
  { "label": "New Jersey", "value": "NJ" },
  { "label": "New Mexico", "value": "NM" },
  { "label": "Nevada", "value": "NV" },
  { "label": "New York", "value": "NY" },
  { "label": "Ohio", "value": "OH" },
  { "label": "Oklahoma", "value": "OK" },
  { "label": "Oregon", "value": "OR" },
  { "label": "Pennsylvania", "value": "PA" },
  { "label": "Puerto Rico", "value": "PR" },
  { "label": "Rhode Island", "value": "RI" },
  { "label": "South Carolina", "value": "SC" },
  { "label": "South Dakota", "value": "SD" },
  { "label": "Tennessee", "value": "TN" },
  { "label": "Texas", "value": "TX" },
  { "label": "Utah", "value": "UT" },
  { "label": "Virginia", "value": "VA" },
  { "label": "Virgin Islands", "value": "VI" },
  { "label": "Vermont", "value": "VT" },
  { "label": "Washington", "value": "WA" },
  { "label": "Wisconsin", "value": "WI" },
  { "label": "West Virginia", "value": "WV" },
  { "label": "Wyoming", "value": "WY" },
]
print(len(States))
res = {}
rank_count = {}
rank_list = []
rank_states = {} # key - state, value - rank
sev_states = {} # key - state, value - rank

for state in states:
    res[state] = []
totalAcc = df_US_states['ID'].count().to_dict() # key - state, value - Number of accidents
print(totalAcc)
for k,v in totalAcc.items():
#     res[k].append({"Number of Accidents" : v})
    rank_count[v] = k
    rank_list.append(v)
    
rank_list = sorted(rank_list, reverse=True)
rank = 1
for val in rank_list:
    rank_states[rank_count[val]] = rank
    rank += 1
    
for name, group in df_US_states:
    sev_state = group['Severity'].value_counts().to_dict()
    sev_count, total_sev_count, total_count = 0, 0, 0
    for k,v in sev_state.items():
        if k == 3 or k == 4:
            sev_count += v
        total_count += v
    
    sev_states[name] = round(sev_count/total_count*100,2)

states_set = set()
for state in states:
    states_set.add(state)
    
for item in States:
    if item["value"] in states_set:
        s, n = item["value"], item["label"]
        res[s].append({"name" : "Name", "value" : n})
        res[s].append({"name" : "Code", "value" : s})
    
for k,v in rank_states.items():
    res[k].append({"name" : "Rank", "value" : v})
for k,v in totalAcc.items():
    res[k].append({"name" : "Number of Accidents", "value" : v})
for k,v in sev_states.items():
    res[k].append({"name" : "Severe Accidents", "value" : str(v) + " %"})

    
print(res)
    

import json
with open('state_stats.json', 'w') as fp:
    json.dump(res, fp)

for item in States:
    if item["value"] not in states_set:
        print(item)
