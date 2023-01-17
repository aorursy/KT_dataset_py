import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
data.head()
states_india = data['State/UnionTerritory']

cases_india = data['Confirmed']

cured_india = data['Cured']
states_set = set(states_india)

states_set = sorted(states_set)

states_set.remove('Unassigned')

#print(states_set)
states_cases = dict()
for item in states_set:

    case_count = np.array(cases_india[states_india == item])

    states_cases[item] = case_count[-1]
data_frame = []

for key,value in states_cases.items():

    data_frame.append(list((key,value)))

df = pd.DataFrame(data_frame, columns = ['States', 'Cases']) 
print(df)
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

states = states_cases.keys()

cases = states_cases.values()

ax.bar(states,cases,color='#581845')

plt.xlabel('States', fontsize=20)

plt.ylabel('No. of Cases', fontsize=20)

plt.xticks(rotation=90)

plt.gcf().set_size_inches([10,8])

plt.show()
states_case_copy = states_cases.copy()

sorted_cases = sorted(states_case_copy.items(),key = lambda item:item[1],reverse=True)
print(sorted_cases)
states_list = list()

case_list   = list()

for item in sorted_cases:

    states_list.append(item[0])

    case_list.append(item[1])
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

states = states_list

cases = case_list

ax.bar(states,cases,color='#581845')

plt.xlabel('States', fontsize=20)

plt.ylabel('No. of Cases', fontsize=20)

plt.xticks(rotation=90)

plt.gcf().set_size_inches([10,8])

plt.show()
states_cured = dict()
cured_count = []

for item in states_set:

    cured_count = np.array(cured_india[states_india == item])

    states_cured[item] = cured_count[-1]
data_frame = []

for key,value in states_cured.items():

    data_frame.append(list((key,value)))

df = pd.DataFrame(data_frame, columns = ['States', 'Cured']) 
print(df)
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

states = states_cured.keys()

cured = states_cured.values()

ax.bar(states,cured,color='#399C00')

plt.xlabel('States', fontsize=20)

plt.ylabel('No. of Cured', fontsize=20)

plt.xticks(rotation=90)

plt.gcf().set_size_inches([10,8])

plt.show()
states_cured_copy = states_cured.copy()

sorted_cured = sorted(states_cured_copy.items(),key = lambda item:item[1],reverse=True)
print(sorted_cured)
states_list = list()

cured_list   = list()

for item in sorted_cured:

    states_list.append(item[0])

    cured_list.append(item[1])
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

states = states_list

cured = cured_list

ax.bar(states,cured,color='#399C00')

plt.xlabel('States', fontsize=20)

plt.ylabel('No. of Cured', fontsize=20)

plt.xticks(rotation=90)

plt.gcf().set_size_inches([10,8])

plt.show()


N = len(states_set)

cases = states_cases.values()



ind = np.arange(N)  

width = 0.35      



fig = plt.figure()

ax = fig.add_subplot(111)

rects1 = ax.bar(ind, cases, width, color='#581845')



cured = states_cured.values()



rects2 = ax.bar(ind+width, cured, width, color='#399C00')

ax.set_xlabel('States')

ax.set_ylabel('Cases/Cured')

ax.set_title('Cured and Cases')

ax.set_xticks(ind + width / 2)

ax.set_xticklabels( states_cured.keys(),rotation = 90)



ax.legend( (rects1[0], rects2[0]), ('Cases', 'Cured') )

plt.gcf().set_size_inches([10,8])

plt.show()
dates_india = data['Date']

cases_india = data['Confirmed']
dates_india_set = sorted(set(dates_india))
dates_cases = dict()
for date in dates_india_set:

    dates_cases[date] = sum(cases_india[dates_india == date])
from datetime import datetime

dates_list = list(dates_cases.keys())

dates_list.sort(key=lambda date: datetime.strptime(date,"%d/%m/%y"))
sorted_growth_dict = dict()

for date in dates_list:

    sorted_growth_dict[date] = dates_cases[date]
print(sorted_growth_dict)
data_frame = []

for key,value in sorted_growth_dict.items():

    data_frame.append(list((key,value)))
df = pd.DataFrame(data_frame, columns = ['Date', 'Cases']) 
print(df)
casesCount_From_March = list(sorted_growth_dict.values())

casesCount_From_March = casesCount_From_March [31:]
print(casesCount_From_March)
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

dates = np.arange(1,len(sorted_growth_dict.keys())-30,1)

cases = casesCount_From_March

ax.plot(dates,cases,color='#399C00',marker='o',markersize = 5,markerfacecolor='#00399C',markeredgecolor='black')

plt.xlabel('DaysFrom March Month', fontsize=20)

plt.ylabel('Cases', fontsize=20)

plt.xticks(rotation=90)

plt.gcf().set_size_inches([10,8])

plt.show()
cases_March = casesCount_From_March[0:31]

cases_April = casesCount_From_March[31:]
dates_march = list(sorted_growth_dict.keys())[31:62]

dates_april = list(sorted_growth_dict.keys())[62:]
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

dates = dates_march

cases = cases_March 

ax.plot(dates,cases,color='#399C00',marker='o',markersize = 5,markerfacecolor='#00399C',markeredgecolor='black')

plt.xlabel('Dates', fontsize=20)

plt.ylabel('Cases', fontsize=20)

plt.xticks(rotation=90)

plt.gcf().set_size_inches([10,8])

plt.show()
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

dates = dates_april

cases = cases_April

ax.plot(dates,cases,color='#399C00',marker='o',markersize = 5,markerfacecolor='#00399C',markeredgecolor='black')

plt.xlabel('Dates', fontsize=20)

plt.ylabel('Cases', fontsize=20)

plt.xticks(rotation=90)

plt.gcf().set_size_inches([10,8])

plt.show()
deaths_india = data['Deaths']
dates_deaths = dict()
for date in dates_india_set:

    dates_deaths[date] = sum(deaths_india[dates_india == date])
sorted_death_dict = dict()

for date in dates_list:

    sorted_death_dict[date] = dates_deaths[date]
deathsCount_From_March = list(sorted_death_dict.values())[31:]
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

dates = np.arange(1,len(sorted_death_dict.keys())-30,1)

deaths = deathsCount_From_March

ax.plot(dates,deaths,color='#DA0000',marker='o',markersize = 5,markerfacecolor='#00399C',markeredgecolor='black')

plt.xlabel('Days From March Month', fontsize=20)

plt.ylabel('Deaths:', fontsize=20)

plt.xticks(rotation=90)

plt.gcf().set_size_inches([10,8])

plt.show()
deaths_March = deathsCount_From_March[0:31]

deaths_April = deathsCount_From_March[31:]
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

dates = dates_march

deaths = deaths_March 

ax.plot(dates,deaths,color='#DA0000',marker='o',markersize = 5,markerfacecolor='#00399C',markeredgecolor='black')

plt.xlabel('Dates', fontsize=20)

plt.ylabel('Deaths', fontsize=20)

plt.xticks(rotation=90)

plt.gcf().set_size_inches([10,8])

plt.show()
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

dates = dates_april

deaths = deaths_April

ax.plot(dates,deaths,color='#DA0000',marker='o',markersize = 5,markerfacecolor='#00399C',markeredgecolor='black')

plt.xlabel('Dates', fontsize=20)

plt.ylabel('Deaths', fontsize=20)

plt.xticks(rotation=90)

plt.gcf().set_size_inches([10,8])

plt.show()
rise_in_cases = np.array(casesCount_From_March)
rise_in_cases.shape
new_cases = rise_in_cases[1:]

old_cases = rise_in_cases[0:len(new_cases)]

percentage_rise_per_day = np.array(((new_cases  - old_cases)/old_cases )* 100)
print(percentage_rise_per_day)
np.mean(percentage_rise_per_day)
np.sort(percentage_rise_per_day)
outliers = np.array([3.33333333,3.44827586,66.66666667,366.66666667])
average_rise = (np.sum(percentage_rise_per_day) - np.sum(outliers))/(len(percentage_rise_per_day) - len(outliers))
average_rise
next_30_days_projection = []

today_cases = 9352 #13 April 2020

for i in range(30):

    cases = today_cases + ((today_cases * average_rise)/100)

    next_30_days_projection.append(cases)

    today_cases = cases
print(next_30_days_projection)
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

dates = np.arange(1,31,1)

cases = next_30_days_projection

ax.plot(dates,cases,color='#399C00',marker='o',markersize = 5,markerfacecolor='#00399C',markeredgecolor='black')

plt.xlabel('Dates:14/04/20 to 14/05/20', fontsize=20)

plt.ylabel('Cases', fontsize=20)

plt.xticks(rotation=90)

plt.gcf().set_size_inches([10,8])

plt.show()
dates_next_30_days = []

intial_date = 14

start_month = "/04/20"

for i in range(30):

    if(intial_date == 30):

        intial_date= 1

        start_month = "/05/20"

    if(intial_date < 10):

        dates_next_30_days.append('0' +str(intial_date)+start_month)

    else:

        dates_next_30_days.append(str(intial_date)+start_month)

    intial_date = intial_date + 1
projection_dict = dict()

for i in range(30):

    projection_dict[dates_next_30_days[i]] = next_30_days_projection[i]
data_frame = []

for key,value in projection_dict.items():

    data_frame.append(list((key,int(value))))

df = pd.DataFrame(data_frame, columns = ['Date', 'Cases'])  
df = pd.DataFrame(data_frame, columns = ['Date', 'Cases'])  
df