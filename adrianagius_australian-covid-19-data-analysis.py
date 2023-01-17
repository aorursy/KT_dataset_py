#package import



import numpy as np 

import matplotlib.pyplot as plt 

import pandas as pd 

import random

import math

import time

import datetime

plt.style.use('ggplot')
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
#prep

cols = confirmed_df.keys()

confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]

deaths = deaths_df.loc[:, cols[4]:cols[-1]]

recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]



#ausfilters

ausdeaths = deaths_df[deaths_df['Country/Region']=='Australia'].loc[:, cols[4]:cols[-1]]

ausrecoveries = recoveries_df[recoveries_df['Country/Region']=='Australia'].loc[:, cols[4]:cols[-1]]

ausconfirmed = confirmed_df[confirmed_df['Country/Region']=='Australia'].loc[:, cols[4]:cols[-1]]

ausall = confirmed_df[confirmed_df['Country/Region']=='Australia']



#southkoreafilters

southkoreadeaths = deaths_df[deaths_df['Country/Region']=='Korea, South'].loc[:, cols[4]:cols[-1]]

southkorearecoveries = recoveries_df[recoveries_df['Country/Region']=='Korea, South'].loc[:, cols[4]:cols[-1]]

southkoreaconfirmed = confirmed_df[confirmed_df['Country/Region']=='Korea, South'].loc[:, cols[4]:cols[-1]]



#italyfilters

italydeaths = deaths_df[deaths_df['Country/Region']=='Italy'].loc[:, cols[4]:cols[-1]]

italyrecoveries = recoveries_df[recoveries_df['Country/Region']=='Italy'].loc[:, cols[4]:cols[-1]]

italyconfirmed = confirmed_df[confirmed_df['Country/Region']=='Italy'].loc[:, cols[4]:cols[-1]]





#workingwithithedateconfines

dates = confirmed.keys()



#worldtotals

world_cases = []

total_deaths = [] 

total_recovered = [] 

total_active = [] 



#worldrates

mortality_rate = []

recovery_rate = [] 



#italyrates

italy_case_rate = []

italy_death_rate = []

italy_recovery_rate = []



#italytotals

italy_cases = [] 

italy_deaths = []

italy_recovery = []

italy_active = [] 



#ausrates

australia_case_rate = []

australia_death_rate = []

australia_recovery_rate = []



#austotals

australia_cases = [] 

australia_deaths = []

australia_recovery = []

australia_active = [] 



#southkoreatotals

southkorea_cases = [] 

southkorea_deaths = []

southkorea_recovery = []

southkorea_active = [] 



#southkorearates

southkorea_case_rate = []

southkorea_death_rate = []

southkorea_recovery_rate = []



for i in dates:

    

    confirmed_sum = confirmed[i].sum()

    death_sum = deaths[i].sum()

    recovered_sum = recoveries[i].sum()

    active_sum = confirmed_sum-death_sum-recovered_sum

    

    aus_confirmed_sum = ausconfirmed[i].sum()   

    aus_death_sum = ausdeaths[i].sum()

    aus_recovered_sum = ausrecoveries[i].sum()   

    aus_active_sum = aus_confirmed_sum-aus_death_sum-aus_recovered_sum

    

    italy_confirmed_sum = italyconfirmed[i].sum()   

    italy_death_sum = italydeaths[i].sum()

    italy_recovered_sum = italyrecoveries[i].sum()   

    italy_active_sum = italy_confirmed_sum-italy_death_sum-italy_recovered_sum

    

    southkorea_confirmed_sum = southkoreaconfirmed[i].sum()   

    southkorea_death_sum = southkoreadeaths[i].sum()

    southkorea_recovered_sum = southkorearecoveries[i].sum()   

    southkorea_active_sum = southkorea_confirmed_sum-southkorea_death_sum-southkorea_recovered_sum

    

    #metricstotals

    world_cases.append(confirmed_sum)

    total_deaths.append(death_sum)

    total_recovered.append(recovered_sum)

    total_active.append(active_sum)

    

    #metricsrates

    mortality_rate.append(death_sum/confirmed_sum)

    recovery_rate.append(recovered_sum/confirmed_sum)



    #austotals

    australia_active.append(aus_active_sum)

    australia_cases.append(confirmed_df[confirmed_df['Country/Region']=='Australia'][i].sum())

    australia_deaths.append(deaths_df[deaths_df['Country/Region']=='Australia'][i].sum())

    australia_recovery.append(recoveries_df[recoveries_df['Country/Region']=='Australia'][i].sum())

    

    #ausrates

    australia_death_rate.append(aus_death_sum/aus_confirmed_sum)

    australia_recovery_rate.append(aus_recovered_sum/aus_confirmed_sum)

    

    #italytotals

    italy_active.append(italy_active_sum)

    italy_cases.append(confirmed_df[confirmed_df['Country/Region']=='Italy'][i].sum())

    italy_deaths.append(deaths_df[deaths_df['Country/Region']=='Italy'][i].sum())

    italy_recovery.append(recoveries_df[recoveries_df['Country/Region']=='Italy'][i].sum())

    

    #italyrates

    italy_death_rate.append(italy_death_sum/italy_confirmed_sum)

    italy_recovery_rate.append(italy_recovered_sum/italy_confirmed_sum)

    

    #southkoreatotals

    southkorea_active.append(southkorea_active_sum)

    southkorea_cases.append(confirmed_df[confirmed_df['Country/Region']=='Korea, South'][i].sum())

    southkorea_deaths.append(deaths_df[deaths_df['Country/Region']=='Korea, South'][i].sum())

    southkorea_recovery.append(recoveries_df[recoveries_df['Country/Region']=='Korea, South'][i].sum())

    

    #southkorearates

    southkorea_death_rate.append(southkorea_death_sum/southkorea_confirmed_sum)

    southkorea_recovery_rate.append(southkorea_recovered_sum/southkorea_confirmed_sum)

    

days_in_future = 10

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)

adjusted_dates = future_forcast[:-10]



days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)

world_cases = np.array(world_cases).reshape(-1, 1)

total_deaths = np.array(total_deaths).reshape(-1, 1)



def daily_increase(data):

    d = [] 

    for i in range(len(data)):

        if i == 0:

            d.append(data[0])

        else:

            d.append(data[i]-data[i-1])

    return d 



def flatten_the_curve(data):

    e = [] 

    for i in range(len(data)):

        if i == 0:

            e.append(data[0])

        else:

            e.append((data[i].sum()/data[i-1].sum()-1))

    return e 



aus_daily_increase = daily_increase(australia_cases)

aus_curve_flatten = flatten_the_curve(australia_cases)

adjusted_dates = adjusted_dates.reshape(1, -1)[0]

total_recovered = np.array(total_recovered).reshape(-1, 1)
plt.figure(figsize=(15, 10))

plt.plot(adjusted_dates, world_cases)

plt.title('Total number of Global COVID-19 Cases Over Time', size=20)

plt.xlabel('Days since 22 January 2020', size=20)

plt.legend(['Cases'], prop={'size': 20})

plt.ylabel('Total Number of Cases', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(15, 10))

plt.plot(adjusted_dates, world_cases)

plt.plot(adjusted_dates, total_deaths)

#plt.plot(adjusted_dates, total_recovered)

plt.title('Total number of Global COVID-19 Cases and Deaths', size=20)

plt.xlabel('Days since 22 January 2020', size=20)

#plt.legend(['Active','Deaths','Recoveries'], prop={'size': 20})

plt.legend(['Cases','Deaths'], prop={'size': 20})

plt.ylabel('Total Number of Cases', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(15, 10))

plt.plot(adjusted_dates, mortality_rate)

#plt.plot(adjusted_dates, recovery_rate)

plt.title('Global COVID-19 Rate of Death Rate Over Time', size=20)

plt.xlabel('Days since 22 January 2020', size=20)

#plt.legend(['Deaths','Recoveries'], prop={'size': 20})

plt.legend(['Deaths'], prop={'size': 20})

plt.ylabel('Rate', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(15, 10))

plt.plot(adjusted_dates, australia_cases)

plt.title('Total number of Australian COVID-19 Cases Over Time', size=20)

plt.xlabel('Days since 22 January 2020', size=20)

plt.vlines(x=47, ymin=0, ymax=100, color='black', linestyle='dashed')

plt.legend(['Cases','100 Cases'], prop={'size': 15})

plt.ylabel('Total Number of Cases', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(16, 9))

plt.bar(adjusted_dates, aus_daily_increase)

plt.title('Australia Daily Increases in Confirmed Cases', size=20)

plt.xlabel('Days Since 22 January 2020', size=20)

plt.ylabel('# of Cases', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(15, 10))

plt.plot(adjusted_dates, aus_curve_flatten)

plt.title('Rate of case increase', size=20)

plt.xlabel('Days since 22 January 2020', size=20)

plt.vlines(x=47, ymin=0, ymax=1, color='black', linestyle='dashed')

plt.legend(['Cases','100 Cases'], prop={'size': 15})

plt.ylabel('Total Number of Cases', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
ausstates = list(ausall['Province/State'].unique())

latest_confirmed = ausall[dates[-1]]



aus_confirmed_cases = []

for i in ausstates:

    cases = latest_confirmed[ausall['Province/State']==i].sum()

    if cases > 0:

        aus_confirmed_cases.append(cases)

        

plt.figure(figsize=(16, 9))

plt.barh(ausstates, aus_confirmed_cases)

plt.title('# of Covid-19 Confirmed Cases in Australian States', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
wacases = []

viccases = []

tascases = []

sacases = []

qldcases = []

ntcases = []

nswcases = []

actcases = []





for i in dates:



    #states

    wacases.append(confirmed_df[confirmed_df['Province/State']=='Western Australia'][i].sum())

    viccases.append(confirmed_df[confirmed_df['Province/State']=='Victoria'][i].sum())

    tascases.append(confirmed_df[confirmed_df['Province/State']=='Tasmania'][i].sum())

    sacases.append(confirmed_df[confirmed_df['Province/State']=='South Australia'][i].sum())

    qldcases.append(confirmed_df[confirmed_df['Province/State']=='Queensland'][i].sum())

    ntcases.append(confirmed_df[confirmed_df['Province/State']=='Northern Territory'][i].sum())

    nswcases.append(confirmed_df[confirmed_df['Province/State']=='New South Wales'][i].sum())

    actcases.append(confirmed_df[confirmed_df['Province/State']=='Australian Capital Territory'][i].sum())



plt.figure(figsize=(15, 10))

plt.plot(adjusted_dates, wacases)

plt.plot(adjusted_dates, viccases)

plt.plot(adjusted_dates, tascases)

plt.plot(adjusted_dates, sacases)

plt.plot(adjusted_dates, qldcases)

plt.plot(adjusted_dates, ntcases)

plt.plot(adjusted_dates, nswcases)

plt.plot(adjusted_dates, actcases)

plt.title('Total number of Australian COVID-19 by State over Time', size=20)

plt.xlabel('Days since 22 January 2020', size=20)

plt.vlines(x=47, ymin=0, ymax=100, color='black', linestyle='dashed')

plt.legend(['WA','VIC','TAS','SA','QLD','NT','NSW','ACT','100 Cases'], prop={'size': 15})

plt.ylabel('Total Number of Cases', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(15, 10))

#plt.plot(adjusted_dates, australia_active)

plt.plot(adjusted_dates, australia_cases)

plt.plot(adjusted_dates, australia_deaths)

plt.title('Total number of Australian COVID-19 Cases and Death Over Time', size=20)

plt.xlabel('Days since 22 January 2020', size=20)

plt.vlines(x=47, ymin=0, ymax=100, color='black', linestyle='dashed')

plt.legend(['Cases','Deaths','100 Cases'], prop={'size': 15})

#plt.legend(['Active Cases','Deaths','Recoveries','100 Cases'], prop={'size': 15})

plt.ylabel('Total Number of Cases', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(15, 10))

plt.plot(adjusted_dates, australia_death_rate)

#plt.plot(adjusted_dates, australia_recovery_rate)

plt.title('Australian COVID-19 Rate of Deaths Over Time', size=20)

plt.xlabel('Days since 22 January 2020', size=20)

plt.vlines(x=47, ymin=0, ymax=0.7, color='black', linestyle='dashed')

#plt.legend(['Deaths','Recoveries','100 Cases'], prop={'size': 15})

plt.legend(['Deaths','100 Cases'], prop={'size': 15})

plt.ylabel('Rate', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(15, 10))

plt.plot(adjusted_dates, southkorea_cases)

plt.title('Total number of South Korean COVID-19 Cases Over Time', size=20)

plt.xlabel('Days since 22 January 2020', size=20)

plt.vlines(x=29, ymin=0, ymax=100, color='black', linestyle='dashed')

plt.legend(['Cases','100 Cases'], prop={'size': 15})

plt.ylabel('Total Number of Cases', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(15, 10))

plt.plot(adjusted_dates, southkorea_cases)

plt.plot(adjusted_dates, southkorea_deaths)

#plt.plot(adjusted_dates, southkorea_recovery)

plt.title('Total number of South Korean COVID-19 Cases and Deaths Over Time', size=20)

plt.xlabel('Days since 22 January 2020', size=20)

plt.vlines(x=29, ymin=0, ymax=100, color='black', linestyle='dashed')

plt.legend(['Cases','Deaths','100 Cases'], prop={'size': 15})

#plt.legend(['Active Cases','Deaths','Recoveries','100 Cases'], prop={'size': 15})

plt.ylabel('Total Number of Cases', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(15, 10))

plt.plot(adjusted_dates, southkorea_death_rate)

#plt.plot(adjusted_dates, southkorea_recovery_rate)

plt.title('South Korean COVID-19 Rate of Deaths Over Time', size=20)

plt.xlabel('Days since 22 January 2020', size=20)

plt.vlines(x=29, ymin=0, ymax=0.7, color='black', linestyle='dashed')

plt.legend(['Deaths','100 Cases'], prop={'size': 15})

plt.ylabel('Rate', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(15, 10))

plt.plot(adjusted_dates, italy_cases)

plt.title('Total number of Italian COVID-19 Cases Over Time', size=20)

plt.xlabel('Days since 22 January 2020', size=20)

plt.legend(['Cases'], prop={'size': 15})

plt.ylabel('Total Number of Cases', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(15, 10))

plt.plot(adjusted_dates, italy_cases)

plt.plot(adjusted_dates, italy_deaths)

plt.title('Total number of Italian COVID-19 Cases and Deaths Over Time', size=20)

plt.legend(['Cases','Deaths'], prop={'size': 15})

plt.xlabel('Days since 22 January 2020', size=20)

plt.ylabel('Total Number of Cases', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(15, 10))

plt.plot(adjusted_dates, italy_death_rate)

plt.title('Italian COVID-19 Rate of Death', size=20)

plt.xlabel('Days since 22 January 2020', size=20)

plt.vlines(x=30, ymin=0, ymax=0.7, color='black', linestyle='dashed')

plt.legend(['Deaths','100 Cases'], prop={'size': 15})

plt.ylabel('Rate', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(15, 10))

plt.plot(adjusted_dates, australia_cases)

plt.plot(adjusted_dates, southkorea_cases)

plt.plot(adjusted_dates, italy_cases)

plt.title('Total number of Australian, Italy and South Korea COVID-19 Cases Over Time', size=20)

plt.legend(['Australia','South Korea','Italy'], prop={'size': 15})

plt.xlabel('Days since 22 January 2020', size=20)

plt.ylabel('Total Number of Cases', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(15, 10))

plt.plot(adjusted_dates, australia_death_rate)

plt.plot(adjusted_dates, southkorea_death_rate)

plt.plot(adjusted_dates, italy_death_rate)

plt.title('Rate of death from COVID-19 cases in Australia, Italy and South Korea Over Time', size=20)

plt.vlines(x=47, ymin=0, ymax=0.1, color='black', linestyle='dashed')

plt.legend(['Australia','South Korea','Italy','100 Aus Cases'], prop={'size': 15})

plt.xlabel('Days since 22 January 2020', size=20)

plt.ylabel('Rate of Death', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()