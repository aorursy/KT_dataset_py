import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



import datetime


fp = "../input/jhu-covid19-deaths-21mar2020/time_series_19-covid-Deaths.csv"

time_series_deaths = pd.read_csv(fp)

time_series_deaths = time_series_deaths.rename(columns = {"Country/Region":"Country_Region","Province/State":"Province_State"})

time_series_deaths.groupby("Country_Region")

time_series_deaths.head()


time_series_deaths_US = time_series_deaths.loc[time_series_deaths.Country_Region == 'US']

print("US deaths",time_series_deaths_US.iloc[:,-1].sum())

time_series_deaths_Italy = time_series_deaths.loc[time_series_deaths.Country_Region == 'Italy']

time_series_deaths_Italy = time_series_deaths_Italy.T

print("Italy Deaths =",time_series_deaths_Italy.values[-1])
startdate  = datetime.date(2020,1,18)

today = datetime.date(2020,3,21)

t1 = today - startdate

days = str(t1)

days = int(days[:2])



US_deaths = []

US_deaths_date = []

for d in np.arange(42,days):

  death = time_series_deaths_US.iloc[:,d+1].sum()

  US_deaths.append(death)

  date = time_series_deaths_US.columns[d]

  US_deaths_date.append(date)

y_US = US_deaths

x = US_deaths_date

figure = plt.figure(figsize=[5,5])

plt.xticks(np.arange(0,100,step=4))

plt.title("US Deaths Since First Reported Death")

plt.plot(x,y_US)

plt.show()



time_series_deaths_Italy = time_series_deaths.loc[time_series_deaths.Country_Region == 'Italy']

time_series_deaths_Italy = time_series_deaths_Italy.T



y_Italy = time_series_deaths_Italy.values[34:]



time_series_deaths_SK = time_series_deaths.loc[time_series_deaths.Country_Region == 'Korea, South']

time_series_deaths_SK = time_series_deaths_SK.T



y_SK = time_series_deaths_SK.values[33:]





x1 = np.arange(0,len(time_series_deaths_Italy.values[34:]))

x = time_series_deaths_Italy.index[34:]

figure = plt.figure(figsize=[5,5])

plt.xticks(np.arange(0,100,step=4))

plt.title("Italy, US, South Korea Deaths Since First Death Report")

days_fd = len(time_series_deaths_Italy.values)-4

print("Italy Deaths =",time_series_deaths_Italy.values[-1])

plt.plot(x1,y_Italy, 'g-',label = "Italy")

plt.plot(y_US, label = "US")

plt.plot(y_SK, label = "South Korea")

plt.legend()

days_sinceItalyLD = days - 51 # days since Italy's Lock down

print("Days since italy's lock down (17 days after first death)= ",days_sinceItalyLD)
d=[0,-1,-2,-3,-4,-5,-6,-7,-8,-9]

New_deaths_rate_US=[]

for d in d:

  today = time_series_deaths_US.iloc[:,d-1].sum()

  yesterday = time_series_deaths_US.iloc[:,d-2].sum()

  New_deaths_rate_US.append(today/yesterday)

New_deaths_rate_US.reverse()

New_deaths_rate_US_v = np.array(New_deaths_rate_US)

# Observed growth rate based on 6 day average

Rolling_day_average = New_deaths_rate_US_v.sum()/len(New_deaths_rate_US)

print ("Observed (US) growth rate based on 10 day average = ", Rolling_day_average)

print("Today's (US) growth rate = ",New_deaths_rate_US[-1])

fig = plt.figure(figsize=(5,5))

one = [1]*len(New_deaths_rate_US)

reported_onepointtwo = [1.2]*len(New_deaths_rate_US)

rolling_average = [Rolling_day_average]*len(New_deaths_rate_US)

plt.xticks(np.arange(0,len(New_deaths_rate_US_v)))

plt.title("Ten Day Death Growth Rate (US)")

plt.plot(New_deaths_rate_US)

plt.plot(reported_onepointtwo,'r-',label = "Reported Average Growth Rate = 1.2")

plt.plot(one,'g',label = "Exponential Decline")

plt.plot(rolling_average, color='black', linestyle='dashed',label = "Rolling Average")



plt.show()

d=[0,-1,-2,-3,-4,-5,-6,-7,-8,-9]

New_deaths_rate_Italy=[]

for d in d:

  today = time_series_deaths_Italy.values[d-1]

  yesterday = time_series_deaths_Italy.values[d-2]

  New_deaths_rate_Italy.append(today/yesterday)

New_deaths_rate_Italy.reverse() 

New_deaths_rate_Italy_v = np.array(New_deaths_rate_Italy)

New_deaths_rate_Italy_v



# Observed growth rate based on 10 day average

Rolling_day_average_Italy = New_deaths_rate_Italy_v.sum()/len(New_deaths_rate_Italy)

print ("Observed growth rate (Italy) based on 10 day average = ", Rolling_day_average_Italy)

print("Today's growth rate (Italy) = ",New_deaths_rate_Italy[-1])

fig = plt.figure(figsize=(5,5))

reported_onepointtwo = [1.2]*len(New_deaths_rate_Italy)

one = [1]*len(New_deaths_rate_Italy)

rolling_average_Italy = [Rolling_day_average]*len(New_deaths_rate_Italy)

plt.xticks(np.arange(0,len(New_deaths_rate_Italy_v)))

plt.title("Ten Day Death Growth Rate (Italy)")

plt.plot(New_deaths_rate_Italy)

plt.plot(one,'g-',label = "Expontial Decline")

plt.plot(reported_onepointtwo,'r-',label="Reported Average Growth Rate 1.2")

plt.plot(rolling_average_Italy, color='black', linestyle='dashed',label = "Rolling Average")

plt.legend()

plt.show()
New_deaths_rate_US
### Projects the observed rolling day average rate of change a day with the 

### reported rate

reported_average = 1.2

projected_deaths = []

projected_death = time_series_deaths_US.iloc[:,-1].sum()*Rolling_day_average

projected_deaths_reported_average = []

projected_death_reported_average = time_series_deaths_US.iloc[:,-1].sum()*reported_average



d = 14 # days projected

for i in np.arange(0,d):

  projected_deaths.append(projected_death)

  projected_death = projected_death*Rolling_day_average



  projected_deaths_reported_average.append(projected_death_reported_average)

  projected_death_reported_average = projected_death_reported_average*reported_average



print("Projected Deaths using observed growth",projected_deaths[-1])



spacing = d%8

fig1 = plt.figure(figsize=[5,5])

plt.xticks(np.arange(0, len(projected_deaths),spacing))

plt.title("Projected US Deaths based on observed growth")

plt.plot(projected_deaths)

plt.show()



print("Projected Deaths using reported growth rate", projected_deaths_reported_average[-1])

fig2 = plt.figure(figsize=[5,5])

plt.xticks(np.arange(0,len(projected_deaths_reported_average),spacing))

plt.title("Projected US Deaths based on reported growth")

plt.plot(projected_deaths_reported_average)

plt.show()