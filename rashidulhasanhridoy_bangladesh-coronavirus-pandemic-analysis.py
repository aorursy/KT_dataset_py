import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/coronavirus-covid19-bangladesh-dataset/covid19Bangladesh.csv')
df.shape
df.head()
df.tail()
df.columns
df.isnull().sum()
print('Latest Update of Covid-19 Pandemic of Bangladesh')

print('Date:', df['Date'].iloc[-1])

print('Newly tested:', df['New Tested'].iloc[-1])

print('Newly confirmed cases:', df['New Cases'].iloc[-1])

print('Newly Deaths:', df['New Deaths'].iloc[-1])

print('Newly Recovered:', df['Newly Recovered'].iloc[-1])

print('\nTotal tested:', df['Total Tested'].iloc[-1])

print('Total confirmed cases:', df['Total Cases'].iloc[-1])

print('Total Deaths:', df['Total Deaths'].iloc[-1])

print('Total Recovered:', df['Total Recovered'].iloc[-1])
j = 0

x = 0

confirmed_cases = list(df['New Cases'])

for i in confirmed_cases:

    x = x + i

    j = j + 1

    if x == 1000 or x > 1000:

        break

print('Number of confirmed cases reached 1k in', j, 'days.')

print('Days differences in 0 and 1k is', j)



m = 0

n = 0

confirmed_cases = list(df['New Cases'])

for l in confirmed_cases:

    m = m + l

    n = n + 1

    if m == 5000 or m > 5000:

        break

print('Number of confirmed cases reached 5k in', n, 'days.')

print('Days differences in 1K and 5k is', n-j)





a = 0

b = 0

confirmed_cases = list(df['New Cases'])

for k in confirmed_cases:

    a = a + k

    b = b + 1

    if a == 10000 or a > 10000:

        break

print('Number of confirmed cases reached 10k in', b, 'days.')

print('Days differences in 5K and 10k is', b-n)



t = 0

v = 0

confirmed_cases = list(df['New Cases'])

for s in confirmed_cases:

    t = t + s

    v = v + 1

    if t == 15000 or t > 15000:

        break

print('Number of confirmed cases reached 15k in', v, 'days.')

print('Days differences in 10K and 15k is', v-b)
confirmed_cases = np.nan_to_num(df['New Cases'])

f = plt.figure(figsize=(15,8))

ax = f.add_subplot(111)

day = df['Day']

marker_style = dict(linewidth=2, linestyle='-', marker='o',markersize=5)

plt.plot(day,confirmed_cases,"-.",color="blue",**marker_style)

ax.tick_params(which='both', width=1,labelsize=14)

ax.tick_params(which='major', length=6)

ax.tick_params(which='minor', length=3, color='0.8')

plt.grid(lw = 1, ls = '-', c = "0.85", which = 'major')

plt.grid(lw = 1, ls = '-', c = "0.95", which = 'minor')

plt.title("COVID-19 Bangladesh Daily Confirmed Cases",{'fontsize':22})

plt.xlabel("Day",fontsize =18)

plt.ylabel("Number of Daily Confirmed Cases",fontsize =18)

plt.legend(['Daily Confirmed Cases'])

plt.show()
confirmed_cases = np.nan_to_num(df['Total Cases'])

f = plt.figure(figsize=(15,8))

ax = f.add_subplot(111)

day = df['Day']

marker_style = dict(linewidth=2, linestyle='-', marker='o',markersize=5)

plt.plot(day,confirmed_cases,"-.",color="blue",**marker_style)

ax.tick_params(which='both', width=1,labelsize=14)

ax.tick_params(which='major', length=6)

ax.tick_params(which='minor', length=3, color='0.8')

plt.grid(lw = 1, ls = '-', c = "0.85", which = 'major')

plt.grid(lw = 1, ls = '-', c = "0.95", which = 'minor')

plt.title("COVID-19 Bangladesh Total Confirmed Cases",{'fontsize':22})

plt.xlabel("Day",fontsize =18)

plt.ylabel("Number of total Confirmed Cases",fontsize =18)

plt.legend(['Total Confirmed Cases'])

plt.show()
df['Confirmed Case Rate'] = (df['Total Cases'] / df['Total Tested']) * 100

df.head()
df.tail()
confirmed_cases = np.nan_to_num(df['Confirmed Case Rate'])

f = plt.figure(figsize=(15,8))

ax = f.add_subplot(111)

day = df['Day']

marker_style = dict(linewidth=2, linestyle='-', marker='o',markersize=5)

plt.plot(day,confirmed_cases,"-.",color="blue",**marker_style)

ax.tick_params(which='both', width=1,labelsize=14)

ax.tick_params(which='major', length=6)

ax.tick_params(which='minor', length=3, color='0.8')

plt.grid(lw = 1, ls = '-', c = "0.85", which = 'major')

plt.grid(lw = 1, ls = '-', c = "0.95", which = 'minor')

plt.title("COVID-19 Bangladesh Confirmed Cases Rate",{'fontsize':22})

plt.xlabel("Day",fontsize =18)

plt.ylabel("Confirmed Cases Rate",fontsize =18)

plt.legend(['Confirmed Cases Rate'])

plt.show()
confirmed_cases = np.nan_to_num(df['New Deaths'])

f = plt.figure(figsize=(15,8))

ax = f.add_subplot(111)

day = df['Day']

marker_style = dict(linewidth=2, linestyle='-', marker='o',markersize=5)

plt.plot(day,confirmed_cases,"-.",color="red",**marker_style)

ax.tick_params(which='both', width=1,labelsize=14)

ax.tick_params(which='major', length=6)

ax.tick_params(which='minor', length=3, color='0.8')

plt.grid(lw = 1, ls = '-', c = "0.85", which = 'major')

plt.grid(lw = 1, ls = '-', c = "0.95", which = 'minor')

plt.title("COVID-19 Bangladesh Daily Death Cases",{'fontsize':22})

plt.xlabel("Day",fontsize =18)

plt.ylabel("Number of Daily Death Cases",fontsize =18)

plt.legend(['Daily Death Cases'])

plt.show()
confirmed_cases = np.nan_to_num(df['Total Deaths'])

f = plt.figure(figsize=(15,8))

ax = f.add_subplot(111)

day = df['Day']

marker_style = dict(linewidth=2, linestyle='-', marker='o',markersize=5)

plt.plot(day,confirmed_cases,"-.",color="red",**marker_style)

ax.tick_params(which='both', width=1,labelsize=14)

ax.tick_params(which='major', length=6)

ax.tick_params(which='minor', length=3, color='0.8')

plt.grid(lw = 1, ls = '-', c = "0.85", which = 'major')

plt.grid(lw = 1, ls = '-', c = "0.95", which = 'minor')

plt.title("COVID-19 Bangladesh Total Death Cases",{'fontsize':22})

plt.xlabel("Day",fontsize =18)

plt.ylabel("Number of Total Death Cases",fontsize =18)

plt.legend(['Total Death Cases'])

plt.show()
df['Death Case Rate'] = (df['Total Deaths'] / df['Total Cases']) * 100

df.head()
df.isnull().sum()
confirmed_cases = np.nan_to_num(df['Death Case Rate'])

f = plt.figure(figsize=(15,8))

ax = f.add_subplot(111)

day = df['Day']

marker_style = dict(linewidth=2, linestyle='-', marker='o',markersize=5)

plt.plot(day,confirmed_cases,"-.",color="red",**marker_style)

ax.tick_params(which='both', width=1,labelsize=14)

ax.tick_params(which='major', length=6)

ax.tick_params(which='minor', length=3, color='0.8')

plt.grid(lw = 1, ls = '-', c = "0.85", which = 'major')

plt.grid(lw = 1, ls = '-', c = "0.95", which = 'minor')

plt.title("COVID-19 Bangladesh Death Cases Rate",{'fontsize':22})

plt.xlabel("Day",fontsize =18)

plt.ylabel("Death Cases Rate",fontsize =18)

plt.legend(['Death Cases Rate'])

plt.show()
confirmed_cases = np.nan_to_num(df['Newly Recovered'])

f = plt.figure(figsize=(15,8))

ax = f.add_subplot(111)

day = df['Day']

marker_style = dict(linewidth=2, linestyle='-', marker='o',markersize=5)

plt.plot(day,confirmed_cases,"-.",color="green",**marker_style)

ax.tick_params(which='both', width=1,labelsize=14)

ax.tick_params(which='major', length=6)

ax.tick_params(which='minor', length=3, color='0.8')

plt.grid(lw = 1, ls = '-', c = "0.85", which = 'major')

plt.grid(lw = 1, ls = '-', c = "0.95", which = 'minor')

plt.title("COVID-19 Bangladesh Daily Recovered Cases",{'fontsize':22})

plt.xlabel("Day",fontsize =18)

plt.ylabel("Daily Recovered Cases",fontsize =18)

plt.legend(['Daily Recovered Cases'])

plt.show()
confirmed_cases = np.nan_to_num(df['Total Recovered'])

f = plt.figure(figsize=(15,8))

ax = f.add_subplot(111)

day = df['Day']

marker_style = dict(linewidth=2, linestyle='-', marker='o',markersize=5)

plt.plot(day,confirmed_cases,"-.",color="green",**marker_style)

ax.tick_params(which='both', width=1,labelsize=14)

ax.tick_params(which='major', length=6)

ax.tick_params(which='minor', length=3, color='0.8')

plt.grid(lw = 1, ls = '-', c = "0.85", which = 'major')

plt.grid(lw = 1, ls = '-', c = "0.95", which = 'minor')

plt.title("COVID-19 Bangladesh Total Recovered Cases",{'fontsize':22})

plt.xlabel("Day",fontsize =18)

plt.ylabel("Total Recovered Cases",fontsize =18)

plt.legend(['Total Recovered Cases'])

plt.show()
df['Recovered Case Rate'] = (df['Total Recovered'] / df['Total Cases']) * 100

df.head()
df.isnull().sum()
confirmed_cases = np.nan_to_num(df['Recovered Case Rate'])

f = plt.figure(figsize=(15,8))

ax = f.add_subplot(111)

day = df['Day']

marker_style = dict(linewidth=2, linestyle='-', marker='o',markersize=5)

plt.plot(day,confirmed_cases,"-.",color="green",**marker_style)

ax.tick_params(which='both', width=1,labelsize=14)

ax.tick_params(which='major', length=6)

ax.tick_params(which='minor', length=3, color='0.8')

plt.grid(lw = 1, ls = '-', c = "0.85", which = 'major')

plt.grid(lw = 1, ls = '-', c = "0.95", which = 'minor')

plt.title("COVID-19 Bangladesh Recovered Cases Rate",{'fontsize':22})

plt.xlabel("Day",fontsize =18)

plt.ylabel("Recovered Cases Rate",fontsize =18)

plt.legend(['Recovered Cases Rate'])

plt.show()
df['Active Cases'] = (df['Total Cases'] - df['Total Deaths'] + df['Total Recovered'])

df.head()
cases = np.nan_to_num(df['Active Cases'])

f = plt.figure(figsize=(15,8))

ax = f.add_subplot(111)

day = df['Day']

marker_style = dict(linewidth=2, linestyle='-', marker='o',markersize=5)

plt.plot(day,cases,"-.",color="orange",**marker_style)

ax.tick_params(which='both', width=1,labelsize=14)

ax.tick_params(which='major', length=6)

ax.tick_params(which='minor', length=3, color='0.8')

plt.grid(lw = 1, ls = '-', c = "0.85", which = 'major')

plt.grid(lw = 1, ls = '-', c = "0.95", which = 'minor')

plt.title("COVID-19 Bangladesh Active Cases",{'fontsize':22})

plt.xlabel("Day",fontsize =18)

plt.ylabel("Active Cases",fontsize =18)

plt.legend(['Active Cases'])

plt.show()
confirmed_cases = np.nan_to_num(df['New Cases'])

new_deaths = np.nan_to_num(df['New Deaths'])

new_recovered = np.nan_to_num(df['Newly Recovered'])

f = plt.figure(figsize=(15,8))

ax = f.add_subplot(111)

day = df['Day']

marker_style = dict(linewidth=2, linestyle='-', marker='o',markersize=5)

plt.plot(day,confirmed_cases,"-.",color="blue",**marker_style)

plt.plot(day,new_deaths,"-.",color="red",**marker_style)

plt.plot(day,new_recovered,"-.",color="green",**marker_style)

ax.tick_params(which='both', width=1,labelsize=14)

ax.tick_params(which='major', length=6)

ax.tick_params(which='minor', length=3, color='0.8')

plt.grid(lw = 1, ls = '-', c = "0.85", which = 'major')

plt.grid(lw = 1, ls = '-', c = "0.95", which = 'minor')

plt.title("COVID-19 Bangladesh Daily Confirmed, Deaths, Recovered Cases",{'fontsize':22})

plt.xlabel("Day",fontsize =18)

plt.ylabel("Cases",fontsize =18)

plt.legend(['Confirmed Cases', 'Deaths', 'Recovered Cases'])

plt.show()
confirmed_cases = np.nan_to_num(df['Confirmed Case Rate'])

new_deaths = np.nan_to_num(df['Death Case Rate'])

new_recovered = np.nan_to_num(df['Recovered Case Rate'])

f = plt.figure(figsize=(15,7.5))

ax = f.add_subplot(111)

day = df['Day']

marker_style = dict(linewidth=2, linestyle='-', marker='o',markersize=5)

plt.plot(day,confirmed_cases,"-.",color="blue",**marker_style)

plt.plot(day,new_deaths,"-.",color="red",**marker_style)

plt.plot(day,new_recovered,"-.",color="green",**marker_style)

ax.tick_params(which='both', width=1,labelsize=14)

ax.tick_params(which='major', length=6)

ax.tick_params(which='minor', length=3, color='0.8')

plt.grid(lw = 1, ls = '-', c = "0.85", which = 'major')

plt.grid(lw = 1, ls = '-', c = "0.95", which = 'minor')

plt.title("COVID-19 Bangladesh Daily Confirmed, Deaths, Recovered Cases Rate",{'fontsize':22})

plt.xlabel("Day",fontsize =18)

plt.ylabel("Cases Rate",fontsize =18)

plt.legend(['Confirmed Cases Rate', 'Deaths Cases Rate', 'Recovered Cases Rate'])

plt.show()