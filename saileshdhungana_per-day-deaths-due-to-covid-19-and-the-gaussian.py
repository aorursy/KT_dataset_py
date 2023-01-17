!wget https://covid.ourworldindata.org/data/ecdc/total_deaths.csv
import pandas as pd

df = pd.read_csv('total_deaths.csv', parse_dates=True, index_col=0)
df = df.fillna(0)

countries = ['World', 'China', 'Italy', 'United States']
df[countries].tail()

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

fig, axs = plt.subplots(2,2, figsize=(40,40))
    
for i in range(2):
    for j in range(2):
        country = countries[i*2+j]
        axs[i, j].plot(df.index, df[country])
        axs[i, j].set_title(country, fontsize=90)
df_each_day = df.diff()
fig, axs = plt.subplots(2,2, figsize=(40,40))
    
for i in range(2):
    for j in range(2):
        country = countries[i*2+j]
        axs[i, j].plot(df_each_day.index, df_each_day[country])
        axs[i, j].set_title(country, fontsize=90)
plt.plot(df_each_day.index, df_each_day.China)
from datetime import datetime

date_feb_13 = datetime.strptime('2020-02-13','%Y-%m-%d')
date_feb_14 = datetime.strptime('2020-02-14','%Y-%m-%d')

df_each_day_days = df_each_day.reset_index()
df_each_day_days = df_each_day_days.fillna(0)
df_each_day_days = df_each_day_days[df_each_day_days.date!=date_feb_13]
df_each_day_days = df_each_day_days[df_each_day_days.date!=date_feb_14]

plt.plot(df_each_day_days.index, df_each_day_days['China'])
import pylab as plb
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import numpy as np

x = df_each_day_days.index
y = df_each_day_days['China']

mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])
plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title("Fig. 3 - Fit for China's COVID-19 deaths")
plt.xlabel('Days')
plt.ylabel('Deaths')
plt.plot(x[int(popt[1])], Gauss(int(popt[1]), *popt), 'ro')
plt.plot(x[11], Gauss(x, *popt)[11], 'bo')
plt.show()
print(df_each_day_days['date'][int(popt[1])])
print(popt)
print(pcov)

countries = ['China', 'United States', 'Italy',
             'Spain', 'France', 'Iran',
             'South Korea', 'Japan', 'Germany']
n = len(countries)
df_each_day_days = df_each_day.reset_index()
df_each_day_days = df_each_day_days.fillna(0)

fig, axs = plt.subplots(3, 3, figsize=(90, 90))

from datetime import timedelta

def get_date(n):
    return datetime.strftime(datetime.strptime('2019-12-31','%Y-%m-%d') + timedelta(days=n), '%Y-%m-%d')

result = []
for i in range(3):
    for j in range(3):
        if i * 3 + j < n:
            country = countries[i * 3 + j]
            x = df_each_day_days.index
            y = df_each_day_days[country]
            last = y.tail(1).values[0]
            axs[i,j].plot(x, y, 'b+:', label='data')
            axs[i,j].set_title(country, fontsize=90)
            try:
                mean = sum(x * y) / sum(y)
                sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

                popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])
                axs[i,j].plot(x, Gauss(x, *popt), 'r-', label='fit')
                axs[i,j].text(0,0,'Peak Deaths: %s\nPeak Date: %s' %(popt[0], get_date(int(popt[1]))), fontsize=40)
                axs[i,j].tick_params(axis='x', labelsize=20)
                axs[i,j].tick_params(axis='y', labelsize=20)
                result.append([country, last, popt[0], popt[2], get_date(int(popt[1]))])
            except:
                pass
df_table = pd.DataFrame(result, columns=['country','per day deaths(latest)', 'per day deaths at Gaussian peak', 'Gaussian standard deviation', 'Gaussian peak date' ])
df_table
try:
    country = input()
except:
    country = "United States"
df_each_day_days = df_each_day.reset_index()
df_each_day_days = df_each_day_days.fillna(0)

result = []
x = df_each_day_days.index
y = df_each_day_days[country]
last = y.tail(1).values[0]
plt.plot(x, y, 'b+:', label='data')
plt.title(country, fontsize=90)
try:
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

    popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])
    plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
    result.append([country, last, popt[0], popt[2], get_date(int(popt[1]))])
except:
    pass
df_country = pd.DataFrame(result, columns=['country','per day deaths(latest)', 'per day deaths at Gaussian peak', 'Gaussian standard deviation', 'Gaussian peak date' ])
df_country
            