import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import datetime as dt

import matplotlib.dates as mdates



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
f1="../input/GlobalLandTemperaturesByCity.csv"

f2="../input/GlobalLandTemperaturesByCountry.csv"

f3="../input/GlobalLandTemperaturesByMajorCity.csv"

f4="../input/GlobalLandTemperaturesByState.csv"

f5="../input/GlobalTemperatures.csv"

byCity =pd.read_csv(f1,sep=",")

byCountry =pd.read_csv(f2,sep=",")

byMajorCity =pd.read_csv(f3,sep=",")

byState =pd.read_csv(f4,sep=",")

globalTemps =pd.read_csv(f5,sep=",")
byCity.info()

byCity.shape

byCity.head(3)
byCountry.info()

byCountry.shape

byCountry.head(3)
byMajorCity.info()

byMajorCity.shape

byMajorCity.head(3)
globalTemps.info()

globalTemps.shape

globalTemps.head(3)
x = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in byCountry['dt']]

y = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in byState['dt']]

z = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in byMajorCity['dt']]

print(type(x))

print(len(x))
plt.figure(figsize=(12,8))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))

plt.title("Year wise Country Average temprature ")

plt.plot(x,byCountry['AverageTemperature'],'red')
plt.figure(figsize=(12,8))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))

plt.title("Year wise State Average temprature ")

plt.plot(y,byState['AverageTemperature'],'blue')
plt.figure(figsize=(12,8))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))

plt.title("Year wise Major City Average temprature ")

plt.plot(z,byMajorCity['AverageTemperature'],'green')
globalTemps.hist(figsize=(12,12))
avgtemp_sum= byCountry.groupby('Country')['AverageTemperature'].sum()

avgtemp_sum_sort = avgtemp_sum.reset_index().sort_values(by='AverageTemperature' ,ascending=False).reset_index(drop=True)

avgtemp_sum_sort.head(5)
plt.figure(figsize=(10,10))

plt.title("Top 20 countries with highest AverageTemprature")

plt.pie(avgtemp_sum_sort['AverageTemperature'][0:20],labels=avgtemp_sum_sort['Country'][0:20])
avgtemp= byCountry.groupby('Country')['AverageTemperature'].mean()

avgtemp[0:5]
avgtemp_sort = avgtemp.reset_index().sort_values(by='AverageTemperature' ,ascending=False).reset_index(drop=True)

avgtemp_sort.head(5)
plt.figure(figsize=(10,10))

plt.title("Top 20 countries with highest average AverageTemprature")

plt.pie(avgtemp_sort['AverageTemperature'][0:20],labels=avgtemp_sort['Country'][0:20])