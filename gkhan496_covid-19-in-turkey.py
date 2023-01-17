import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.basemap import Basemap
import folium
df = pd.read_csv("../input/covid19-in-turkey/covid_19_data_tr.csv")
df_confirmed = pd.read_csv("../input/covid19-in-turkey/time_series_covid_19_confirmed_tr.csv")
df_deaths = pd.read_csv("../input/covid19-in-turkey/time_series_covid_19_deaths_tr.csv")
df_recovered = pd.read_csv("../input/covid19-in-turkey/time_series_covid_19_recovered_tr.csv")
df_test_numbers = pd.read_csv("../input/covid19-in-turkey/test_numbers.csv")
df_intubated = pd.read_csv("../input/covid19-in-turkey/time_series_covid_19_intubated_tr.csv")
df_intensive_care = pd.read_csv("../input/covid19-in-turkey/time_series_covid_19_intensive_care_tr.csv")
df
df_confirmed
dates = df_confirmed.iloc[:, 4:].columns
confirmed_num = df_confirmed.iloc[:, 4:].values[0]

fig = figure(num=None, figsize=(24, 10), dpi=100, facecolor='w', edgecolor='white')

linewidth = 3.5
linewidth1 = 0.6

plt.plot(dates,confirmed_num,color='purple', alpha=1, linewidth = linewidth,  label = "CONFIRMED")
plt.scatter(dates,confirmed_num,color='white', alpha=1, linewidth = linewidth1)

ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor((0, 0, 0))

plt.xticks(rotation='vertical')
plt.yticks(np.arange(0,max(confirmed_num) + 500, 5000))
plt.legend(loc = 2)
plt.grid(color='white', linestyle="--", linewidth=0.3, dash_joinstyle = "bevel")
plt.title('Confirmed cases')
plt.show()
df_recovered
recovered_num = df_recovered.iloc[:, 4:].values[0]

fig = figure(num=None, figsize=(24, 10), dpi=100, facecolor='w', edgecolor='white')

plt.plot(dates,recovered_num,color='#dfff00', alpha=1, linewidth = linewidth, label = "RECOVERED")
plt.scatter(dates,recovered_num,color='white', alpha=1, linewidth = linewidth1)

ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor((0, 0, 0))


plt.xticks(rotation='vertical')
plt.yticks(np.arange(0,max(recovered_num) + 500, 4500))
plt.legend(loc = 2)
plt.grid(color='white', linestyle="--", linewidth=0.3, dash_joinstyle = "bevel")
plt.title('Recovered')
plt.show()
df_deaths
deaths_num = df_deaths.iloc[:, 4:].values[0]

fig = figure(num=None, figsize=(24, 10), dpi=100, facecolor='w', edgecolor='white')



plt.plot(dates,deaths_num,color='red', alpha=1, linewidth = linewidth, label = "DEATHS")
plt.scatter(dates,deaths_num,color='green', alpha=1, linewidth = linewidth1)

ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor((0, 0, 0))


plt.xticks(rotation='vertical')
plt.yticks(np.arange(0,max(deaths_num) + 50, 170))
plt.legend(loc = 2)
plt.grid(color='white', linestyle="--", linewidth=0.3, dash_joinstyle = "bevel")
plt.title('Deaths')
plt.show()
df_test_numbers
df_test_numbers_t = df_test_numbers.iloc[:, 4:].values[0]

fig = figure(num=None, figsize=(24, 10), dpi=100, facecolor='w', edgecolor='white')


plt.plot(dates,df_test_numbers_t,color='white', alpha=1, linewidth = linewidth, label = "TESTNUM")
plt.scatter(dates,df_test_numbers_t,color='white', alpha=1, linewidth = linewidth1)

ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor((0, 0, 0))


plt.xticks(rotation='vertical')
plt.legend(loc = 2)
plt.grid(color='white', linestyle="--", linewidth=0.3, dash_joinstyle = "bevel")
plt.title('TESTNUMBERS')
plt.show()
df_intubated
intubated = df_intubated.iloc[:, 4:].values[0]

fig = figure(num=None, figsize=(24, 10), dpi=100, facecolor='w', edgecolor='white')


plt.plot(dates,intubated,color='purple', alpha=1, linewidth = linewidth, label = "INTUBATED")
plt.scatter(dates,intubated,color='white', alpha=1, linewidth = linewidth1)

ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor((0, 0, 0))

plt.xticks(rotation='vertical')
plt.legend(loc = 2)
plt.grid(color='white', linestyle="--", linewidth=0.3, dash_joinstyle = "bevel")
plt.title('INTUBATED')
plt.show()
df_intensive_care
intensive_care = df_intensive_care.iloc[:, 4:].values[0]

fig = figure(num=None, figsize=(24, 10), dpi=100, facecolor='w', edgecolor='white')

linewidth = 3.5
linewidth1 = 0.6

plt.plot(dates,intensive_care,color='gray', alpha=1, linewidth = linewidth, label = "Intensive Care")
plt.scatter(dates,intensive_care,color='white', alpha=1, linewidth = linewidth1)

ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor((0, 0, 0))

plt.xticks(rotation='vertical')
plt.legend(loc = 2)
plt.grid(color='white', linestyle="--", linewidth=0.3, dash_joinstyle = "bevel")
plt.title('Intensive Care')
plt.show()
dates = df_confirmed.iloc[:, 4:].columns
confirmed_num = df_confirmed.iloc[:, 4:].values[0]
recovered_num = df_recovered.iloc[:, 4:].values[0]
deaths_num = df_deaths.iloc[:, 4:].values[0]
test_numbers = df_test_numbers.iloc[:, 4:].values[0]


fig = figure(num=None, figsize=(24, 10), dpi=100, facecolor='w', edgecolor='white')

linewidth = 3.5
linewidth1 = 0.6

plt.plot(dates,confirmed_num,color='purple', alpha=1, linewidth = linewidth, label = "CONFIRMED")
plt.scatter(dates,confirmed_num,color='white', alpha=1, linewidth = linewidth1)


plt.plot(dates,deaths_num,color='red', alpha=1, linewidth = linewidth, label = "DEATHS")
plt.scatter(dates,deaths_num,color='white', alpha=1, linewidth = linewidth1)


plt.plot(dates,recovered_num,color='#dfff00', alpha=1, linewidth = linewidth, label = "RECOVERED")
plt.scatter(dates,recovered_num,color='white', alpha=1, linewidth = linewidth1)

plt.plot(dates,test_numbers,color='white', alpha=1, linewidth = linewidth, label = "TESTNUM")
plt.scatter(dates,test_numbers,color='white', alpha=1, linewidth = linewidth1)

ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor((0, 0, 0))

plt.xticks(rotation='vertical')
plt.yticks(np.arange(0,max(confirmed_num) + 500, 5500))
plt.legend(loc = 2)
plt.grid(color='white', linestyle="--", linewidth=0.3, dash_joinstyle = "bevel")
plt.title('nCOV-2019 in Turkey')
plt.show()