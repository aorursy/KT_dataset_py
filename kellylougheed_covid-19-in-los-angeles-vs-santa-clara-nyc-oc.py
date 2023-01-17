import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv("../input/us-counties-covid-19-dataset/us-counties.csv")
data.head()
data.tail()
los_angeles = data.loc[data["county"] == "Los Angeles"]
los_angeles = los_angeles.loc[los_angeles["date"] > "2020-03-13"]
los_angeles.head()
import matplotlib.pyplot as plt

cases = los_angeles["cases"]
dates = los_angeles["date"]
y_pos = np.arange(len(dates))

plt.figure(figsize=(20,10))

clrs = ['orange' if (x < max(cases)-5000) else 'red' for x in cases ]
 
# Create bars
plt.bar(y_pos, cases, color=clrs)
 
# Create names on the x-axis
plt.xticks(y_pos, dates, rotation=90)
 
# Show graphic
plt.show()
# Starter code: https://python-graph-gallery.com/8-add-confidence-interval-on-barplot/

barWidth = 0.4

bars1 = los_angeles["cases"]
bars2 = los_angeles["deaths"]
 
# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

plt.figure(figsize=(20,10))
 
# Create blue bars
plt.bar(r1, bars1, width = barWidth, color = 'pink')
 
# Create cyan bars
plt.bar(r2, bars2, width = barWidth, color = 'red')

# https://stackoverflow.com/questions/57340415/matplotlib-bar-plot-add-legend-from-categories-dataframe-column
colors = {'Cases':'pink', 'Deaths':'red'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
 
# general layout
plt.xticks([r + barWidth for r in range(len(bars1))], los_angeles["date"], rotation=90)
plt.ylabel('Cases')
plt.title("COVID-19 Cases & Deaths in Los Angeles")
 
# Show graphic
plt.show()
import matplotlib.pyplot as plt

daily_cases = []
prev_value = 0
for index, value in los_angeles["cases"].items():
    daily_cases.append(value - prev_value)
    prev_value = value

dates = los_angeles["date"]
y_pos = np.arange(len(dates))

plt.figure(figsize=(20,10))

clrs = ['orange' if (x < max(daily_cases)-250) else 'red' for x in daily_cases ]
 
# Create bars
plt.bar(y_pos, daily_cases, color=clrs)
 
# Create names on the x-axis
plt.xticks(y_pos, dates, rotation=90)
 
# Show graphic
plt.show()
santa_clara = data.loc[data["county"] == "Santa Clara"]
santa_clara = santa_clara.loc[santa_clara["date"] > "2020-03-13"]
santa_clara.head()
barWidth = 0.4

# Normalize by population
# Population numbers from https://www.california-demographics.com/counties_by_population
bars1 = los_angeles["cases"].apply(lambda x: (x / 10098052) * 10000)
bars2 = santa_clara["cases"].apply(lambda x: (x / 1922200 ) * 10000)
 
# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

plt.figure(figsize=(20,10))
 
# Los Angeles bars
plt.bar(r1, bars1, width = barWidth, color = 'cyan')
 
# Santa Clara bars
plt.bar(r2, bars2, width = barWidth, color = 'blue')

colors = {'Los Angeles':'cyan', 'Santa Clara':'blue'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
 
# general layout
plt.xticks([r + barWidth for r in range(len(bars1))], los_angeles["date"], rotation=90)
plt.ylabel('Cases per 10,000 people')
plt.title("COVID-19 Cases in Los Angeles County vs. Santa Clara County")
 
# Show graphic
plt.show()
barWidth = 0.4

def calc_daily_cases(col):
    prev_value = 0
    daily_cases = []
    for index, value in col.items():
        daily_cases.append(value - prev_value)
        prev_value = value
    return daily_cases

la_daily_cases = calc_daily_cases(los_angeles["cases"])
sc_daily_cases = calc_daily_cases(santa_clara["cases"])

# Normalize by population
# Population numbers from https://www.california-demographics.com/counties_by_population
bars1 = [(x / 10098052) * 10000 for x in la_daily_cases]
bars2 = [(x / 1922200 ) * 10000 for x in sc_daily_cases]
 
# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

plt.figure(figsize=(20,10))
 
# Los Angeles bars
plt.bar(r1, bars1, width = barWidth, color = 'cyan')
 
# Santa Clara bars
plt.bar(r2, bars2, width = barWidth, color = 'blue')

colors = {'Los Angeles':'cyan', 'Santa Clara':'blue'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
 
# general layout
plt.xticks([r + barWidth for r in range(len(bars1))], los_angeles["date"], rotation=90)
plt.ylabel('Cases per 10,000 people')
plt.title("Daily Cases of COVID-19 per 10,000 People in Los Angeles County vs. Santa Clara County")
 
# Show graphic
plt.show()
barWidth = 0.4

la_daily_cases = calc_daily_cases(los_angeles["deaths"])
sc_daily_cases = calc_daily_cases(santa_clara["deaths"])

# Normalize by population
# Population numbers from https://www.california-demographics.com/counties_by_population
bars1 = [(x / 10098052) * 10000 for x in la_daily_cases]
bars2 = [(x / 1922200 ) * 10000 for x in sc_daily_cases]
 
# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

plt.figure(figsize=(20,10))
 
# Los Angeles bars
plt.bar(r1, bars1, width = barWidth, color = 'cyan')
 
# Santa Clara bars
plt.bar(r2, bars2, width = barWidth, color = 'blue')

colors = {'Los Angeles':'cyan', 'Santa Clara':'blue'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
 
# general layout
plt.xticks([r + barWidth for r in range(len(bars1))], los_angeles["date"], rotation=90)
plt.ylabel('Deaths per 10,000 people')
plt.title("Daily Deaths from COVID-19 per 10,000 People in Los Angeles County vs. Santa Clara County")
 
# Show graphic
plt.show()
nyc = data.loc[data["county"] == "New York City"]
nyc = nyc.loc[nyc["date"] > "2020-03-13"]
nyc.head()
barWidth = 0.4
 
# Normalize by population
bars1 = los_angeles["cases"].apply(lambda x: (x / 10098052) * 10000)
bars2 = nyc["cases"].apply(lambda x: (x / 8399000) * 10000)
 
# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

plt.figure(figsize=(20,10))
 
# Los Angeles bars
plt.bar(r1, bars1, width = barWidth, color = 'cyan')
 
# NYC bars
plt.bar(r2, bars2, width = barWidth, color = 'magenta')

colors = {'New York': 'magenta', 'Los Angeles':'cyan'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
 
# general layout
plt.xticks([r + barWidth for r in range(len(bars1))], los_angeles["date"], rotation=90)
plt.ylabel('Cases per 10,000 people')
plt.title("COVID-19 in Los Angeles vs. NYC")
 
# Show graphic
plt.show()
barWidth = 0.4

la_daily_cases = calc_daily_cases(los_angeles["cases"])
nyc_daily_cases = calc_daily_cases(nyc["cases"])

# Normalize by population
# Population numbers from https://www.california-demographics.com/counties_by_population
bars1 = [(x / 10098052) * 10000 for x in la_daily_cases]
bars2 = [(x / 8399000) * 10000 for x in nyc_daily_cases]
 
# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

plt.figure(figsize=(20,10))
 
# Los Angeles bars
plt.bar(r1, bars1, width = barWidth, color = 'cyan')
 
# NYC bars
plt.bar(r2, bars2, width = barWidth, color = 'magenta')

colors = {'New York': 'magenta', 'Los Angeles':'cyan'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
 
# general layout
plt.xticks([r + barWidth for r in range(len(bars1))], los_angeles["date"], rotation=90)
plt.ylabel('Cases per 10,000 people')
plt.title("Daily Cases of COVID-19 per 10,000 People in Los Angeles vs. NYC")
 
# Show graphic
plt.show()
orange = data.loc[data["county"] == "Orange"]
orange = orange.loc[orange["state"] == "California"]
orange = orange.loc[orange["date"] > "2020-03-13"]
orange.head()
barWidth = 0.4

# Normalize by population
bars1 = los_angeles["cases"].apply(lambda x: (x / 10098052) * 10000)
bars2 = orange["cases"].apply(lambda x: (x / 3164182 ) * 10000)
 
# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

plt.figure(figsize=(20,10))
 
# Los Angeles bars
plt.bar(r1, bars1, width = barWidth, color = 'cyan')
 
# OC bars
plt.bar(r2, bars2, width = barWidth, color = 'orange')

colors = {'Los Angeles':'cyan', 'Orange County':'orange'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
 
# general layout
plt.xticks([r + barWidth for r in range(len(bars1))], los_angeles["date"], rotation=90)
plt.ylabel('Cases per 10,000 people')
plt.title("COVID-19 in Los Angeles vs. Orange County")
 
# Show graphic
plt.show()
san_mateo = data.loc[data["county"] == "San Mateo"]
san_mateo = san_mateo.loc[san_mateo["date"] > "2020-03-13"]
san_mateo.head()
barWidth = 0.4
 
bars1 = los_angeles["cases"].apply(lambda x: (x / 10098052) * 10000)
bars2 = san_mateo["cases"].apply(lambda x: (x / 765935) * 10000)
 
# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

plt.figure(figsize=(20,10))
 
# Los Angeles bars
plt.bar(r1, bars1, width = barWidth, color = 'cyan')
 
# San Mateo bars
plt.bar(r2, bars2, width = barWidth, color = 'blue')

colors = {'Los Angeles':'cyan', 'Santa Mateo':'blue'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
 
# general layout
plt.xticks([r + barWidth for r in range(len(bars1))], san_mateo["date"], rotation=90)
plt.ylabel('Cases per 10,000 people')
plt.title("COVID-19 in Los Angeles vs. San Mateo County")
 
# Show graphic
plt.show()