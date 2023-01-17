import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from datetime import datetime

import ipywidgets as widgets

from IPython.display import display, clear_output

sns.set(style="whitegrid", color_codes=True)

np.random.seed(19900311)
def plot_bars(data):

    keys = list(data.keys())

    vals = list(data.values())

    stats_max = np.zeros(len(keys))

    stats_min = np.zeros(len(keys))

    stats_avg = np.zeros(len(keys))

    for i,v in enumerate(vals):

        stats_max[i] = np.max(v)

        stats_min[i] = np.min(v)

        stats_avg[i] = np.mean(v)

    plt.bar(keys, stats_max, 0.4)

    plt.bar(keys, stats_avg, 0.4)

    plt.bar(keys, stats_min, 0.4)
def plot_dists(data):

    for x,k in zip(data.values(), data.keys()):

        sns.distplot(x, hist=False, label=str(k))

        

def get_stats(visits, j, delta=28):

    t = visits.dates[j]

    weekday = []

    holiday = []

    workday = []

    dates = []

    values = []

    for i in range(7):

        weekday.append([])

    for i in range(0, j):

        t2 = visits.dates[i]

        v = visits.visitors[i]

        if (t-t2).days <= delta:

            values.append(v)

            dates.append(v)

            weekday[t2.weekday()].append(v)

            if holidays[t2]:

                holiday.append(v)

            else:

                workday.append(v)

    return weekday, holiday, workday, values, dates



def predict(visits, j, delta=28):

    days, hdays, wdays, vals, dats = get_stats(visits, j)

    date = visits.dates[j]

    weekday = date.weekday()

    is_holiday = holidays[date]

    day_mean = np.mean(days[weekday])

    day_trend = np.mean(np.array(days[weekday][1:])/np.array(days[weekday][:-1]))



    all_mean = np.mean(vals)

    if weekday > 0:

        week_trend = 0

        for i in range(weekday):

            week_trend += days[i][-1]/np.mean(days[i][:-1])

        week_trend /= weekday

    else: 

        week_trend = 1

    return day_trend * day_mean * week_trend



def measure(predicted, observed):

    return np.sqrt(np.mean((np.log(predicted+1)-np.log(observed+1))**2))

stores_id = pd.read_csv("../input/store_id_relation.csv")

air_visit = pd.read_csv("../input/air_visit_data.csv")

date_info = pd.read_csv("../input/date_info.csv")

tot_restos = len(stores_id.air_store_id)
holidays = {}

for (date, _, flag) in date_info.values:

    holidays[datetime.strptime(date, '%Y-%m-%d')] = bool(flag)
resto_num = np.random.randint(0, tot_restos)

resto_id = stores_id.air_store_id[resto_num]

print(f'selected: {resto_id} [{resto_num}]')
data = {'dates': [], 'visitors': []}

for (r_id, date, visit) in air_visit.values:

    if r_id == resto_id:

        data['dates'].append(datetime.strptime(date, '%Y-%m-%d'))

        data['visitors'].append(visit)

visits = pd.DataFrame(data=data)

plt.plot(visits.dates, visits.visitors)

plt.ylabel("N. Visitors")

plt.xlim([np.min(visits.dates), np.max(visits.dates)])

plt.show()

data = {

    'Holidays':[],

    'Working days':[],

    'Overall': visits.visitors

}



for date, visit in visits.values:

    if holidays[date]:

        data['Holidays'].append(visit)

    else:

        data['Working days'].append(visit)

plot_bars(data)

plt.ylabel("N. Visitors")

plt.figure()

plot_dists(data)

plt.xlabel('N. Visitors')



plt.show()
weekdays = {0:[],1:[],2:[],3:[],4:[],5:[],6:[]}



for date, visit in visits.values:

    weekdays[date.dayofweek].append(visit)

plot_bars(weekdays)

plt.xlabel("Week days")

plt.ylabel("N. Visitors")

plt.figure()

plot_dists(weekdays)

plt.xlabel('N. Visitors')

plt.show()  
tf = len(visits.dates)

t0 = 30 

predicted = np.zeros(tf-t0)

observed = np.array(visits.visitors[t0:])

delta = 28

for t in range(t0, tf):

    predicted[t-t0] = predict(visits, t, delta=delta)

plt.plot(predicted)

plt.plot(observed)

print(measure(predicted, observed))

plt.show()