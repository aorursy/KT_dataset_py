import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime



def to_day(date):

    return date.item().days

def start_of_year(date):

    return to_day(date-jan1)



df = pd.read_csv("../input/alberta-coronavirus-cases/alberta_covid.csv", parse_dates=True).drop("Unnamed: 0", axis=1)

today, jan1 = datetime.date.today(), np.datetime64("2020-01-01")

curr_month, curr_year = today.month, today.year

mask_start_date = start_of_year(np.datetime64("2020-08-01"))

ahsz, date, all_zones = "Alberta Health Services Zone", "Date reported", {}    #column names

uniques = df[ahsz].unique().tolist()

#--- build months array ---

first_allmonths = [(np.datetime64(f"2020-0{i}-01")-jan1).item().days for i in range(1, 10) if i <= curr_month]

first_allmonths += [(np.datetime64(f"2020-{i}-01")-jan1).item().days for i in range(10, 13) if i <= curr_month]

if curr_year > 2020:

    first_allmonths.append((np.datetime64(f"2021-01-01")-jan1).item().days)
numrows = np.ceil(len(uniques)/2)

plt.figure(figsize=(15, numrows*4))

for z, k in zip(range(len(uniques)), range(1, len(uniques)*2)):

    zone = uniques[z]

    df_zone = df[df[ahsz]==zone]

    cases_by_date = df_zone[date].value_counts()    #equivalent to new cases per day when using date index

    index = cases_by_date.index.to_numpy(dtype='datetime64')

    data = cases_by_date.to_numpy()

    firstday = np.min(index)                        #get first case date

    index = index-firstday                               #calculate number of days since first case for each item

    compiled = [0] * to_day(np.max(index)+1)        #initialize array to prevent index error

    for i in range(len(index)):

        compiled[to_day(index[i])] = data[i]        #map number of cases to number of days since start of year

    year_to_first = [0] * start_of_year(firstday)   #pad out beginning of year with 0 cases

    compiled = year_to_first + compiled

    all_zones[zone] = compiled

    

    plt.xticks(first_allmonths)

    plt.subplot(numrows, 2, k)

    plt.plot(compiled)

    plt.title(zone)

plt.show()
plt.figure(figsize=(15, 4))

plt.legend(labels=uniques)

plt.title("All Zones")

plt.axvline(x=mask_start_date, color="red", linestyle="dashed")

for zone in all_zones:

    plt.plot(all_zones[zone])
plt.figure(figsize=(14, 4))

for k, zone in enumerate(["Calgary Zone", "Edmonton Zone"], 1):

    plt.subplot(1, 2, k)

    plt.axvline(x=30, color="red", linestyle="dashed")

    plt.axvline(x=44, color="red", linestyle="dashed")

    compiled = all_zones[zone]

    plt.plot(compiled[mask_start_date-30:])

    plt.title(zone)
plt.figure(figsize=(14, 4))

delay = 7

for k, zone in enumerate(["Calgary Zone", "Edmonton Zone"], 1):

    plt.subplot(1, 2, k)

    compiled = all_zones[zone]

    plt.axvline(x=mask_start_date, color="red", linestyle="dashed")

    plt.axvline(x=mask_start_date+14, color="red", linestyle="dashed")

    x = pd.DataFrame(compiled).rolling(delay).mean().shift(-delay//2)

    plt.plot(compiled, 'c')

    plt.plot(x, 'r')

    plt.title(zone)