import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os as os

import matplotlib.pyplot as plt
global_deaths = pd.read_csv("/kaggle/input/jhucovid19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")

# grouping by country at first (laziness)

global_deaths = global_deaths.groupby(["Country/Region"], as_index=False).sum()

# removing lat and long, the groupby summed the values...

global_deaths = global_deaths.drop(["Lat", "Long"], axis=1)



# transposing the columns to rows

global_deaths = global_deaths.transpose()

# making the top row as columns

new_header = global_deaths.iloc[0]

global_deaths.columns = new_header.str.lower()

# removing the first row from dataframe

global_deaths = global_deaths[1:]

# converting the main time series index to datetime

global_deaths.index = pd.to_datetime(global_deaths.index)

global_deaths.index = global_deaths.index.rename("date")
brazil_raw_deaths = pd.read_csv("/kaggle/input/corona-virus-brazil/brazil_covid19.csv")

brazilian_states = brazil_raw_deaths.state.unique()

brazil_deaths = pd.DataFrame()

i = 0

for state in brazilian_states:

    state_deaths = brazil_raw_deaths[brazil_raw_deaths.state==state]

    #grouping by date if any state has more than one death information by day

    state_deaths = state_deaths.groupby(["date"], as_index=False).max()

    state_deaths = state_deaths.set_index("date")

    state_deaths.index = pd.to_datetime(state_deaths.index)

    state_deaths = state_deaths[["deaths"]]

    state_deaths = state_deaths.rename(columns={"deaths": state.lower()})

    if i == 0:

        brazil_deaths = state_deaths

    else:

        brazil_deaths = brazil_deaths.join(state_deaths)

    i += 1
all_deaths = global_deaths.join(brazil_deaths)

all_deaths.tail()
min_deaths_count = 0 #configuring how many deaths to start the timeline

df_first_deaths = pd.DataFrame(columns=["place", "event_date"])



#creating a dataframe with the first death case for each place

for label in all_deaths.columns:

    event_date = all_deaths[all_deaths[label] > min_deaths_count].index.min()

    # print(label,first_case_date)

    df_first_deaths = df_first_deaths.append(

        {"place": label,

         "event_date": all_deaths[all_deaths[label] > min_deaths_count].index.min()},

        ignore_index=True

    )



    

#creating a dataframe where all places starts COVID-19 deaths at the same time    

all_deaths_timeless = pd.DataFrame(index=range(0, all_deaths.shape[0]))

all_deaths_timeless_item = pd.DataFrame()

all_deaths_timeless.index.values

column_names = []

for index, row in df_first_deaths.iterrows():

    all_deaths_timeless_item = all_deaths[[row["place"]]][all_deaths.index >= row["event_date"]]

    all_deaths_timeless_item.index = range(0, all_deaths_timeless_item.shape[0])

    all_deaths_timeless = pd.concat(

        [all_deaths_timeless, all_deaths_timeless_item],

        ignore_index=True, axis=1

    )

    column_names.append(row["place"])



all_deaths_timeless.columns = column_names

all_deaths_timeless.head()
places = ["Brazil", "Rio de Janeiro", "Italy", "São Paulo"]

days = 28

is_logarithmic = True

places = [x.lower() for x in places]
filtered_deaths = all_deaths_timeless[places]

filtered_deaths = filtered_deaths[filtered_deaths.index <= days]

if is_logarithmic:

    title = "COVID-19 Total Deaths (logarithmic)"

else:

    title = "COVID-19 Total Deaths (linear)"

    

filtered_deaths.plot(logy=is_logarithmic)
places = ["Brazil", "Rio de Janeiro", "Italy", "São Paulo", "Spain", "US"]

days = 25

is_logarithmic = True

places = [x.lower() for x in places]



filtered_deaths = all_deaths_timeless[places]

filtered_deaths = filtered_deaths[filtered_deaths.index <= days]

if is_logarithmic:

    title = "COVID-19 Total Deaths (logarithmic)"

else:

    title = "COVID-19 Total Deaths (linear)"

    

filtered_deaths.plot(logy=is_logarithmic)
all_deaths_timeless.plot(legend=False) #trust me with legend the chart xplodes