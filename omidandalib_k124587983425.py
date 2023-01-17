# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly

import plotly.express as px

import plotly.graph_objs as go

import plotly.offline as py

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from datetime import datetime



py.init_notebook_mode(connected=True)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

pd.set_option("display.max_rows", None, "display.max_columns", None)
korea_cases_unfiltered_df = pd.read_csv('/kaggle/input/coronavirusdataset/Case.csv')

korea_cases_unfiltered_df.head()

# Remove results where tests are negative

korea_cases_df = korea_cases_unfiltered_df.loc[korea_cases_unfiltered_df['group'] == True]

korea_cases_df.head()
# Cases in Korea by location

korea_locations = korea_cases_df.drop(columns=[' case_id', 'city', 'group', 'latitude', 'longitude'])

korea_locations = korea_locations.groupby('infection_case', as_index=False).sum()



korea_locations = korea_locations.nlargest(10, 'confirmed')

korea_locations
cases_in_korea_by_location = px.pie(korea_locations, values='confirmed', names='infection_case')

cases_in_korea_by_location.update_layout(

    title = "Top 10 possible infection site categories in Korea",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )   

    

py.iplot(cases_in_korea_by_location)
korea_provinces = korea_cases_df.drop(columns=[' case_id', 'city', 'group', 'latitude', 'longitude'])

korea_provinces = korea_provinces.groupby('province', as_index=False).sum()

korea_provinces = korea_provinces.sort_values(by='confirmed', ascending=False)



korea_provinces = korea_provinces.nlargest(10, 'confirmed')

korea_provinces
cases_in_korea_by_province = px.pie(korea_provinces, values='confirmed', names='province')

cases_in_korea_by_province.update_layout(

    title = "Top 10 infection provinces in Korea",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )   

    

py.iplot(cases_in_korea_by_province)
# manually cleaned up

korea_policies = pd.read_csv('/kaggle/input/korea-covid19/korea_policies.csv')

korea_policies = korea_policies.drop(columns=['policy_id', 'type', 'gov_policy'])

korea_policies.start_date = pd.to_datetime(korea_policies.start_date, format='%Y-%m-%d')



# I USED THIS RESOURCE --> https://matplotlib.org/3.2.1/gallery/lines_bars_and_markers/timeline.html



korea_dates = korea_policies.start_date

korea_detail = korea_policies.detail

korea_detail = korea_detail.to_numpy()
levels = np.tile([-15, 15, -10, 10, -5, 5, 20, -20],

                 int(np.ceil(len(korea_dates)/8)))[:len(korea_dates)]



# Create figure and plot a stem plot with the date

fig, ax = plt.subplots(figsize=(25, 12), constrained_layout=True)

ax.set(title="Korea COVID19 prevention steps")



markerline, stemline, baseline = ax.stem(korea_dates, levels,

                                         linefmt="C3-", basefmt="k-",

                                         use_line_collection=True)



plt.setp(markerline, mec="k", mfc="w", zorder=3)



# Shift the markers to the baseline by replacing the y-data by zeros.

markerline.set_ydata(np.zeros(len(korea_dates)))



# annotate lines

vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]

for d, l, r, va in zip(korea_dates, levels, korea_detail, vert):

    ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*3),

                textcoords="offset points", va=va, ha="right")



# format xaxis with 1 month intervals

ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=1))

ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))

plt.setp(ax.get_xticklabels(), rotation=30, ha="right")



# remove y axis and spines

ax.get_yaxis().set_visible(False)

for spine in ["left", "top", "right"]:

    ax.spines[spine].set_visible(False)



ax.margins(y=0.1)

plt.show()
# manually sourced data from news articles

canada_policies = pd.read_csv('/kaggle/input/canada-covid19-aug8/canada_policies.csv')

canada_policies.start_date = pd.to_datetime(canada_policies.start_date, format='%Y-%m-%d')



# I USED THIS RESOURCE --> https://matplotlib.org/3.2.1/gallery/lines_bars_and_markers/timeline.html



canada_dates = canada_policies.start_date

canada_detail = canada_policies.detail

canada_detail = canada_detail.to_numpy()
levels = np.tile([-15, 15, -10, 10, -5, 5, 20, -20],

                 int(np.ceil(len(canada_dates)/8)))[:len(canada_dates)]



# Create figure and plot a stem plot with the date

fig, ax = plt.subplots(figsize=(25, 12), constrained_layout=True)

ax.set(title="Canada COVID19 prevention steps")



markerline, stemline, baseline = ax.stem(canada_dates, levels,

                                         linefmt="C3-", basefmt="k-",

                                         use_line_collection=True)



plt.setp(markerline, mec="k", mfc="w", zorder=3)



# Shift the markers to the baseline by replacing the y-data by zeros.

markerline.set_ydata(np.zeros(len(canada_dates)))



# annotate lines

vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]

for d, l, r, va in zip(canada_dates, levels, canada_detail, vert):

    ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*3),

                textcoords="offset points", va=va, ha="right")



# format xaxis with 1 month intervals

ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=1))

ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))

plt.setp(ax.get_xticklabels(), rotation=30, ha="right")



# remove y axis and spines

ax.get_yaxis().set_visible(False)

for spine in ["left", "top", "right"]:

    ax.spines[spine].set_visible(False)



ax.margins(y=0.1)

plt.show()
korea_time = pd.read_csv('/kaggle/input/coronavirusdataset/Time.csv')



korea_time.loc[0,'confirmed_daily'] = korea_time.loc[0,'confirmed']

for i in range(1, len(korea_time)):

    korea_time.loc[i, 'confirmed_daily'] = korea_time.loc[i, 'confirmed'] - korea_time.loc[i-1, 'confirmed']

    

korea_time.date = pd.to_datetime(korea_time.date)

mask = (korea_time['date'] >= '01-31-2020') & (korea_time['date'] <= '06-30-2020')

korea_time = korea_time.loc[mask]



korea_time_bar = korea_time[['date', 'confirmed_daily']]

korea_time_bar = korea_time_bar.groupby(pd.Grouper(key='date', freq='14D')).sum()



korea_time_bar

canada = pd.read_csv('/kaggle/input/canada-covid19-aug8/canada_covid19.csv')



canada_time = canada.loc[canada['prname'] == 'Canada']

canada_time = canada_time[['date', 'numtoday']]



canada_time.date = pd.to_datetime(canada_time.date, dayfirst=True)

mask = (canada_time['date'] >= '01-31-2020') & (canada_time['date'] <= '06-30-2020')

canada_time = canada_time.loc[mask]



canada_time_bar = canada_time.groupby(pd.Grouper(key='date', freq='14D')).sum()



canada_time_bar

data_labels = {

    "labels": [

        "2020-01-31",

        "2020-02-14",

        "2020-02-28",

        "2020-03-13",

        "2020-03-27",

        "2020-04-10",

        "2020-04-24",

        "2020-05-08",

        "2020-05-22",

        "2020-06-05",

        "2020-06-19"

    ]

}



total_confirmed_cases = go.Figure(

    data=[

        go.Bar(

            name="Korea",

            x=data_labels["labels"],

            y=korea_time_bar["confirmed_daily"],

            offsetgroup=0,

        ),

        go.Bar(

            name="Canada",

            x=data_labels["labels"],

            y=canada_time_bar["numtoday"],

            offsetgroup=1,

        ),

    ],

    layout=go.Layout(

        title="COVID19 daily confirmed cases (Jan. 31 to June 30)",

        yaxis_title="Number of cases",

        xaxis_title="Date (2 week periods)"

    )

)

total_confirmed_cases.show()

total_confirmed_cases_by_1Mpopulation = go.Figure(

    data=[

        go.Bar(

            name="Korea",

            x=data_labels["labels"],

            y=korea_time_bar["confirmed_daily"]/51.64,

            offsetgroup=0,

        ),

        go.Bar(

            name="Canada",

            x=data_labels["labels"],

            y=canada_time_bar["numtoday"]/37.59,

            offsetgroup=1,

        ),

    ],

    layout=go.Layout(

        title="COVID19 daily confirmed cases per 1M population (Jan. 31 to June 30)",

        yaxis_title="Number of cases per 1M population",

        xaxis_title="Date (2 week periods)"

    )

)

total_confirmed_cases_by_1Mpopulation.show()
