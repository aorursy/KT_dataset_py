import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from plotnine import *

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, KFold

import lightgbm as lgb

from bayes_opt import BayesianOptimization

from sklearn.metrics import mean_squared_log_error

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dirc = "/kaggle/input/ashrae-energy-prediction/"

train = pd.read_csv(dirc + 'train.csv')

weather_train = pd.read_csv(dirc + 'weather_train.csv')

df_building = pd.read_csv(dirc + 'building_metadata.csv')

print('Train data shape:', train.shape,'\nTrain weather data shape:', weather_train.shape, '\nBuilding data shape:', df_building.shape)
train.head()
train.meter.unique()
(train.meter.value_counts('percent')*100).plot(kind='bar')

print((train.meter.value_counts('percent')*100))
fig, ax = plt.subplots(figsize=(12,9))

sns.boxplot(x='meter', y='meter_reading', data=train)

plt.xlabel('Meter type', fontsize=19, color='blue')

plt.ylabel('Meter Reading', fontsize=19, color='blue')
df_building.head()
df_building.isnull().sum(axis=0)/df_building.shape[0]*100
data = df_building.primary_use.value_counts('percent')*100

data.sort_values(ascending=True, inplace=True)

fig, ax = plt.subplots(figsize=(12,9))

data.plot(kind='barh')

for i, j in enumerate(data.values):

    plt.text(j, i-0.25, str(round(j,2))+ '%', fontsize=16, color='k')

plt.yticks(fontsize=17, color='b')

fig.set_facecolor('yellow')

ax.set_facecolor('pink')
df_building.describe()
fig, ax = plt.subplots(figsize=(12,9))

sns.boxplot(x='primary_use', y='square_feet', data=df_building)

plt.xticks(rotation=90)
weather_train.head()
weather_train.isnull().sum(axis=0)/weather_train.shape[0]*100
train = pd.merge(train, df_building, on='building_id', how='left')
weather_train['timestamp1'] = pd.to_datetime(weather_train.timestamp)

weather_train['month'] = np.uint8(weather_train.timestamp1.apply(lambda x:x.month))

weather_train['dom'] = np.uint8(weather_train.timestamp1.apply(lambda x:x.day))

weather_train['dow'] = np.uint8(weather_train.timestamp1.apply(lambda x:x.weekday()))

weather_train['hour'] = np.uint8(weather_train.timestamp1.apply(lambda x:x.hour))
del weather_train['timestamp1'] 
train = pd.merge(train, weather_train, on=['site_id', 'timestamp'], how='left')

del weather_train
from bokeh.models import Panel, Tabs

from bokeh.plotting import figure

from bokeh.io import output_notebook, show

output_notebook()



color_map = {

    "air_temperature": "yellow",

    "dew_temperature": "brown",

    "sea_level_pressure": "green",

    "wind_speed": "red",

    "cloud_coverage": "blue",

}



col_map = {

    "air_temperature": "Air Temperature",

    "dew_temperature": "Dew Temperature",

    "sea_level_pressure": "Sea Level Pressure",

    "wind_speed": "Wind Speed",

    "cloud_coverage": "Cloud Coverage",

}

def get_bar_plot_by_site(df,By):

    def get_plots(data, col, color, By):

        p = figure(plot_width=1000, plot_height=350, title=f"Mean of {col} by {By}")

        p.vbar(data[By], top=data[col], color=color, width=0.5)

        return p

    main_tabs_list = []

    cols = ["air_temperature","dew_temperature", "sea_level_pressure", "wind_speed", "cloud_coverage"]

    for col in cols:

        tab_list = []

        for site in range(16):

            temp = df[df["site_id"]==site]

            temp = temp.groupby(['site_id', By])[col].agg({col:'mean'})

            temp.reset_index(inplace=True)

            p = get_plots(temp, col, color_map[col],By)

            tab = Panel(child=p, title=f"Site:{site}")

            tab_list.append(tab)

        tabs = Tabs(tabs=tab_list)

        panel = Panel(child=tabs, title=col_map[col])

        main_tabs_list.append(panel)



    tabs = Tabs(tabs=main_tabs_list)

    show(tabs)
get_bar_plot_by_site(train, 'month')
get_bar_plot_by_site(train, 'dom')