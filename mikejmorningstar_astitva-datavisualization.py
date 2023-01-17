# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.express as px

from plotly.offline import init_notebook_mode

from pprint import pprint

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics



init_notebook_mode(connected = True)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# IMPORT DATA AS DATAFRAME

df_pg1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

df_pg2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')

df_wsd1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

df_wsd2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
# Extract DATE_TIME object from string

df_pg1['DATE_TIME'] = pd.to_datetime(df_pg1['DATE_TIME'], format = '%d-%m-%Y %H:%M')

df_pg2['DATE_TIME'] = pd.to_datetime(df_pg2['DATE_TIME'], format = '%Y-%m-%d %H:%M')

df_wsd1['DATE_TIME'] = pd.to_datetime(df_wsd1['DATE_TIME'], format = '%Y-%m-%d %H:%M')

df_wsd2['DATE_TIME'] = pd.to_datetime(df_wsd2['DATE_TIME'], format = '%Y-%m-%d %H:%M')

df_pg1['DATE'] = df_pg1['DATE_TIME'].dt.date

df_pg2['DATE'] = df_pg2['DATE_TIME'].dt.date

df_wsd1['DATE'] = df_wsd1['DATE_TIME'].dt.date

df_wsd2['DATE'] = df_wsd2['DATE_TIME'].dt.date

df_pg1['TIME'] = df_pg1['DATE_TIME'].dt.time

df_pg2['TIME'] = df_pg2['DATE_TIME'].dt.time

df_wsd1['TIME'] = df_wsd1['DATE_TIME'].dt.time

df_wsd2['TIME'] = df_wsd2['DATE_TIME'].dt.time
# AC vs DC

tmp = df_pg1.groupby(['TIME'], as_index = False).agg({'DC_POWER': 'mean', 'AC_POWER': 'mean'})

fig_gph = go.Figure()

fig_gph.add_trace(go.Scatter(x = tmp.DC_POWER, y = tmp.AC_POWER, mode = 'lines'))

fig_gph.update_layout(title = 'Mean AC/DC Power by Module at Station 1', xaxis_title = 'DC Power', yaxis_title = 'AC Power')

fig_gph.show()



tmp = df_pg2.groupby(['TIME'], as_index = False).agg({'DC_POWER': 'mean', 'AC_POWER': 'mean'})

fig_gph = go.Figure()

fig_gph.add_trace(go.Scatter(x = tmp.DC_POWER, y = tmp.AC_POWER, mode = 'lines'))

fig_gph.update_layout(title = 'Mean AC/DC Power by Module at Station 2', xaxis_title = 'DC Power', yaxis_title = 'AC Power')

fig_gph.show()
# Irradiation vs Temp diff

fig_gph = go.Figure()

fig_gph.add_trace(go.Scatter(x = df_wsd1['IRRADIATION'], y = df_wsd1['MODULE_TEMPERATURE'] - df_wsd1['AMBIENT_TEMPERATURE'], mode = 'markers', name = 'Module Temp'))

fig_gph.update_layout(title = 'Irradiation vs Difference of Module Temperature and Ambient Temperature at Plant 1', xaxis_title = 'Irradiation', yaxis_title = 'Temperature Difference')

fig_gph.show()



fig_gph = go.Figure()

fig_gph.add_trace(go.Scatter(x = df_wsd2['IRRADIATION'], y = df_wsd2['MODULE_TEMPERATURE'] - df_wsd1['AMBIENT_TEMPERATURE'], mode = 'markers', name = 'Module Temp'))

fig_gph.update_layout(title = 'Irradiation vs Difference of Module Temperature and Ambient Temperature at Plant 2', xaxis_title = 'Irradiation', yaxis_title = 'Temperature Difference')

fig_gph.show()
#Time vs Temp

tmp = df_wsd1.groupby('TIME', as_index = False).agg({'MODULE_TEMPERATURE': 'mean', 'AMBIENT_TEMPERATURE': 'mean'})

fig_gph = go.Figure()

fig_gph.add_trace(go.Scatter(x = tmp.TIME, y = tmp.MODULE_TEMPERATURE, mode = 'lines', name = 'Module Temp'))

fig_gph.add_trace(go.Scatter(x = tmp.TIME, y = tmp.AMBIENT_TEMPERATURE, mode = 'lines', name = 'Ambient Temp'))

fig_gph.add_trace(go.Scatter(x = tmp.TIME, y = np.abs(tmp.MODULE_TEMPERATURE - tmp.AMBIENT_TEMPERATURE), mode = 'lines', name = 'Difference'))

fig_gph.update_layout(title = 'Mean Tempature over time at Station 1', xaxis_title = 'Temp', yaxis_title = 'Dates')

fig_gph.show()



tmp = df_wsd2.groupby('TIME', as_index = False).agg({'MODULE_TEMPERATURE': 'mean', 'AMBIENT_TEMPERATURE': 'mean'})

fig_gph = go.Figure()

fig_gph.add_trace(go.Scatter(x = tmp.TIME, y = tmp.MODULE_TEMPERATURE, mode = 'lines', name = 'Module Temp'))

fig_gph.add_trace(go.Scatter(x = tmp.TIME, y = tmp.AMBIENT_TEMPERATURE, mode = 'lines', name = 'Ambient Temp'))

fig_gph.add_trace(go.Scatter(x = tmp.TIME, y = np.abs(tmp.MODULE_TEMPERATURE - tmp.AMBIENT_TEMPERATURE), mode = 'lines', name = 'Difference'))

fig_gph.update_layout(title = 'Mean Tempature over time at Station 2', xaxis_title = 'Temp', yaxis_title = 'Dates')

fig_gph.show()
#Time vs AC/DC

tmp = df_pg1.groupby('TIME', as_index = False).agg({'DC_POWER': 'mean', 'AC_POWER': 'mean'})

fig_gph = go.Figure()

fig_gph.add_trace(go.Scatter(x = tmp.TIME, y = tmp.DC_POWER/10, mode = 'lines', name = 'Mean DC Power'))

fig_gph.add_trace(go.Scatter(x = tmp.TIME, y = tmp.AC_POWER, mode = 'lines', name = 'Mean AC Power'))

fig_gph.update_layout(title = 'Mean AC/DC Power over Time Intervals at Station 1', xaxis_title = 'Time', yaxis_title = 'Power', xaxis_tickangle = 45)

fig_gph.show()



tmp = df_pg2.groupby('TIME', as_index = False).agg({'DC_POWER': 'mean', 'AC_POWER': 'mean'})

fig_gph = go.Figure()

fig_gph.add_trace(go.Scatter(x = tmp.TIME, y = tmp.DC_POWER, mode = 'lines', name = 'Mean DC Power'))

fig_gph.add_trace(go.Scatter(x = tmp.TIME, y = tmp.AC_POWER, mode = 'lines', name = 'Mean AC Power'))

fig_gph.update_layout(title = 'Mean AC/DC Power over Time Intervals at Station 2', xaxis_title = 'Time', yaxis_title = 'Power', xaxis_tickangle = 45)

fig_gph.show()
#Date vs AC/DC

tmp = df_pg1.groupby('DATE', as_index = False).agg({'DC_POWER': 'mean', 'AC_POWER': 'mean'})

fig_gph = go.Figure()

fig_gph.add_trace(go.Scatter(x = tmp.DATE, y = tmp.DC_POWER/10, mode = 'lines', name = 'Mean DC Power'))

fig_gph.add_trace(go.Scatter(x = tmp.DATE, y = tmp.AC_POWER, mode = 'lines', name = 'Mean AC POWER'))

fig_gph.update_layout(title = 'Mean AC/DC Power over 34 days at Station 1', xaxis_title = 'Dates', yaxis_title = 'Mean Power', xaxis_tickangle = 45)

fig_gph.show()



tmp = df_pg2.groupby('DATE', as_index = False).agg({'DC_POWER': 'mean', 'AC_POWER': 'mean'})

fig_gph = go.Figure()

fig_gph.add_trace(go.Scatter(x = tmp.DATE, y = tmp.DC_POWER, mode = 'lines', name = 'Mean DC Power'))

fig_gph.add_trace(go.Scatter(x = tmp.DATE, y = tmp.AC_POWER, mode = 'lines', name = 'Mean AC POWER'))

fig_gph.update_layout(title = 'Mean AC/DC Power over 34 days at Station 2', xaxis_title = 'Dates', yaxis_title = 'Mean Power', xaxis_tickangle = 45)

fig_gph.show()
#Source key wise DC/AC

tmp = df_pg1.groupby(['SOURCE_KEY', 'DATE'], as_index = False).agg({'DC_POWER': 'mean', 'AC_POWER': 'mean'})

fig_gph = go.Figure()

keys = df_pg1['SOURCE_KEY'].unique()

for key in keys:

    fig_gph.add_trace(go.Scatter(x = tmp[tmp['SOURCE_KEY'] == key].DATE, y = tmp[tmp['SOURCE_KEY'] == key].AC_POWER, mode = 'lines', name = key))

fig_gph.update_layout(title = 'Mean AC Power by each Module per day at Station 1', xaxis_title = 'Dates', yaxis_title = 'Power')

fig_gph.show()

fig_gph = go.Figure()

for key in keys:

    fig_gph.add_trace(go.Scatter(x = tmp[tmp['SOURCE_KEY'] == key].DATE, y = tmp[tmp['SOURCE_KEY'] == key].DC_POWER, mode = 'lines', name = key))

fig_gph.update_layout(title = 'Mean DC Power by each Module per day at Station 1', xaxis_title = 'Dates', yaxis_title = 'Power')

fig_gph.show()



tmp = df_pg2.groupby(['SOURCE_KEY', 'DATE'], as_index = False).agg({'DC_POWER': 'mean', 'AC_POWER': 'mean'})

fig_gph = go.Figure()

keys = df_pg2['SOURCE_KEY'].unique()

for key in keys:

    fig_gph.add_trace(go.Scatter(x = tmp[tmp['SOURCE_KEY'] == key].DATE, y = tmp[tmp['SOURCE_KEY'] == key].AC_POWER, mode = 'lines', name = key))

fig_gph.update_layout(title = 'Mean AC Power by each Module per day at Station 2', xaxis_title = 'Dates', yaxis_title = 'Power')

fig_gph.show()

fig_gph = go.Figure()

for key in keys:

    fig_gph.add_trace(go.Scatter(x = tmp[tmp['SOURCE_KEY'] == key].DATE, y = tmp[tmp['SOURCE_KEY'] == key].DC_POWER, mode = 'lines', name = key))

fig_gph.update_layout(title = 'Mean DC Power by each Module per day at Station 2', xaxis_title = 'Dates', yaxis_title = 'Power')

fig_gph.show()
#SeaBorn PairPlot

tmp = pd.merge(df_pg1.drop(columns = ['PLANT_ID']), df_wsd1.drop(columns = ['DATE', 'TIME', 'PLANT_ID', 'SOURCE_KEY']), on = 'DATE_TIME', how = 'left')

print('Station 1 Power Data Pair Plot')

sns.pairplot(tmp)

plt.show()

tmp = pd.merge(df_pg2.drop(columns = ['PLANT_ID']), df_wsd1.drop(columns = ['DATE', 'TIME', 'PLANT_ID', 'SOURCE_KEY']), on = 'DATE_TIME', how = 'left')

print('Station 2 Power Data Pair Plot')

sns.pairplot(tmp)

plt.show()
#Scatter Summary

tmp = df_pg1[['DATE', 'SOURCE_KEY', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']]

fig_gph = go.Figure(data = go.Splom(dimensions=

                                        [dict(label='DC', values=tmp['DC_POWER']), 

                                         dict(label='AC', values=tmp['AC_POWER']), 

                                         dict(label='Daily Yield', values=tmp['DAILY_YIELD']), 

                                         dict(label='Total Yield', values=tmp['TOTAL_YIELD']), 

                                         dict(label='Date', values=tmp['DATE'])], 

                                    diagonal_visible = False, marker = dict(color = tmp['SOURCE_KEY'].astype('category').cat.codes, showscale = False)))

fig_gph.update_layout(title = 'Plant 1 Summary')

fig_gph.show()



tmp = df_pg2[['DATE', 'SOURCE_KEY', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']]

fig_gph = go.Figure(data = go.Splom(dimensions=

                                        [dict(label='DC', values=tmp['DC_POWER']), 

                                         dict(label='AC', values=tmp['AC_POWER']), 

                                         dict(label='Daily Yield', values=tmp['DAILY_YIELD']), 

                                         dict(label='Total Yield', values=tmp['TOTAL_YIELD']), 

                                         dict(label='Date', values=tmp['DATE'])], 

                                    diagonal_visible = False, marker = dict(color = tmp['SOURCE_KEY'].astype('category').cat.codes, showscale = False)))

fig_gph.update_layout(title = 'Plant 1 Summary')

fig_gph.show()