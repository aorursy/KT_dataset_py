import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/environmental-sensor-data-132k/iot_telemetry_data.csv', engine='python')
data
# !pip install -q pandas-profiling[notebook]

# from pandas_profiling import ProfileReport

# profile = ProfileReport(data, title='Pandas Profiling Report')

# profile.to_notebook_iframe()
# convert the boolean columns to int32 for plotting

data['light_int'] = data['light'].astype('int32')

data['motion_int'] = data['motion'].astype('int32')
# convert unix time to time of day

from datetime import datetime, timedelta

start = datetime(1970, 1, 1)  # Unix epoch start time

data['datetime'] = data.ts.apply(lambda x: start + timedelta(seconds=x))

data['string_time'] = data.datetime.apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
# separate out the data for the different devices with a groupby

data_device_gb = data.groupby('device')
for i in data_device_gb:

    print(i[0])
!pip install -q plotly
# plot our time series again with a more meaningful time axis and the ability to select individual sensor time series by double clicking on them in the legend



cols = data.columns

unwanted_cols = set(['motion','ts', 'device', 'light', 'datetime', 'string_time'])



import plotly.express as px 



plt_idx = 0

for z in data_device_gb:

    fig = px.line(log_y = True, title = z[0])

    for i, j in enumerate(cols):

       # print(i)

        if j in unwanted_cols:

            continue

        else:

            fig.add_scatter(x=z[1].iloc[:,-1], y=z[1].iloc[:,i], mode='lines')

            #print(i, j)

            fig.data[plt_idx].name = j

            plt_idx += 1



    fig.show()

    fig.data = []

    plt_idx = 0
subset = set(['smoke', 'humidity', 'temp'])

f, axes = plt.subplots(1,3, figsize=(30, 10))



for i, j in enumerate(subset):

    sns.boxplot(  y=data[j], x= "device", data=data, hue = 'device', orient='v' , ax=axes[i])
!pip install -q fbprophet
data['ds'] = data['datetime']

data['y'] = data['smoke']

data_device_gb = data.groupby('device')
# create a dictionary of dataframes from the groupby

df_dict = {}

for i, j in enumerate(data_device_gb):

    df_dict[i] = j[1]
df_dict[0][['ds','y']]
# Be advised - the code below, fitting the prophet model, takes a very long time to run

from fbprophet import Prophet
m = Prophet()



prophet_dict = {}

for i in df_dict:

    prophet_dict[i] = m.fit(df_dict[i][['ds','y']])

    m = Prophet()
future_dict = {}

for i in prophet_dict:

    m = prophet_dict[i]

    future_dict[i] = m.make_future_dataframe(periods=0, freq='H')

# future = m.make_future_dataframe(periods=0, freq='H')
fcst_dict = {}

for i in future_dict:

    m = prophet_dict[i]

    fcst_dict[i] = m.predict(future_dict[i])

# fcst = m.predict(future)
for i in fcst_dict:

    m = prophet_dict[i]

    fig = m.plot_components(fcst_dict[i])

    ax = fig.gca()

    ax.set_title("Smoke - Device {}".format(i+1), size=16, loc = 'right')

#fig = m.plot_components(fcst)
# make the datetime column the index and then use index.day to groupby day



# DFList = []

# for group in df.groupby(df.index.day):

#     DFList.append(group[1])