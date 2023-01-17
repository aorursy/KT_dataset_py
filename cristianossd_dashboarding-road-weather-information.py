# -*- coding: utf-8 -*-
%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os


print(os.listdir("../input"))
METADATA_DOC = '../input/RWIS Metadata.docx'
METADATA_JSON = '../input/socrata_metadata.json'
DATASET_FILE = '../input/road-weather-information-stations.csv'

df = pd.read_csv(DATASET_FILE)
df.head()
import sys
!{sys.executable} -m pip install csvvalidator
from csvvalidator import *


field_names = ('StationName', 'StationLocation', 'DateTime', 'RecordId', 'RoadSurfaceTemperature', 'AirTemperature')
validator = CSVValidator(field_names)
validator.add_value_check('StationName', str, 'EX1', 'station name must be string')
validator.add_value_check('DateTime', datetime_string('%Y-%m%dT%H:%M:%S'), 'EX2', 'invalid date')
validator.add_value_check('RoadSurfaceTemperature', float, 'EX3', 'tempearture must be float')
'Validation errors: {}'.format(len(validator.validate(df)))
df['StationName'].value_counts()
# reduce dataset for JoseRizalBridgeNorth station
df = df[df['StationName'] == 'JoseRizalBridgeNorth']
df.drop(columns=['StationName'], inplace=True)
# surface temperature distribution from last day
from datetime import datetime, timedelta

last_collected_day = df.tail(1)['DateTime'].values[0]
last_collected_day = datetime.strptime(last_collected_day, '%Y-%m-%dT%H:%M:%S')
last_day = last_collected_day - timedelta(1)
start = last_day.strftime('%Y-%m-%dT00:00:00')
end = last_day.strftime('%Y-%m-%dT23:59:59')

df_last_hour = df[(df['DateTime'] >= start) & (df['DateTime'] <= end)]
df_last_hour.shape
df_last_hour['DateTime'] = pd.to_datetime(df_last_hour['DateTime'], format='%Y-%m-%dT%H:%M:%S')
df_by_hour = df_last_hour.groupby([df_last_hour['DateTime'].dt.hour])['RoadSurfaceTemperature'].median()
index = np.arange(len(df_by_hour))

plt.figure(figsize=(12, 4))
plt.plot(index, df_by_hour)
plt.xlabel('hour', fontsize=14)
plt.ylabel('temperature (Celsius)', fontsize=14)
plt.xticks(index, df_by_hour.index.values.astype(int), fontsize=14, rotation=30)
plt.title('Road surface temperature last day ({})'.format(last_day.strftime('%Y-%m-%d')))
plt.show()
import plotly.plotly as py
import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
data = [go.Scatter(x=df_by_hour.index.values.astype(int),
                   y=df_by_hour, mode='lines+markers')]

layout = dict(title='Road surface temperature last collected day ({})'.format(last_day.strftime('%Y-%m-%d')),
             xaxis=dict(title='Hour', ticklen=5, zeroline=False))

fig = dict(data=data, layout=layout)
iplot(fig)
