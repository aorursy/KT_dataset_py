import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import plotly.express as px
df = pd.read_csv('../input/carla-driver-behaviour-dataset/full_data_carla.csv')
df.head()
df['class'].value_counts().plot(kind='bar')

plt.grid()

plt.show()
# extract only one driver

df_example = df[df['class']=='gonca']
fig = px.scatter_3d(x=df_example['accelX'], y=df_example['accelY'], z=df_example['accelZ'], opacity=0.5)

fig.update_layout(title='Acceleration for specific driver')

fig.show()
fig = px.scatter_3d(x=df_example['gyroX'], y=df_example['gyroY'], z=df_example['gyroZ'], opacity=0.5)

fig.update_layout(title='Gyro Data for specific driver')

fig.show()