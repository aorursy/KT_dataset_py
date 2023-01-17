'''
Author: Ritwik Biswas
Description: Analysis of temperature time_series data
'''

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import os
df = pd.read_csv('../input/temp-humidity.csv',header=0)
df.columns = ['Time', 'Temperature', 'Humidity']
df.head()
# Temperature and Humidy Time Series
df.plot(title="Temperature and Humidity vs Time",figsize=(18, 10))
plt.show()
# Temperature vs Humidity
df.plot(x=1,y=2,kind="scatter",title="Temperature vs Humidity",figsize=(18, 10)) 
plt.show()
print("Correlation Score: " + str(df['Temperature'].corr(df['Humidity'])))
# Calculate Moving Average with window 10
df['temperature_ma'] = df['Temperature'].rolling(1000).mean()
df['humidity_ma'] = df['Humidity'].rolling(1000).mean()
df.tail()
df.plot(y=[3,4],kind="line",title="Temperature vs Humidity",figsize=(18, 10)) 
plt.show()
df.plot(x=3,y=4,kind="scatter",title="Temperature_MA vs Humidity_MA",figsize=(18, 10)) 
plt.show()
print("Moving Average Correlation Score: " + str(df['temperature_ma'].corr(df['humidity_ma'])))
df.temperature_ma.plot.hist(alpha=0.5, figsize=(18, 10),title="Freq Distribution of Temp and Humidity",bins=50, legend=True)
df.humidity_ma.plot.hist(alpha=0.5,figsize=(18, 10), bins=50, legend=True)
plt.show()