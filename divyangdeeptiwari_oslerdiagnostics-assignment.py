# I'm going to use pandas library to import the csv file in a dataframe.

import pandas as pd
voltage_data = pd.read_csv('../input/voltagetime/voltage_data.csv')
voltage_data.head()
voltage_data.shape
voltage_data.isna().sum()
# There are only 5 not defined observations. Dropping them is not a significant loss.

voltage_data = voltage_data.dropna()

voltage_data.shape
# Resetting the index.

voltage_data.reset_index(drop= True, inplace=True)
# Let's use matplotlib library to visualize the trend between voltage and time

import matplotlib.pyplot as plt
plt.plot(voltage_data['Time'],voltage_data['Voltage'])

plt.xlabel('Time')

plt.ylabel('Voltage')

plt.title('Voltage v/s Time')
# Let's define a function to implement EWMA

def EWMA_func(voltage_data, beta=0.5):

    voltage_data['Volt_EWMA'] = voltage_data['Voltage'].values

    for i in range(1,voltage_data['Voltage'].shape[0]):

        voltage_data['Volt_EWMA'][i] = beta*voltage_data['Volt_EWMA'][i-1] + ((1-beta)*voltage_data['Voltage'][i])

    return beta
# Let's plot the voltage v/s time for different values of 'beta'

beta = EWMA_func(voltage_data, beta=0.9)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

fig.suptitle('Voltage-Time profile before and after applying EWMA')



ax1.plot(voltage_data['Time'],voltage_data['Voltage'], c='blue')

ax1.set_title('Voltage v/s Time without EWMA')

ax1.set_xlabel('Time')

ax1.set_ylabel('Voltage')



ax2.plot(voltage_data['Time'],voltage_data['Volt_EWMA'], c='blue')

ax2.set_title('Voltage v/s Time after EWMA for beta =' + ' ' + str(beta))

ax2.set_xlabel('Time')

ax2.set_ylabel('Voltage')
from scipy.signal import lfilter
n = 15  # high value of n increases the smoothness in curve

b = [1.0 / n] * n

a = 1
# Let's make a separate column in dataframe for this approach

voltage_data['Volt_lfilter'] = voltage_data['Voltage'].values
voltage_data
voltage_data['Volt_lfilter'] = lfilter(b, a, voltage_data['Voltage'])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

fig.suptitle('Voltage-Time profile before and after applying lfilter')



ax1.plot(voltage_data['Time'],voltage_data['Voltage'], c='blue')

ax1.set_title('Voltage v/s Time without lfilter')

ax1.set_xlabel('Time')

ax1.set_ylabel('Voltage')



ax2.plot(voltage_data['Time'],voltage_data['Volt_lfilter'], c='blue')

ax2.set_title('Voltage v/s Time after implementing lfilter')

ax2.set_xlabel('Time')

ax2.set_ylabel('Voltage')
#beta = EWMA_func(voltage_data, beta=0.9)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

fig.suptitle('Comparison between results from EWMA and lfilter')



ax1.plot(voltage_data['Time'],voltage_data['Volt_EWMA'], c='blue')

ax1.set_title('Voltage v/s Time after EWMA for beta =' + ' ' + str(beta))

ax1.set_xlabel('Time')

ax1.set_ylabel('Voltage')



ax2.plot(voltage_data['Time'],voltage_data['Volt_lfilter'], c='blue')

ax2.set_title('Voltage v/s Time after implementing lfilter')

ax2.set_xlabel('Time')

ax2.set_ylabel('Voltage')