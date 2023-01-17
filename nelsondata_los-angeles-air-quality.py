import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

pd.options.display.max_columns=50
a = pd.read_csv("../input/airquality-July19.csv")



# clean data

a['MAX_CO2_PPM'] = pd.Series()

a['MAX_CO2_PPM'] = a['MAX_CO2_PPM'].fillna(5000.0)



# remove readings that are beyond the sensor's range of 400 to 8192 ppm for CO2 and 0 to 1187 for TVOCs

a = a.loc[a['CO2_PPM'] > 0]

a = a.loc[a['CO2_PPM'] < 8192]

%matplotlib inline

fig = plt.figure(figsize=(15,5))

lines = plt.plot(a['TIMESTAMP'], a['CO2_PPM'], a['MAX_CO2_PPM']) 

plt.setp(lines,color='r',linewidth=1.0, marker='')

plt.xlabel('Time')

plt.ylabel('Parts Per Million (PPM)')

plt.title("CO2_PPM - Indoor levels compared to Auto Exhaust")

plt.xticks(rotation='vertical', fontsize='7')

plt.show()

fig = plt.figure(figsize=(15,5))

lines = plt.plot(a['TIMESTAMP'], a['TVOC_PPB']) 

plt.setp(lines,color='k',linewidth=1.0, marker='')

plt.xlabel('Time')

plt.ylabel('Parts Per Billion (PPB)')

plt.title("TVOC_PPB (Total Volatile Organic Compound) - Indoor levels compared to Auto Exhaust")

plt.xticks(rotation='vertical', fontsize='7')

plt.show()
