import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use ("fivethirtyeight")
data=pd.read_csv('../input/land-ocean-temperature-index/Land Ocean Temp Index.csv')
df=pd.DataFrame (data)
df.sample(10)
year= df ['Year']
temp_change_rate= df ['No_Smoothing']
x=year
y=temp_change_rate
plt.plot (x,y, label='Change in Temperature', color='#1a0000')
plt.title ("\nChange In Global Surface Temperature\n", color='k')
plt.xlabel ("\nYear\n", color= 'k')
plt.ylabel ("\nChange in Temperature\n", color= 'k')
plt.legend ()
plt.grid (True, color='gray')
plt.rcParams['figure.figsize'] = [20, 10]
plt.show()
data=pd.read_csv('../input/ozone-depleting-substance-emissions/ozone-depleting-substance-emissions.csv')
df=pd.DataFrame (data)
df.sample(10)
year = df ['Year']
emission = df ['Ozone-depleting substance emissions (Scientific Assessment 2014) (tonnes CFC11-equivalents)']
x=year
y=emission
plt.plot (x,y, label='Ozone-Depleting Substance Emission', color='#008B8B')
plt.title ("\nTotal Ozone-Depleting Substance Emission over the years\n", color='k')
plt.xlabel ("\nYear\n", color= 'k')
plt.ylabel ("\nOzone-Depleting Substance Emissions (tonnes CFC11-equivalents)\n", color= 'k')
plt.legend ()
plt.grid (True, color='grey')
plt.rcParams['figure.figsize'] = [20, 10]
plt.show()
data=pd.read_csv('../input/carbon-dioxide-emission-rates/Carbon Dioxide Emission.csv')
df=pd.DataFrame (data)
df.sample(10)
year = df ['Decimal_Date'].tail(400)
avg = df ['Average'].tail(400)
x=year
y=avg
plt.plot (x,y, label='CO2 Emission Rate', color='#8A3324')
plt.title ("\nCO2 Emission Rates over the years\n", color='k')
plt.xlabel ("\nYear\n", color= 'k')
plt.ylabel ("\nCO2 Emission  (PPM)\n", color= 'k')
plt.legend ()
plt.grid (True, color='grey')
plt.rcParams['figure.figsize'] = [20, 10]
plt.show()
data=pd.read_csv('../input/antarctic-ozone-hole-area/antarctic-ozone-hole-area.csv')
df=pd.DataFrame (data)
df.sample(10)
year = df ['Year']
max = df ['Maximum ozone hole area (square kilometres)']
mean = df ['Mean ozone hole area (square kilometres)']
plt.plot (year,max, label='Max', color='r')
plt.plot (year,mean, label='Mean', color='b')
plt.title ("\nAntarctic Ozone Hole Area\n", color='k')
plt.xlabel ("\nYear\n", color= 'k')
plt.ylabel ("\nOzone Hole Area (Square Kilometres)\n", color= 'k')
plt.legend ()
plt.grid (True, color='grey')
plt.rcParams['figure.figsize'] = [20, 10]
plt.show()
data=pd.read_csv('../input/dhaka-temperature/Dhaka Temperature.csv')
df=pd.DataFrame (data)
df.sample(10)
df.set_index('Date')
df = df.sort_values(by ='Date' )
df.tail (10)
date = df ['Date'].tail(61)
min = df ['min'].tail(61)
max = df ['max'].tail(61)
y_pos = np.arange(len(date))
plt.bar(y_pos, min, align='center', color='#E0115F', alpha=.80),
plt.bar(y_pos, max, align='center', color='#002147',alpha=0.75),
plt.xticks(y_pos)
plt.ylabel('\n Temperature (in Celcius)\n')
plt.title('\n Daily Temperatures in Dhaka During March & April (2020) Days\n')
plt.grid (True, color='grey')
leg=['Min', 'Max']
plt.legend(leg)
plt.rcParams['figure.figsize'] = [25, 10]
plt.show()
