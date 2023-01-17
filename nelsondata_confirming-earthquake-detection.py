# Find earthquake data at: https://www.cisn.org/shakemap/sc/shake/archive/

# Copy shakemap to data/CISN-Shakemap-May-2020.csv

# Set the delimiter to |

# Change the column headings to:

#     EQ_EventID|EQ_Epicenter|EQ_Date|EQ_Time|EQ_Lat|EQ_Lng|EQ_Mag



AstroPiQuake_data = "/kaggle/input/shakemap/AstroPiQuake-2019.csv"

shakemap = "/kaggle/input/shakemap/CISN-Shakemap-July-2019.csv"



# earthquake magnitude threshold

EQmagnitude = 3.5



# import the necessary libraries and set the dataframe viewing options

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

pd.options.display.max_columns=50

pd.options.display.max_rows=250
# A sensor reading was taken every 5 minutes with a timestamp in this format:  20190715 01:50

# Reformat this timestamp into three fields: READING_DATE:  2019-07-15, READING_TIME: 01:50, and MERGEHOUR: 20190715 01

# MERGEHOUR is used so that sensor readings taken within one hour of a known earthquake can be merged into a new dataframe for further analysis to find sensor readings during an eathquake



aq = pd.read_csv(AstroPiQuake_data)

print(aq.info())

aq['READING_DATE'] = pd.to_datetime(aq['timestamp'].str[:8]).apply(lambda x:x.strftime('%Y-%m-%d'))

aq['READING_TIME'] = aq['timestamp'].str[-5:]

aq['MERGEHOUR'] = aq['timestamp'].str[:11]

print("\nBefore dropping duplicates: ", aq.shape)

aq.sort_values(["READING_DATE", "READING_TIME"], inplace = True) 

aq.drop_duplicates(subset =["READING_DATE", "READING_TIME"], keep='first', inplace=True) 

print("\nAfter dropping duplicates: ", aq.shape)

print("\nUnique dates in AstroPiQuake data:\n",aq['READING_DATE'].unique())

# Format the shakemap EQ_Date and EQ_Time into the MERGEHOUR field



sm = pd.read_csv(shakemap, delimiter='|')

print(sm.info())

sm['EQ_DATE'] = pd.to_datetime(sm['EQ_Date']).apply(lambda x:x.strftime('%Y-%m-%d'))

sm['EQ_TIME'] = [str(int(str(x)[:2])+12)+str(x)[2:5] if "PM" in x else (str(x)[:5]) for x in sm['EQ_Time']]

sm['MERGEHOUR'] = sm["EQ_DATE"].str[:4]+sm["EQ_DATE"].str[5:7]+sm["EQ_DATE"].str[8:10]+" " +sm["EQ_TIME"].str[:2]

# find earthquakes greater than the threshold



t = sm.loc[sm['EQ_Mag'] > EQmagnitude]

t = t.sort_values(["EQ_DATE", "EQ_TIME"])

print("\nCISN Shakemap data: ",t.shape)

# Merge AstroPiQuake and shakemap data to find sensor data at the time of an earthquake



eq = pd.merge(aq, t, how='inner', on='MERGEHOUR', suffixes=(False,False))

eq['EQ_MIN'] = eq['EQ_TIME'].str[3:]

eq['READING_MIN'] = eq['READING_TIME'].str[3:]

eq['EQ_MIN'] = eq.EQ_MIN.astype('int64')

eq['READING_MIN'] = eq.READING_MIN.astype('int64')



eq['x'] = eq['x'].round(decimals=6)

eq['y'] = eq['y'].round(decimals=6)

eq['z'] = eq['z'].round(decimals=6)

eq['pitch'] = eq['pitch'].round()

eq['roll'] = eq['roll'].round()

eq['yaw'] = eq['yaw'].round()



matchEQ = eq.loc[eq['EQ_MIN'].between(eq['READING_MIN']-1,eq['READING_MIN']+5)]

eq = matchEQ



# add missing latitudes and longitudes for AstroPiQuake

eq["lng"] = eq["lng"].fillna("-118.323411")

eq["lat"] = eq["lat"].fillna("33.893916")



# check for nulls

print("\nThere should be zero null values in earthquake data found:")

print(eq.isnull().sum())

# EARTHQUAKES FOUND

# This list contains the date, time, magnitude, and epicenter of known earthquakes in Southern California along with the sensor reading near that time



print("\nEarthquake locations and magnitudes sorted by date, time, reading date, and reading time\n")

eq.sort_values(['EQ_DATE','EQ_TIME','READING_DATE','READING_TIME'], inplace=True, ascending=True)

display(eq[['EQ_DATE','EQ_TIME','READING_DATE','READING_TIME','EQ_Mag','EQ_Epicenter','EQ_Lat','EQ_Lng']])
# MOVEMENT PLOTS

fig, axs = plt.subplots(3,3,sharex=True)

fig.suptitle('4.2 Earthquake 5.4 miles east of Coso Junction, CA on July 15, 2019')



# Temperature

axs[0,0].plot(eq['MERGEHOUR'], eq['temperature'], color="orange") 

axs[0,0].set_ylabel("temperature Celsius", fontsize='7')

axs[0,0].set_yticklabels(eq['temperature'].round(), rotation=0, fontsize=7)

axs[0,0].set_xticklabels(eq['MERGEHOUR'], rotation=45, fontsize='7')

axs[0,0].axvline(x=2, ymin=0, ymax=1, ls='--', c="red")

axs[0,0].legend()



# Humidity

axs[1,0].plot(eq['MERGEHOUR'], eq['humidity'], color="green") 

axs[1,0].set_ylabel("percent humidity", fontsize='7')

axs[1,0].set_yticklabels(eq['humidity'].round(), rotation=0, fontsize=7)

axs[1,0].set_xticklabels(eq['MERGEHOUR'], rotation=45, fontsize='7')

axs[1,0].axvline(x=2, ymin=0, ymax=1, ls='--', c="red")

axs[1,0].legend()



# Pressure

axs[2,0].plot(eq['MERGEHOUR'], eq['pressure']) 

axs[2,0].set_ylabel("barometric pressure", fontsize='7')

axs[2,0].set_yticklabels(eq['pressure'].round(), rotation=0, fontsize=7)

axs[2,0].set_xticklabels(eq['MERGEHOUR'], rotation=45, fontsize='7')

axs[2,0].axvline(x=2, ymin=0, ymax=1, ls='--', c="red")

axs[2,0].legend()





axs[2,1].set_title("Accelerometer")

axs[2,1].plot(eq['MERGEHOUR'], eq['x'])

axs[2,1].plot(eq['MERGEHOUR'], eq['y'])

axs[2,1].plot(eq['MERGEHOUR'], eq['z'])

axs[2,1].set_ylabel('G-Forces')

axs[2,1].set_xticklabels(eq['MERGEHOUR'], rotation=45, fontsize='7')

axs[2,1].axvline(x=2, ymin=0, ymax=1, ls='--', c="red")

axs[2,1].legend()



axs[0,1].set_visible(False)

axs[1,1].set_visible(False)



axs[0,2].set_title("Gyroscope")

axs[0,2].plot(eq['MERGEHOUR'], eq['yaw'], color="silver") 

axs[0,2].set_ylabel('degrees', fontsize='7')

axs[0,2].set_yticklabels(eq['yaw'], rotation=0, fontsize=7)

axs[0,2].set_xticklabels(eq['MERGEHOUR'], rotation=45, fontsize='7')

axs[0,2].axvline(x=2, ymin=0, ymax=1, ls='--', c="red")

axs[0,2].legend()



axs[1,2].plot(eq['MERGEHOUR'], eq['pitch'], color="gray") 

axs[1,2].set_ylabel('degrees', fontsize='7')

axs[1,2].set_yticklabels(eq['pitch'], rotation=0, fontsize=7)

axs[1,2].set_xticklabels(eq['MERGEHOUR'], rotation=45, fontsize='7')

axs[1,2].axvline(x=2, ymin=0, ymax=1, ls='--', c="red")

axs[1,2].legend()



axs[2,2].plot(eq['MERGEHOUR'], eq['roll'], color="black") 

axs[2,2].set_ylabel('degrees', fontsize='7')

axs[2,2].set_yticklabels(eq['roll'], rotation=0, fontsize=7)

axs[2,2].set_xticklabels(eq['MERGEHOUR'], rotation=45, fontsize='7')

axs[2,2].axvline(x=2, ymin=0, ymax=1, ls='--', c="red")

axs[2,2].legend()



plt.show()

# MAGNITUDE REPORTS

print("\nSORT BY EARTHQUAKE MAGNITUDE")

eq.sort_values(['EQ_Mag','READING_DATE','READING_TIME'], inplace=True, ascending=False)

display(eq[['EQ_Mag','EQ_Epicenter','EQ_Lat','EQ_Lng','EQ_DATE','EQ_TIME','READING_DATE','READING_TIME' ]])

# NOTE:  HOW TO FIND DATA

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.