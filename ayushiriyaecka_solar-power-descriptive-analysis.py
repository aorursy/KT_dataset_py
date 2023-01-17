# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd 
import numpy as np
import math 
import matplotlib.pyplot as plt
import seaborn as sn
import datetime as dt
# Read files WS = Weather_Sensor and G = Generation
Plant2_WS_df = pd.read_csv("../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")
Plant2_G_df = pd.read_csv("../input/solar-power-generation-data/Plant_2_Generation_Data.csv")
Plant1_WS_df = pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")
Plant1_G_df = pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv")
#displaying first few records of dataframe and exploring dataset
Plant2_WS_df.head(5)

Plant2_WS_df.info()
Plant2_WS_df.shape

Plant2_WS_df.describe()
Plant2_WS_df['DATE_TIME'] 
#convert object into datetime
Plant2_WS_df['DATE_TIME'] = pd.to_datetime(Plant2_WS_df['DATE_TIME'])
Plant2_WS_df['DATE_TIME']
#Extract the datetime
Plant2_WS_df['DATE'] = Plant2_WS_df['DATE_TIME'].dt.date
Plant2_WS_df['DATE']
# total irradiation per day 
Per_day_irradiation = Plant2_WS_df.groupby('DATE')['IRRADIATION'].sum()
print('Total irradiation per day in Plant 2',Per_day_irradiation)
Plant2_WS_df.info()
Plant2_G_df.head(5)

Plant2_G_df.info()
Plant2_G_df.shape
Plant2_G_df.describe()
Plant2_G_df['DATE_TIME'] = pd.to_datetime(Plant2_G_df['DATE_TIME'])
Plant2_G_df['DATE'] = Plant2_G_df['DATE_TIME'].dt.date
Plant2_G_df['DATE']

Plant1_WS_df.head(5)
Plant1_WS_df.info()
Plant1_WS_df.shape
Plant1_WS_df.describe()
Plant1_WS_df['DATE_TIME'] = pd.to_datetime(Plant2_WS_df['DATE_TIME'])
#Extract the datetime
Plant1_WS_df['DATE'] = Plant1_WS_df['DATE_TIME'].dt.date
Plant1_WS_df['DATE']

# total irradiation per day in Plant 1
Per_day_irradiation = Plant1_WS_df.groupby('DATE')['IRRADIATION'].sum()
print('Total irradiation per day in Plant 1',Per_day_irradiation)
Plant1_G_df.info()
Plant1_G_df.shape
Plant1_G_df.describe()
Plant1_G_df['DATE_TIME'] = pd.to_datetime(Plant2_G_df['DATE_TIME'])
#Extract the datetime
Plant1_G_df['DATE'] = Plant1_G_df['DATE_TIME'].dt.date
Plant1_G_df['DATE']
#mean value of daily yeild
Plant1_Daily_Yeild =  Plant1_G_df['DAILY_YIELD'].mean()
print('The average value of daily yeild for Plant 1:',round(Plant1_Daily_Yeild, 3))
Plant2_Daily_Yield = Plant2_G_df['DAILY_YIELD'].mean()
print('The average value of daily yeild for Plant 2:', round(Plant2_Daily_Yield, 3))
#The average daily yield for Plant 1 is slightly higher than Plant2.
#Number of intverters in Plant 1
Total_inverters = Plant1_G_df['SOURCE_KEY'].count()
print("The total number of inverters in Plant 1 is ", Total_inverters)
#Number of intverters in Plant 2
Total_inverters = Plant2_G_df['SOURCE_KEY'].count()
print("The total number of inverters in Plant 2 is ", Total_inverters)
#max/ min of AC/DC power generated in a time interval day wise
#Plant 2
date_wise_DC_PG_min = Plant2_G_df.groupby('DATE')['DC_POWER'].min()
date_wise_DC_PG_max = Plant2_G_df.groupby('DATE')['DC_POWER'].max()
DC_min_max = date_wise_DC_PG_min.to_frame().merge(date_wise_DC_PG_max, on = 'DATE', how= 'outer')
date_wise_AC_PG_min = Plant2_G_df.groupby('DATE')['AC_POWER'].min()
date_wise_AC_PG_max = Plant2_G_df.groupby('DATE')['AC_POWER'].max()
AC_min_max = date_wise_AC_PG_min.to_frame().merge(date_wise_AC_PG_max, on = 'DATE', how= 'outer')
AC_DC_comparison= DC_min_max.merge(AC_min_max, on ='DATE', how = 'outer') 
AC_DC_comparison.rename(columns ={'DC_POWER_x': 'DC_POWER_MIN', 'DC_POWER_y': 'DC_POWER_MAX', 'AC_POWER_x': 'AC_POWER_MIN', 'AC_POWER_y': 'AC_POWER_MAX'}, inplace = True)
print('The min / max of AC/DC power generated in a time interval day wise for Plant 2')
print(AC_DC_comparison)
#Plant 1
date_wise_DC_PG_min = Plant1_G_df.groupby('DATE')['DC_POWER'].min()
date_wise_DC_PG_max = Plant1_G_df.groupby('DATE')['DC_POWER'].max()
DC_min_max = date_wise_DC_PG_min.to_frame().merge(date_wise_DC_PG_max, on = 'DATE', how= 'outer')
date_wise_AC_PG_min = Plant1_G_df.groupby('DATE')['AC_POWER'].min()
date_wise_AC_PG_max = Plant1_G_df.groupby('DATE')['AC_POWER'].max()
AC_min_max = date_wise_AC_PG_min.to_frame().merge(date_wise_AC_PG_max, on = 'DATE', how= 'outer')
AC_DC_comparison= DC_min_max.merge(AC_min_max, on ='DATE', how = 'outer') 
AC_DC_comparison.rename(columns ={'DC_POWER_x': 'DC_POWER_MIN', 'DC_POWER_y': 'DC_POWER_MAX', 'AC_POWER_x': 'AC_POWER_MIN', 'AC_POWER_y': 'AC_POWER_MAX'}, inplace = True)
print('The min / max of AC/DC power generated in a time interval day wise for Plant 1')
print(AC_DC_comparison)

#Inverter producing maximum AC/DC power 
#PLANT 2
Plant2_G_df[['AC_POWER', 'DC_POWER']].idxmax()
print('The invertors with the maximum AC/DC Power Generation in Plant 2 is ')
Plant2_G_df['SOURCE_KEY'][41423]
#Inverter producing maximum AC/DC power
#PLANT1
Plant1_G_df[['AC_POWER', 'DC_POWER']].idxmax()
print('The invertors with the maximum AC/DC Power Generation in Plant 1 is ')
Plant1_G_df['SOURCE_KEY'][61624]
#Missing Data
Plant1_WS_df.isnull()
Plant1_G_df.isnull()
Plant2_WS_df.isnull()
Plant2_G_df.isnull()
#There is no missing data