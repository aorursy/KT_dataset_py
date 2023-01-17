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
import numpy as np
import matplotlib.pyplot as plt
import datetime 
df= pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df2= pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df

df2


type(df)
print (df.describe())
print (df2.describe())

print (df2.nunique())
print (df.nunique())
if 'HmiyD2TTLFNqkNe'in df['SOURCE_KEY']:
    print ('t')
else:
    print ('f')

totalir= 0
for i in df2['IRRADIATION']:
    totalir += i
print ( "irraditiation per day=" , totalir/34)

df[df['DC_POWER']== 14471.125000]     
df[df['AC_POWER']== 1410.950000]     

xx = df[['SOURCE_KEY' , 'DC_POWER']]
xx.sort_values('DC_POWER', ascending= True)

xx= df['SOURCE_KEY'].unique()
xx 

xx = df[['SOURCE_KEY' , 'AC_POWER']]
xx.sort_values('AC_POWER', ascending= True)
df['DATE_TIME']= pd.to_datetime(df['DATE_TIME'],format = '%d-%m-%Y %H:%M')
df.info()
df['DATE']= pd.to_datetime(df['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.date
df.info()
df['TIME']= pd.to_datetime(df['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.time
df.info()
df['DATE']=pd.to_datetime(df['DATE'])
df['TIME']=pd.to_datetime(df['TIME'], format = '%H:%M:%S')
df.info()
date_unique = df['DATE'].value_counts()           #to find missing values 
date_unique
df['DATE']







