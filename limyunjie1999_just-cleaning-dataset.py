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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
df_wind_energy_2017 = pd.read_csv('../input/eolien/wind energy.csv')

df_wind_energy_2018 =pd.read_csv('../input/eolien-2018/wind energy_2018.csv')

df_wind_energy_2019 =pd.read_csv('../input/eolien2019/wind energy_2019.csv')

df_wind_energy_2020 =pd.read_csv('../input/eolien-2020/wind_energy_2020.csv')
df_wind_energy = pd.concat([df_wind_energy_2017, df_wind_energy_2018, df_wind_energy_2019], ignore_index=True)
shape_len = df_wind_energy.shape[0]
shape_len_4 = shape_len/4
i=0

while (i<shape_len_4):

    first_num= df_wind_energy['Eolien'][4*i]

    sec_num = df_wind_energy['Eolien'][(4*i)+2]

    average = (first_num + sec_num)/2

    df_wind_energy['Unnamed: 3'][4*i] = average

    print(df_wind_energy['Unnamed: 3'][4*i])

    i = i+1

    first_num=0

    sec_num=0

    
df_wind_energy_edit = df_wind_energy.copy()
i=0

while (i<shape_len_4):

    df_wind_energy_edit.drop([(4*i)+1,(4*i)+2,(4*i)+3],inplace=True)

    #df_wind_energy_edit.drop(4*i+2)

    #df_wind_energy_edit.drop(4*i+3)

    i=i+1

print(df_wind_energy_edit)
df_wind_energy_edit = df_wind_energy_edit.reset_index(drop=True)
df_wind_energy_edit.drop(['Eolien'], axis=1,inplace=True)

df_wind_energy_edit.rename(columns={"Unnamed: 3": "Wind"},inplace=True)
#Start on feature engineering
split_date = df_wind_energy_edit.Date.str.split('/')

split_time = df_wind_energy_edit.Heures.str.split(':')
df_wind_energy_edit["Year"] = ""

df_wind_energy_edit["Month"] = ""

df_wind_energy_edit["Day"] = ""

df_wind_energy_edit["Hour"] = ""

length = df_wind_energy_edit.shape[0]

i=0

while (i<length):

    df_wind_energy_edit['Year'][i] = split_date[i][2]

    df_wind_energy_edit['Month'][i] = split_date[i][1]

    df_wind_energy_edit['Day'][i] = split_date[i][0]

    df_wind_energy_edit['Hour'][i] = split_time[i][0]

    i=i+1
#df["Year_full"] = ""



#length_df=df.shape[0]

#i=0

#while (i<length_df):

#    df['Year_full'][i] = str(20) + str(df['Year'][i])

#    i=i+1
df_wind_energy_edit
from datetime import datetime

df = df_wind_energy_edit.copy()

df["datetime"] = ""

length = df.shape[0]

i=0

while (i<length):

    df["datetime"][i] = str(df["Date"][i]) + str(" ") + str(df["Heures"][i])

    i=i+1

df["Datetime"] = ""

i=0

while (i<length):

    df["Datetime"][i] = datetime.strptime(df["datetime"][i], '%d/%m/%y %H:%M')

    i = i+1



df.drop(['Date', 'Heures','Year', 'Month', 'Day', 'Hour', 'datetime'], axis=1,inplace=True)



df = df.set_index("Datetime")

df.plot()



plt.show()
#df['Wind'].plot(linewidth=0.5);
#df.to_csv("wind_energy.csv")
df
df_wind_speed_direction = pd.read_csv('../input/wind-speed-direction/lvs-pussay.csv')

df_wind_speed_direction.dtypes

pd.to_datetime(df_wind_speed_direction["Time"])

df_wind_speed_direction.shape
df_wind_speed_direction["datetime"] = ""

length_wind_speed_direction = df_wind_speed_direction.shape[0]

i=0

while (i<length_wind_speed_direction):

    df_wind_speed_direction["datetime"][i] = df_wind_speed_direction["Time"][i][:-3]

    i=i+1
from datetime import datetime

df_wind_speed_direction["Datetime"] = ""



i=0

while (i<length_wind_speed_direction):

    df_wind_speed_direction["Datetime"][i] = datetime.strptime(df_wind_speed_direction["datetime"][i], '%Y/%m/%d %H:%M')

    i=i+1
df_wind_speed_direction.drop(['Time', 'datetime'], axis=1,inplace=True)
#r = pd.date_range(start=df_wind_speed_direction.Datetime.hour.min(), end=df_wind_speed_direction.Datetime.hour.max())

#df_wind_speed_direction.set_index('Datetime').reindex(r).fillna(0.0).rename_axis('Datetime').reset_index()

df_wind_speed_direction = df_wind_speed_direction.set_index("Datetime")
#plotting graph

df_wind_speed_direction['Speed(m/s)'].plot(linewidth=0.5);
merge=pd.merge(df,df_wind_speed_direction, how='outer', left_index=True, right_index=True)
merge
test = merge.copy()
df_all = test[:-157]
length_all=df.shape[0]

length_all_6=length_all/6
i=0

while (i<length_all_6):

    speed = df_all['Speed(m/s)'][(6*i)]

    direction=df_all['Direction (deg N)'][(6*i)]

    df_all['Speed(m/s)'][(6*i)+1] = speed

    df_all['Speed(m/s)'][(6*i)+2] = speed

    df_all['Speed(m/s)'][(6*i)+3] = speed

    df_all['Speed(m/s)'][(6*i)+4] = speed

    df_all['Speed(m/s)'][(6*i)+5] = speed

    df_all['Direction (deg N)'][(6*i)+1] = direction

    df_all['Direction (deg N)'][(6*i)+2] = direction

    df_all['Direction (deg N)'][(6*i)+3] = direction

    df_all['Direction (deg N)'][(6*i)+4] = direction

    df_all['Direction (deg N)'][(6*i)+5] = direction

    i=i+1
df_all.to_csv("wind_all.csv")