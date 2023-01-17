# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#Installing windrose to have the wind direction overview

!pip install windrose

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Use seaborn style defaults and set the default figure size

sns.set(rc={'figure.figsize':(11, 4)})

from windrose import WindroseAxes



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing the data needed for the analysis into panda dataframe



df = pd.read_csv('../input/wind-turbine-scada-dataset/T1.csv')
#checking the first 5 set of data in the dataframe

df.head()
#checking if the dataframe contains null

df.isna().sum()
#Covert Data/time to index and drop columns Date/Time

df.index=df['Date/Time']

df.drop(['Date/Time'], axis=1, inplace=True)
#New DataFrame after dropping column Date/Time

df.head()
#plotting each data

cols_plot = ['LV ActivePower (kW)', 'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)','Wind Direction (°)']

axes = df[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
# Plot the data distributions

plt.figure(figsize=(10, 8))

for i in range(4):

    plt.subplot(2, 2, i+1)

    sns.kdeplot(df.iloc[:,i], shade=True)

    plt.title(df.columns[i])

plt.tight_layout()

plt.show()
# Create wind speed and direction variables

ax = WindroseAxes.from_ax()

ax.bar(df['Wind Direction (°)'], df['Wind Speed (m/s)'], normed=True, opening=0.8, edgecolor='white')

ax.set_legend()
#Checking for maximum and minimum value of the wind direction to help in choosing the right binning value

print(df['Wind Direction (°)'].max())

print(df['Wind Direction (°)'].min())
#Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut

#Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html

#df['Wind Speed (m/s) Bin'] = pd.qcut(df['Wind Speed (m/s)'], 4)



 #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html

#df['Wind Direction (°)'] = pd.cut(df['Wind Direction (°)'].astype(int), 45)
#df
#Bining the data by the wind direction

bins_range = np.arange(0,375,30)
print(bins_range)
#Write a short code to map the bins data

def binning(x, bins):

    kwargs = {}

    if x == max(bins):

        kwargs['right'] = True

    bin = bins[np.digitize([x], bins, **kwargs)[0]]

    bin_lower = bins[np.digitize([x], bins, **kwargs)[0]-1]

    return '[{0}-{1}]'.format(bin_lower, bin)
df['Bin'] = df['Wind Direction (°)'].apply(binning, bins=bins_range)
#group the binned data by mean and std

grouped = df.groupby('Bin')

grouped_std = grouped.std()

grouped_mean = grouped.mean()

grouped_mean.head()
#Checking for maximum and minimum value of the windspeed to help in choosing the right binning value

print(df['Wind Speed (m/s)'].max())

print(df['Wind Speed (m/s)'].min())
#Bining the data by the wind direction

bins_range_ws = np.arange(0,26,0.5)
df['Bin'] = df['Wind Speed (m/s)'].apply(binning, bins=bins_range_ws)
#Group by windspeed bin

grouped = df.groupby('Bin')

grouped_std = grouped.std()

grouped_mean = grouped.mean()

grouped_mean
#lets rearrange the index for proper visualisation

step = bins_range_ws[1]-bins_range_ws[0]

new_index = ['[{0}-{1}]'.format(x, x+step) for x in bins_range_ws]

new_index.pop(-1) #We dont need [360-375]...

grouped_mean = grouped_mean.reindex(new_index)
#Rearranged and visulaizing the mean of each windspeed bin 

grouped_mean
#Power Curve Anaylsis

#Theoretical power curve

plt.scatter(df['Wind Speed (m/s)'],df['Theoretical_Power_Curve (KWh)'])

plt.ylabel('Theoretical_Power (KWh)')

plt.xlabel('Wind speed (m/s)')

plt.grid(True)

plt.legend([' Theoretical_Power_Curve'], loc='upper left')

plt.show()
# LV ActivePower (kW) CP_CURVE

plt.scatter(df['Wind Speed (m/s)'],df['LV ActivePower (kW)'])

plt.ylabel('LV ActivePower (kW)')

plt.xlabel('Wind speed (m/s)')

plt.grid(True)

plt.legend([' LV ActivePower (kW) CP_CURVE'], loc='upper left')

plt.show()
#Condition 1

#The first step is the removal of downtime events, which can be identified as near-zero power at high wind speeds.



#Eliminate datas where wind speed is bigger than 3.5 and active power is zero.

new_df=df[((df["LV ActivePower (kW)"]!=0)&(df["Wind Speed (m/s)"]>3.5)) | (df["Wind Speed (m/s)"]<=3.5)]
#Condition 2

new_1 = (new_df[ (new_df['Wind Speed (m/s)'] < 12.5)  | (new_df['LV ActivePower (kW)'] >= 3000) ])
#Condition 3

new_2 = (new_1[ (new_1['Wind Speed (m/s)'] < 9.5)  | (new_1['LV ActivePower (kW)'] >= 1500) ])
#Condition 3

new_3 = (new_2[ (new_2['Wind Speed (m/s)'] < 6.5)  | (new_2['LV ActivePower (kW)'] >= 500) ])
#Theoretical_Power_Curve and Filtered LV ActivePower (kW) CP_CURVE Visualisation

plt.scatter(new_3['Wind Speed (m/s)'],new_3['LV ActivePower (kW)'])

plt.scatter(df['Wind Speed (m/s)'],df['Theoretical_Power_Curve (KWh)'], label='Theoretical_Power_Curve (KWh)')

plt.ylabel('Power (kW)')

plt.xlabel('Wind speed (m/s)')

plt.grid(True)

plt.legend(['Theoretical_Power_Curve and Filtered LV ActivePower (kW) CP_CURVE'], loc='lower right')

plt.show()
#Function to create more feature as WS and  Category

def CP_group(val):

    if val<3.5:

        return 'Region_1'

    elif val> 3.5 and val < 10:

        return 'Region_1.5'

    elif val>10 and val < 15:

        return 'Region_2'

    elif val>15 and val < 23:

        return 'Region_2.5'

    else:

        return 'Region_3'

df['Operational Category']=df['Wind Speed (m/s)'].apply(CP_group)
df.head(5)
#Checking the data type for better understanding

df.dtypes
#Splitting the data into categorical data and float

df_float = df[df.dtypes[df.dtypes == "float"].index]



df_Cat = df[df.dtypes[df.dtypes == "object"].index]
df_float.head(5)
df_Cat.head(5)
#Converting the categorical data into dummy variable for easy analysis

df_Cat = pd.get_dummies(df_Cat)
df_Cat.head(5)
#concatinating the two data type together

Result=df_float.join([df_Cat])
Result.head(5)