# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



from datetime import date, timedelta

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_pgen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

df_pgen2 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

df_pgen3 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')

df_pgen4 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')

df_pgen1['DATE_TIME'] = pd.to_datetime(df_pgen1['DATE_TIME'], format = '%d-%m-%Y %H:%M')

df_pgen1['DATE'] = df_pgen1['DATE_TIME'].apply(lambda x:x.date())

df_pgen1['TIME'] = df_pgen1['DATE_TIME'].apply(lambda x:x.time())

df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE'],format = '%Y-%m-%d')

df_pgen1['HOUR'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.hour

df_pgen1['MINUTES'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.minute

df_pgen2['DATE_TIME'] = pd.to_datetime(df_pgen2['DATE_TIME'], format = '%Y-%m-%d %H:%M:%S')

df_pgen2['DATE'] = df_pgen2['DATE_TIME'].apply(lambda x:x.date())

df_pgen2['TIME'] = df_pgen2['DATE_TIME'].apply(lambda x:x.time())

df_pgen2['DATE'] = pd.to_datetime(df_pgen2['DATE'],format = '%Y-%m-%d')

df_pgen2['HOUR'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.hour

df_pgen2['MINUTES'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.minute

df_pgen3['DATE_TIME'] = pd.to_datetime(df_pgen3['DATE_TIME'], format = '%Y-%m-%d %H:%M:%S')

df_pgen3['DATE'] = df_pgen3['DATE_TIME'].apply(lambda x:x.date())

df_pgen3['TIME'] = df_pgen3['DATE_TIME'].apply(lambda x:x.time())

df_pgen3['DATE'] = pd.to_datetime(df_pgen3['DATE'],format = '%Y-%m-%d')

df_pgen3['HOUR'] = pd.to_datetime(df_pgen3['TIME'],format='%H:%M:%S').dt.hour

df_pgen3['MINUTES'] = pd.to_datetime(df_pgen3['TIME'],format='%H:%M:%S').dt.minute

df_pgen4['DATE_TIME'] = pd.to_datetime(df_pgen4['DATE_TIME'], format = '%Y-%m-%d %H:%M:%S')

df_pgen4['DATE'] = df_pgen4['DATE_TIME'].apply(lambda x:x.date())

df_pgen4['TIME'] = df_pgen4['DATE_TIME'].apply(lambda x:x.time())

df_pgen4['DATE'] = pd.to_datetime(df_pgen4['DATE'],format = '%Y-%m-%d')

df_pgen4['HOUR'] = pd.to_datetime(df_pgen4['TIME'],format='%H:%M:%S').dt.hour

df_pgen4['MINUTES'] = pd.to_datetime(df_pgen4['TIME'],format='%H:%M:%S').dt.minute

r_left= pd.merge(df_pgen2,df_pgen1,on= 'DATE_TIME',how='left')

print(len(df_pgen1['SOURCE_KEY'].unique()))

print(len(df_pgen3['SOURCE_KEY'].unique()))
#Exploring Data 

df_pgen1['DC_POWER'].mean()

df_pgen1[df_pgen1['SOURCE_KEY'] == 'wCURE6d3bPkepu2']['DC_POWER'].mean()

df_pgen1.head()

df_pgen1.tail()

df_pgen1.value_counts()

df_pgen1['DATE_TIME'].value_counts()

df_pgen1.describe()





#Mean Task1 Subtask 1

_, ax = plt.subplots(1, 1, figsize=(16, 9))



ax.plot(df_pgen2.DATE_TIME,df_pgen2.AMBIENT_TEMPERATURE.rolling(window=20).mean(),label='Plant 1',color='#4CB5F5')

ax.plot(df_pgen4.DATE_TIME,df_pgen4.AMBIENT_TEMPERATURE.rolling(window=20).mean(),label='Plant 2',color='#D32D41')







ax.grid()

ax.margins(0.05)

ax.legend()





plt.title('Ambient Temperature for both plants')

plt.xlabel('Date and Time')

plt.ylabel('Ambient Temperature')

plt.show()





print ("Minimum Ambient Temperature for Plant 1= "+ str(df_pgen2.AMBIENT_TEMPERATURE.min()) )

print ("Maximum Ambient Temperature for Plant 1 = "+ str(df_pgen2.AMBIENT_TEMPERATURE.max()) )

print ("Mean Ambient Temperature for Plant 1 = "+ str(df_pgen2.AMBIENT_TEMPERATURE.mean()) )

print()

print ("Minimum Ambient Temperature for Plant 2= "+ str(df_pgen2.AMBIENT_TEMPERATURE.min()) )

print ("Maximum Ambient Temperature for Plant 2 = "+ str(df_pgen2.AMBIENT_TEMPERATURE.max()) )

print ("Mean Ambient Temperature for Plant 2 = "+ str(df_pgen2.AMBIENT_TEMPERATURE.mean()) )

_, ax = plt.subplots(1, 1, figsize=(16, 9))



ax.plot(df_pgen1.DATE_TIME,df_pgen1.AC_POWER.rolling(window=20).mean(),label='Plant 1',color='#4CB5F5')

ax.plot(df_pgen3.DATE_TIME,df_pgen3.AC_POWER.rolling(window=20).mean(),label='Plant 2',color='#D32D41')





ax.grid()

ax.margins(0.05)

ax.legend()





plt.title('AC_POWER for both plants')

plt.xlabel('Date and Time')

plt.ylabel('AC_POWER')

plt.show()
iplot([go.Histogram2dContour(x=df_pgen2.head(10000)['AMBIENT_TEMPERATURE'], 

                             y=df_pgen2.head(10000)['DATE_TIME'], 

                             contours=go.Contours(coloring='heatmap')),

       go.Scatter(x=df_pgen2.head(20000)['AMBIENT_TEMPERATURE'], y=df_pgen2.head(20000)['DATE_TIME'], mode='markers')])


_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(df_pgen1.DATE_TIME,

        df_pgen1.TOTAL_YIELD,

        linewidth='0.1',

        label='TOTAL YIELD'

       )





ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Line Graph for the Total Yield vs Date_Time for 34 days for Plant 1')

plt.xlabel('Date and Time')

plt.ylabel('Total  Yield')

plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(df_pgen3.DATE_TIME,

        df_pgen3.TOTAL_YIELD,

        linewidth='0.1',

        label='TOTAL YIELD'

       )





ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Line Graph for the Total Yield vs Date_Time for 34 days for Plant 2')

plt.xlabel('Date and Time')

plt.ylabel('Total  Yield')

plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

df_subset = df_pgen1





ax.plot(df_subset.DATE_TIME,

        df_subset.DC_POWER/10,

        label='DC_POWER'

        

       )



ax.plot(df_subset.DATE_TIME,

        df_subset.AC_POWER,

        label='AC_POWER'

       )



ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('DC Power and AC Power over 34 Days')

plt.xlabel('Date and Time')

plt.ylabel('Power')

plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

df_subset = df_pgen1[df_pgen1['DATE']=='2020-05-21']





ax.plot(df_subset.DATE_TIME,

        df_subset.DC_POWER/10,

        marker = 'o',

        linestyle='',

        label='DC_POWER'

       )



ax.plot(df_subset.DATE_TIME,

        df_subset.AC_POWER,

        marker = 'o',

        linestyle='',

        label='AC_POWER'

       )



ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('DC Power and AC Power over 34 Days')

plt.xlabel('Date and Time')

plt.ylabel('Power')

plt.show()
iplot([go.Histogram2dContour(x=df_pgen1.head(10000)['AC_POWER'], 

                             y=df_pgen1.head(10000)['DC_POWER'], 

                             contours=go.Contours(coloring='heatmap')),

       go.Scatter(x=df_pgen1.head(20000)['AC_POWER'], y=df_pgen1.head(20000)['DC_POWER'], mode='markers')])
iplot([go.Histogram2dContour(x=df_pgen2.head(10000)['AMBIENT_TEMPERATURE'], 

                             y=df_pgen2.head(10000)['MODULE_TEMPERATURE'], 

                             contours=go.Contours(coloring='heatmap')),

       go.Scatter(x=df_pgen2.head(20000)['MODULE_TEMPERATURE'], y=df_pgen2.head(20000)['MODULE_TEMPERATURE'], mode='markers')])
plt.figure(figsize=(20,10))

plt.plot(df_pgen2['AMBIENT_TEMPERATURE'],df_pgen2['MODULE_TEMPERATURE'],marker = 'o',linestyle='', color='c', alpha=0.25)

plt.title('Scatter Plot for Module Temperature vs Ambient Temp at Plant 1')

plt.xlabel('Ambient Temperature')

plt.ylabel('Module Temperature')

plt.grid()



plt.show()
plt.figure(figsize=(20,10))

plt.plot(df_pgen4['AMBIENT_TEMPERATURE'],df_pgen4['MODULE_TEMPERATURE'],marker = 'o',linestyle='', color='c', alpha=0.25)

plt.title('Scatter Plot for Module Temperature vs Ambient Temp at Plant 1')

plt.xlabel('Ambient Temperature')

plt.ylabel('Module Temperature')

plt.grid()



plt.show()
plt.figure(figsize=(20,10))

plt.plot(df_pgen4['AMBIENT_TEMPERATURE'],df_pgen4['MODULE_TEMPERATURE'],marker = 'o',linestyle='', color='r', alpha=0.25,label='Plant 2')

plt.plot(df_pgen2['AMBIENT_TEMPERATURE'],df_pgen2['MODULE_TEMPERATURE'],marker = 'o',linestyle='', color='c', alpha=0.15,label='Plant 1')

plt.title('Scatter Plot for Module Temperature vs Ambient Temp at Plant 2')

plt.xlabel('Ambient Temperature')

plt.ylabel('Module Temperature')

plt.grid()

plt.legend()

plt.show()


plt.figure(figsize=(20,10))

for date in dates:

    data = df_pgen2[df_pgen2['DATE'] == date][df_pgen2['IRRADIATION']>0]

    plt.plot(data['AMBIENT_TEMPERATURE'],data['MODULE_TEMPERATURE'],marker = 'o',linestyle='',label = pd.to_datetime(date,format = '%Y-%m-%d').date())

plt.legend()
#Plot bar graph of sourcekey vs total yield for a particular inverter

plt.figure(figsize= (20,10))

inv_lst= df_pgen1['SOURCE_KEY'].unique()

plt.bar(inv_lst,df_pgen1.groupby('SOURCE_KEY')['TOTAL_YIELD'].max())





plt.xticks(rotation = 45)

plt.grid()

plt.show()



df_pgen1['AC_POWER'].argmax() 

print("Plant 1:")



print("Maximum Total Yield:", df_pgen1['SOURCE_KEY'].values[df_pgen1['TOTAL_YIELD'].argmax()])

print("Minimum Total Yield:", df_pgen1['SOURCE_KEY'].values[df_pgen1['TOTAL_YIELD'].argmin()])


#Plot bar graph of sourcekey vs total yield for a particular inverter

plt.figure(figsize= (20,10))

inv_lst2= df_pgen3['SOURCE_KEY'].unique()

plt.bar(inv_lst2,df_pgen3.groupby('SOURCE_KEY')['TOTAL_YIELD'].max())



plt.xticks(rotation = 90)

plt.grid()

plt.show()



df_pgen1['AC_POWER'].argmax() 

print("Plant 2:")



print("Maximum Total Yield:", df_pgen3['SOURCE_KEY'].values[df_pgen3['TOTAL_YIELD'].argmax()])



plt.plot(r_left['IRRADIATION'],r_left['DC_POWER'],c='cyan',marker ='o',linestyle='',alpha = 0.07,label ='DC POWER')

plt.legend()

plt.xlabel('irradiation')

plt.ylabel('dc power')

plt.show()
dates = df_pgen2['DATE'].unique()



_, ax = plt.subplots(1, 1, figsize=(18, 9))



for date in dates:

    df_data = df_pgen2[df_pgen2['DATE']==date]



    ax.plot(df_data.AMBIENT_TEMPERATURE,

            df_data.MODULE_TEMPERATURE,

            marker='.',

            linestyle='',

            alpha=.5,

            ms=10,

            label=pd.to_datetime(date,format='%Y-%m-%d').date()

           )



ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Scatter Plot for Module Temperature vs Ambient Temperature for 34 Days for Plant 1')

plt.xlabel('Ambient Temperature')

plt.ylabel('Module Temperature')

plt.show()
das = df_pgen4['DATE'].unique()



_, ax = plt.subplots(1, 1, figsize=(18, 9))



for date in das:

    df_data = df_pgen4[df_pgen4['DATE']==date]



    ax.plot(df_data.AMBIENT_TEMPERATURE,

            df_data.MODULE_TEMPERATURE,

            marker='o',

            linestyle='',

            alpha=.5,

            ms=10,

            label=pd.to_datetime(date,format='%Y-%m-%d').date()

           )



ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Scatter Plot for Module Temperature vs Ambient Temperature for 34 Days for Plant 2')

plt.xlabel('Ambient Temperature')

plt.ylabel('Module Temperature')

plt.show()