from pandas_profiling import ProfileReport
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid') 



import plotly



pd.set_option('display.max_columns', 100) #show all columns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/us-accidents/US_Accidents_Dec19.csv')

data.head()
# F to C

data['Temperature(C)'] = data['Temperature(F)'].apply(lambda x: format((x - 32) * 5/9, '.2f') if x is not None else np.nan)

data['Wind_Chill(C)'] = data['Wind_Chill(F)'].apply(lambda x: format((x - 32) * 5/9, '.2f') if x is not None else np.nan)



# Inch to mm

data['Precipitation(mm)'] = data['Precipitation(in)'].apply(lambda x: format(x * 25.400, '.2f') if x is not None else np.nan)

data['Pressure(mm)'] = data['Pressure(in)'].apply(lambda x: format(x * 25.400, '.2f') if x is not None else np.nan)



# Miles to kilometrs

data['Wind_Speed(kmh)'] = data['Wind_Speed(mph)'].apply(lambda x: format(x * 1.609344, '.2f') if x is not None else np.nan)

data['Visibility(km)'] = data['Visibility(mi)'].apply(lambda x: format(x * 1.609344, '.2f') if x is not None else np.nan)



data['Distance(km)'] = data['Distance(mi)'].apply(lambda x: format(x * 1.609344, '.2f') if x is not None else np.nan)
#deal with dates



data['Start_Time'] = pd.to_datetime(data['Start_Time'])

data['End_Time'] = pd.to_datetime(data['End_Time'])



data['StartMonth'] = data['Start_Time'].dt.month

data['EndMonth'] = data['End_Time'].dt.month



data['StartDay'] = data['Start_Time'].dt.weekday_name

data['EndDay'] = data['End_Time'].dt.weekday_name



data['StartHour'] = data['Start_Time'].dt.hour

data['EndHour'] = data['End_Time'].dt.hour



data['StartMinute'] = data['Start_Time'].dt.minute

data['EndMinute'] = data['End_Time'].dt.minute



data['StartSecond'] = data['Start_Time'].dt.second

data['EndSecond'] = data['End_Time'].dt.second
data.head()
import plotly.express as px

df = data.groupby(['Side', 'City']).count().xs('ID', axis=1, drop_level=True).reset_index().sort_values(by='ID', ascending=False)

fig = px.bar(df, x='City', y='ID', color='Side')

fig.show()
plt.figure(figsize=[20, 8])

plt.bar(*zip(*data['City'].value_counts()[:10].items()))

plt.show()
plt.figure(figsize=[20, 8])

plt.bar(*zip(*data['City'].value_counts()[-10:].items()))

plt.show()
sns.countplot(x='Side', hue='Severity', data=data)
plt.figure(figsize=[10, 8])

days = ['Monday','Tuesday', 'Wednesday', 'Thursday','Friday', 'Saturday', 'Sunday']

for num, day in enumerate(days):

    df1 = data.loc[data['StartDay'] == day]

    df2 = data.loc[data['EndDay'] == day]

    

    plt.figure(num)

    plt.subplot(121)

    #plt.hist(df1['StartHour'])

    sns.countplot(df1['StartHour'])

    plt.title(day)

    

    plt.subplot(122)

    #plt.hist(df2['EndHour'])

    sns.countplot(df2['EndHour'])

    plt.title(day)
#How much accidents ends at next day

next_day_end = data.loc[data['StartDay'] != data['EndDay']]

print(len(next_day_end))
# next month

len(data.loc[data['StartMonth'] != data['EndMonth']])
#small sample for speedup autopandas and geoplot

data = data.sample(1000)
autoreport = ProfileReport(data, minimal=True, title='Auto Report', html={'style':{'full_width':True}})
autoreport
autoreport.to_file(output_file="autoreport.html")
import plotly.graph_objects as go

fig = go.Figure(data=go.Scattergeo(

        lon = data['Start_Lng'],

        lat = data['Start_Lat'],

        mode = 'markers',

        ))



fig.update_layout(

        title = 'Accident location',

        geo_scope='usa',

    )

fig.show()