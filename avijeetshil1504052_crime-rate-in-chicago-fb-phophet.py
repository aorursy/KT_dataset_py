import folium

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from fbprophet import Prophet



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

sns.set_style("darkgrid")



df_1 = pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2005_to_2007.csv',error_bad_lines=False)

df_2 = pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2008_to_2011.csv',error_bad_lines=False)

df_3 = pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv',error_bad_lines=False)

df = pd.concat([df_1, df_2, df_3],ignore_index=False, axis=0)



del df_1

del df_2

del df_3

df.shape


Map = folium.Map(location=[41.864073,-87.706819],

                        zoom_start=11, tiles='Stamen Terrain')





loclist = df.loc[:, ['Latitude', 'Longitude']]



for i in range(len(loclist)):

    x = loclist.iloc[i][0]

    y = loclist.iloc[i][1]

    popup_text = """ <br>"""

    popup_text = popup_text.format(loclist.index[i])

    folium.Marker(location = [x, y], popup= popup_text, fill = True).add_to(Map)
Map
df.isnull().sum().sort_values(ascending=False).keys()
df.drop(['Unnamed: 0','Case Number','ID','IUCR','Y Coordinate', 'X Coordinate','Updated On','FBI Code','Beat','Community Area', 'Ward', 'District','Location', 'Latitude','Longitude'],inplace=True, axis=1)
df.Date=pd.to_datetime(df.Date,format='%m/%d/%Y %I:%M:%S %p')

df.index=pd.DatetimeIndex(df.Date)
df.head()


plt.figure(figsize=(10,10))



df.groupby(['Primary Type']).size().sort_values(ascending=True).plot(kind='barh')

plt.show()
df_primary=df['Primary Type'].value_counts().iloc[:20].index

plt.figure(figsize=(15,10))

sns.countplot(y='Primary Type',data=df,order=df_primary)
plt.figure(figsize=(15,10))

df_loc=df['Location Description'].value_counts().iloc[:20].index

sns.countplot(y='Location Description',data=df,order=df_loc)

plt.title('Most location Description')
plt.plot(df.resample('Y').size())

plt.title('No of Crime per year')

plt.xlabel('Years')

plt.ylabel('Crime')

plt.plot(df.resample('M').size())

plt.title('No of Crime per Month')

plt.xlabel('Month')

plt.ylabel('Crime')

plt.plot(df.resample('Q').size())

plt.title('No of Crime per Quarter')

plt.xlabel('Quarter')

plt.ylabel('Crime')

plt.figure(figsize=(15,15))

plt.plot(df.resample('D').size())

plt.title('No of Crime per day')

plt.xlabel('Day')

plt.ylabel('Crime')

df_prophet=df.resample('M').size().reset_index()

df_prophet.head()
df_prophet.columns=['Date','Crime value']

df_prophet
df_prophet_final=df_prophet.rename(columns={'Date':'ds','Crime value':'y'})
prop= Prophet()

prop.fit(df_prophet_final)

future = prop.make_future_dataframe(periods=1500)

forcast= prop.predict(future)

forcast
figure=prop.plot(forcast, xlabel='Date',ylabel='Crime Rate')
figure=prop.plot_components(forcast)