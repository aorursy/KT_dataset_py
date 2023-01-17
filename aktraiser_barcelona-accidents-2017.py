# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import os

import  numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.offline as offline

offline.init_notebook_mode()

from plotly import tools

import plotly.tools as tls

import json

import seaborn as sns



import folium

from folium import plugins

from folium.plugins import HeatMap

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_a = pd.read_csv('../input/accidents_2017.csv')
df_a.head(5)
print(df_a.shape)

print(df_a.info())

print(df_a.describe())

print(df_a.columns)
def data_date(x):

    if x == 'January':

        return 1

    if x == 'February':

        return 2

    if x == 'March':

        return 3

    if x == 'April':

        return 4

    if x == 'May':

        return 5

    if x == 'June':

        return 6

    if x == 'July':

        return 7

    if x == 'August':

        return 8

    if x == 'September':

        return 9

    if x == 'October':

        return 10

    if x == 'November':

        return 11

    if x == 'December':

        return 12



df_a['Months'] = df_a['Month'].apply(data_date)

df_a['Year'] = 2017

data = df_a[['Year','Months','Day']]
data.DATE = pd.to_datetime(data)

data.DATE = data.DATE.dt.date

data['Date'] = pd.to_datetime(data.DATE)

print(data.info)
accidents=pd.merge(df_a, data, how='inner')

accident = accidents[['Date','District Name', 'Neighborhood Name', 'Street', 'Weekday',

       'Month', 'Day', 'Hour', 'Part of the day', 'Mild injuries',

       'Serious injuries', 'Victims', 'Vehicles involved', 'Longitude',

       'Latitude']]

df = accident.set_index('Date')

df.head(5)
df.columns
weekdays={}



# Group and sum() by Districts

temp_mildInjury = df.groupby(['District Name'])['Mild injuries'].sum()

temp_seriousInjury = df.groupby(['District Name'])['Serious injuries'].sum()

temp_victims = df.groupby(['District Name'])['Victims'].sum()

temp_vehicles = df.groupby(['District Name'])['Vehicles involved'].sum()

#Make a frame for Districts

District_name = pd.DataFrame([temp_mildInjury] + [temp_seriousInjury] 

                             + [temp_victims] + [temp_vehicles]).T

District_name = District_name.sort_values(by=['Mild injuries'], ascending=False)





#Group and sum() by Street

temp_StreetMildInjury = df.groupby(['Street'])['Mild injuries'].sum()

temp_StreetSeriousInjury = df.groupby(['Street'])['Serious injuries'].sum()

temp_StreetVictims = df.groupby(['Street'])['Victims'].sum()

temp_StreetVehicles = df.groupby(['Street'])['Vehicles involved'].sum()

#Make a frame for Street

Street = pd.DataFrame([temp_StreetMildInjury] + [temp_StreetSeriousInjury] 

                      + [temp_StreetVictims] + [temp_StreetVehicles]).T

Street = Street.sort_values(by='Mild injuries', ascending=False)





#Group and sum() by Date

temp_DateMildInjury = df.groupby(['Date'])['Mild injuries'].sum()

temp_DateSeriousInjury = df.groupby(['Date'])['Serious injuries'].sum()

temp_DateVictims = df.groupby(['Date'])['Victims'].sum()

temp_DateVehicles = df.groupby(['Date'])['Vehicles involved'].sum()

#Make a frame for Date

Date = pd.DataFrame([temp_DateMildInjury] + [temp_DateSeriousInjury] 

                    + [temp_DateVictims] + [temp_DateVehicles]).T







#Group and sum() by Part of the day

temp_TimeofDayMildInjury = df.groupby(['Part of the day'])['Mild injuries'].sum()

temp_TimeofDaySeriousInjury = df.groupby(['Part of the day'])['Serious injuries'].sum()

temp_TimeofDayVictims = df.groupby(['Part of the day'])['Victims'].sum()

temp_TimeofDayVehicles = df.groupby(['Part of the day'])['Vehicles involved'].sum()

#Make a frame by Part of the day

PartoftheDay = pd.DataFrame([temp_TimeofDayMildInjury] + [temp_TimeofDaySeriousInjury] 

                            + [temp_TimeofDayVictims] + [temp_TimeofDayVehicles]).T

PartoftheDay = PartoftheDay.sort_values(by='Mild injuries', ascending=False)





#Value and percentage about Districts

Percent_mildInjury = (District_name['Mild injuries'] /  sum(District_name['Mild injuries']))*100

Percent_seriousInjury = (District_name['Serious injuries'] / sum(District_name['Serious injuries']))*100

Percent_victims = (District_name['Victims'] / sum(District_name['Victims']))*100

Percent_districts = pd.DataFrame([Percent_mildInjury] + [Percent_seriousInjury] 

                                 + [Percent_victims]).T

Percent_districts = Percent_districts.sort_values(by='Mild injuries', ascending=False)





#Value and percentage about Street

Percent_StreetMildInjury = (Street['Mild injuries'] / sum(Street['Mild injuries']))*100

Percent_StreetSeriousInjury = (Street['Serious injuries'] / sum(Street['Serious injuries']))*100

Percent_StreetVictims = (Street['Victims'] / sum(Street['Victims']))*100

Percent_StreetVehicles = (Street['Vehicles involved'] / sum(Street['Vehicles involved']))*100

Percent_street = pd.DataFrame([Percent_StreetMildInjury] + [Percent_StreetSeriousInjury] 

                              + [Percent_StreetVictims] + [Percent_StreetVehicles]).T

Percent_street = Percent_street.sort_values(by='Mild injuries', ascending=False)





#Value and percentage about Date

Percent_DateMildInjury = (Date['Mild injuries'] / sum(Date['Mild injuries']))*100

Percent_DateSeriousInjury = (Date['Serious injuries'] / sum(Date['Serious injuries']))*100

Percent_DateVictims = (Date['Victims'] / sum(Date['Victims']))*100

Percent_DateVehicles = (Date['Vehicles involved'] / sum(Date['Vehicles involved']))*100

Percent_date = pd.DataFrame([Percent_DateMildInjury] + [Percent_DateSeriousInjury] 

                            + [Percent_DateVictims] + [Percent_DateVehicles]).T



#Value and percentage about Day

Percent_TimeofDayMildInjury = (PartoftheDay['Mild injuries'] / sum(PartoftheDay['Mild injuries']))*100

Percent_TimeofDaySeriousInjury = (PartoftheDay['Serious injuries'] / sum(PartoftheDay['Serious injuries']))*100

Percent_TimeofDayVictims = (PartoftheDay['Victims'] / sum(PartoftheDay['Victims']))*100

Percent_TimeofDayVehicles = (PartoftheDay['Vehicles involved'] / sum(PartoftheDay['Vehicles involved']))*100

Percent_Day = pd.DataFrame([Percent_TimeofDayMildInjury] + [Percent_TimeofDaySeriousInjury] + 

                           [Percent_TimeofDayVictims] + [Percent_TimeofDayVehicles]).T

Percent_Day = Percent_Day.sort_values(by='Mild injuries', ascending=False)

print(Percent_date.head(15))
District_name
Mildinjuries = District_name['Mild injuries']

Seriousinjuries = District_name['Serious injuries']

Victims = District_name['Victims']

Vehiclesinvolved = District_name['Vehicles involved']



trace0 = go.Bar(x = Mildinjuries.index,

                y= Mildinjuries.values,

                name = "Mild Injuries",

                marker = dict(color='rgb(236,154,41)'),

                opacity = 0.8

               )



trace1 = go.Bar(x = Seriousinjuries.index,

                y = Seriousinjuries.values,

                name = "Serious Injuries",

                marker = dict(color='rgb(168,32,26)'),

                opacity = 0.8

               )

trace2 = go.Bar(x = Victims.index,

                y = Victims.values,

                name = "Victims",

                marker = dict(color='rgb(23,89,76)'),

                opacity = 0.8

               )

trace3 = go.Bar(x = Vehiclesinvolved.index,

                y = Vehiclesinvolved.values,

                name = 'Vehicles Involved',

                marker = dict(color='rgb(56,90,3)'),

                opacity = 0.8)



data = [trace0,trace1,trace2,trace3]

layout = go.Layout(barmode = 'group',

                   xaxis = dict(tickangle=-20),

                   title="Mild Injuries - Serious Injuries - Victims - Vehicles Involved by Districts",

                      )

fig = go.Figure(data=data,layout=layout)



plotly.offline.iplot(fig)
Percent_districts
Mildinjuries = Percent_districts['Mild injuries']

Seriousinjuries = Percent_districts['Serious injuries']

Victims = Percent_districts['Victims']



trace0 = go.Bar(x = Mildinjuries.index,

                y= Mildinjuries.values,

                name = "Mild Injuries",

                marker = dict(color='rgb(236,154,41)'),

                opacity = 0.8

               )



trace1 = go.Bar(x = Seriousinjuries.index,

                y = Seriousinjuries.values,

                name = "Serious Injuries",

                marker = dict(color='rgb(168,32,26)'),

                opacity = 0.8

               )

trace2 = go.Bar(x = Victims.index,

                y = Victims.values,

                name = "Victims",

                marker = dict(color='rgb(23,89,76)'),

                opacity = 0.8

               )



data = [trace0,trace1,trace2]

layout = go.Layout(barmode = 'group',

                   xaxis = dict(tickangle=-20),

                   title="Percent: Mild Injuries - Serious Injuries - Victims - Vehicles Involved by Districts",

                      )

fig = go.Figure(data=data,layout=layout)



plotly.offline.iplot(fig)
Street.head(15)
Mildinjuries = Street['Mild injuries'].head(10)

Seriousinjuries = Street['Serious injuries'].head(10)

Victims = Street['Victims'].head(10)

Vehiclesinvolved = Street['Vehicles involved'].head(10)





trace0 = go.Bar(x = Mildinjuries.index,

                y= Mildinjuries.values,

                name = "Mild Injuries",

                marker = dict(color='rgb(236,154,41)'),

                opacity = 0.8

               )



trace1 = go.Bar(x = Seriousinjuries.index,

                y = Seriousinjuries.values,

                name = "Serious Injuries",

                marker = dict(color='rgb(168,32,26)'),

                opacity = 0.8

               )

trace2 = go.Bar(x = Victims.index,

                y = Victims.values,

                name = "Victims",

                marker = dict(color='rgb(23,89,76)'),

                opacity = 0.8

               )

trace3 = go.Bar(x = Vehiclesinvolved.index,

                y = Vehiclesinvolved.values,

                name = 'Vehicles Involved',

                marker = dict(color='rgb(56,90,3)'),

                opacity = 0.8)



data = [trace0,trace1,trace2,trace3]

layout = go.Layout(barmode = 'group',

                   xaxis = dict(tickangle=10),

                   title="Mild Injuries - Serious Injuries - Victims - Vehicles Involved by Street",

                      )

fig = go.Figure(data=data,layout=layout)



plotly.offline.iplot(fig)
Percent_street.head(10)
Mildinjuries = Percent_street['Mild injuries'].head(10)

Seriousinjuries = Percent_street['Serious injuries'].head(10)

Victims = Percent_street['Victims'].head(10)

Vehiclesinvolved = Percent_street['Vehicles involved'].head(10)





trace0 = go.Bar(x = Mildinjuries.index,

                y= Mildinjuries.values,

                name = "Mild Injuries",

                marker = dict(color='rgb(236,154,41)'),

                opacity = 0.8

               )



trace1 = go.Bar(x = Seriousinjuries.index,

                y = Seriousinjuries.values,

                name = "Serious Injuries",

                marker = dict(color='rgb(168,32,26)'),

                opacity = 0.8

               )

trace2 = go.Bar(x = Victims.index,

                y = Victims.values,

                name = "Victims",

                marker = dict(color='rgb(23,89,76)'),

                opacity = 0.8

               )

trace3 = go.Bar(x = Vehiclesinvolved.index,

                y = Vehiclesinvolved.values,

                name = 'Vehicles Involved',

                marker = dict(color='rgb(56,90,3)'),

                opacity = 0.8)



data = [trace0,trace1,trace2,trace3]

layout = go.Layout(barmode = 'group',

                   xaxis = dict(tickangle=10),

                   title="Percent : Mild Injuries - Serious Injuries - Victims - Vehicles Involved by Street",

                      )

fig = go.Figure(data=data,layout=layout)



plotly.offline.iplot(fig)
Date.head(10)
Date['Mild injuries'].plot(legend=True,figsize = (20,10))

Date['Serious injuries'].plot(legend=True,figsize = (20,10))

Date['Victims'].plot(legend=True,figsize = (20,10))

Date['Vehicles involved'].plot(legend=True,figsize = (20,10))
sns.pairplot(Date.dropna())
returns_fig = sns.PairGrid(Date.dropna())



returns_fig.map_upper(plt.scatter,color = 'purple')

returns_fig.map_lower(sns.kdeplot, cmap='cool_d')



returns_fig.map_diag(plt.hist,bins=30)
Percent_date.head(10)
Percent_date['Mild injuries'].plot(legend=True,figsize = (20,10))

Percent_date['Serious injuries'].plot(legend=True,figsize = (20,10))

Percent_date['Victims'].plot(legend=True,figsize = (20,10))

Percent_date['Vehicles involved'].plot(legend=True,figsize = (20,10))
PartoftheDay
Mildinjuries = PartoftheDay['Mild injuries']

Seriousinjuries = PartoftheDay['Serious injuries']

Victims = PartoftheDay['Victims']

Vehiclesinvolved = PartoftheDay['Vehicles involved']





trace0 = go.Bar(x = Mildinjuries.index,

                y= Mildinjuries.values,

                name = "Mild Injuries",

                marker = dict(color='rgb(236,154,41)'),

                opacity = 0.8

               )



trace1 = go.Bar(x = Seriousinjuries.index,

                y = Seriousinjuries.values,

                name = "Serious Injuries",

                marker = dict(color='rgb(168,32,26)'),

                opacity = 0.8

               )

trace2 = go.Bar(x = Victims.index,

                y = Victims.values,

                name = "Victims",

                marker = dict(color='rgb(23,89,76)'),

                opacity = 0.8

               )

trace3 = go.Bar(x = Vehiclesinvolved.index,

                y = Vehiclesinvolved.values,

                name = 'Vehicles Involved',

                marker = dict(color='rgb(56,90,3)'),

                opacity = 0.8)



data = [trace0,trace1,trace2,trace3]

layout = go.Layout(barmode = 'group',

                   xaxis = dict(tickangle=5),

                   title="Mild Injuries - Serious Injuries - Victims - Vehicles Involved by Part of the day",

                      )

fig = go.Figure(data=data,layout=layout)



plotly.offline.iplot(fig)
Percent_Day
Mildinjuries = Percent_Day['Mild injuries']

Seriousinjuries = Percent_Day['Serious injuries']

Victims = Percent_Day['Victims']

Vehiclesinvolved = Percent_Day['Vehicles involved']





trace0 = go.Bar(x = Mildinjuries.index,

                y= Mildinjuries.values,

                name = "Mild Injuries",

                marker = dict(color='rgb(236,154,41)'),

                opacity = 0.8

               )



trace1 = go.Bar(x = Seriousinjuries.index,

                y = Seriousinjuries.values,

                name = "Serious Injuries",

                marker = dict(color='rgb(168,32,26)'),

                opacity = 0.8

               )

trace2 = go.Bar(x = Victims.index,

                y = Victims.values,

                name = "Victims",

                marker = dict(color='rgb(23,89,76)'),

                opacity = 0.8

               )

trace3 = go.Bar(x = Vehiclesinvolved.index,

                y = Vehiclesinvolved.values,

                name = 'Vehicles Involved',

                marker = dict(color='rgb(56,90,3)'),

                opacity = 0.8)



data = [trace0,trace1,trace2,trace3]

layout = go.Layout(barmode = 'group',

                   xaxis = dict(tickangle=5),

                   title="Percent : Mild Injuries - Serious Injuries - Victims - Vehicles Involved by Part of the day",

                      )

fig = go.Figure(data=data,layout=layout)



plotly.offline.iplot(fig)
df_a.columns.values

df_a['District Name'].unique()

districts = {}

weekdays = {}

month={}

days = {}

a = 0

for i in df_a['District Name'].unique():

    districts[i] = a

    a = a+1

a = 0

for i in df_a['Weekday'].unique():

    weekdays[i] = a

    a = a+1

a = 0

for i in df_a['Month'].unique():

    month[i] = a

    a = a+1

a = 0

for i in df_a['Part of the day'].unique():

    days[i] = a

    a = a+1

a = 0

df_a['Weekday'] = df_a['Weekday'].apply(lambda x:weekdays[x])

df_a['District Name'] = df_a['District Name'].apply(lambda x:districts[x]) 

df_a['Month'] = df_a['Month'].apply(lambda x:month[x])

df_a['Part of the day'] = df_a['Part of the day'].apply(lambda x:days[x])

df.head(5)
barcelona_coordinates = [41.406141, 2.168594]

accidents_df = df_a

from folium.plugins import HeatMap



map_accidents = folium.Map(location=barcelona_coordinates, tiles='CartoDB Dark_Matter', zoom_start=13)



lat_long_df = accidents_df[['Latitude','Longitude']].as_matrix()



map_accidents.add_child(plugins.HeatMap(lat_long_df))



map_accidents
df_a['killed+injured'] = df_a['Mild injuries'] + df_a['Serious injuries'] + df_a['Victims']

temp_df = df_a.groupby(['District Name'])['killed+injured'].sum()

temp_df['Vehicles involved'] = df_a['Vehicles involved']





data= df_a[['District Name','Month','Weekday','Part of the day','Mild injuries','Serious injuries','Victims', 'Vehicles involved']]

corr = data.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, 

            square=True, linewidths=.5, annot=True, cmap=cmap)

plt.yticks(rotation=0)

plt.title('Correlation Matrix of all Numerical Variables')

plt.show()
df_a["Mild injuries"].value_counts()
df_a.describe()
%matplotlib inline

df_a.hist(bins=50, figsize = (20,15))

plt.show()
np.random.seed(42)
import numpy as np

import numpy.random as rnd





def split_train_test(data, test_ratio):

    shuffled_indices = rnd.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]

    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]
train_set, test_set = split_train_test(df_a, 0.2)
print(len(train_set), "train +" , len(test_set), "test" )
import hashlib



def test_set_check(identifier, test_ratio, hash):

    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio



def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):

    ids = data[id_column]

    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))

    return data.loc[~in_test_set], data.loc[in_test_set]
df_with_id = df_a.reset_index()

train_set, test_set = split_train_test_by_id(df_with_id, 0.2, "index")
df_with_id["id"] = df_a["Longitude"] * 1000 + df_a["Latitude"]

train_set, test_set = split_train_test_by_id(df_with_id, 0.2, "id")
test_set.head()
from sklearn.model_selection import train_test_split



train_set, test_set = train_test_split(df_a, test_size=0.2, random_state=42)
test_set.head()
df_a['Hour'].hist()
df_a.plot(kind="scatter", x="Longitude", y="Latitude")
df_a.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.1)
df_a.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.4,

    s=df["Victims"], label="Victims", figsize=(10,7),

    c="District Name", cmap=plt.get_cmap("jet"), colorbar=True,

    sharex=False)

plt.legend()
import seaborn as sns

df_victims = df_a.loc[(df_a['Victims']!=0)]



sns.lmplot('Longitude', 

           'Latitude',

           data=df_victims[:],

           fit_reg=False, 

           hue="District Name",

           palette='Dark2',

           size=12,

           ci=2,

           scatter_kws={"marker": "D", 

                        "s": 10})

ax = plt.gca()

ax.set_title("All Crime Distribution per District")
g = sns.lmplot(x="Longitude",

               y="Latitude",

               col="Victims",

               data=df_victims.dropna(), 

               col_wrap=2, size=6, fit_reg=False, 

               sharey=False,

               scatter_kws={"marker": "D",

                            "s": 10})
barcelona_coordinates = [41.406141, 2.168594]

accidents_df = df_a.head(100)

df_victims = df.loc[(df['Victims']!=0)]

from folium.plugins import HeatMap



map_accidents = folium.Map(location=barcelona_coordinates, tiles='cartodbpositron', zoom_start=13)



lat_long_df = df_victims[['Latitude','Longitude']].as_matrix()





map_accidents.add_child(plugins.HeatMap(lat_long_df))



map_accidents
