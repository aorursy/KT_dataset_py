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
#Seeing number of confirmed cases globally

confirmed_cases_world = "/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv"

df4 = pd.read_csv(confirmed_cases_world)

df4.head()



df4["Total_Cases"]=df4[df4.columns[4:]].sum(axis=1)
#Using folium to create a world map for showing number of confirmed cases globally

import folium

country=list(df4.iloc[:,1])

lat=list(df4.iloc[:,2])

long=list(df4.iloc[:,3])

total_cases=list(df4["Total_Cases"])

median1=df4["Total_Cases"].median()

def fill_color(total_cases):

    if total_cases>=median1:

        return "red"

    else:

        return "green"

# print(total_cases)

html = """

Country name:%s<br>

<a href ="https://www.google.com/search?q=%s coronavirus status" target="_blank">%s coronavirus status</a><br>

Total Cases: %s 

"""

map2=folium.Map(location=[0,0],zoom_start=6,tiles="Stamen Terrain")

fg=folium.FeatureGroup(name="My Map")



for i in range(len(lat)):

    iframe=folium.IFrame(html= html %(country[i],country[i],country[i],total_cases[i]),width=200,height=100)

    fg.add_child(folium.CircleMarker([lat[i],long[i]],radius=6,popup=folium.Popup(iframe),fill_color=fill_color(total_cases[i]),color="grey",fill_opacity="0.7"))



map2.add_child(fg)

map2.save("Map2.html")
def confirmed_US_case():

  confirmed_cases_US = "/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv"

  df = pd.read_csv(confirmed_cases_US)

  df = df[df['Country/Region'] == "US"]

  df_new = df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'])

  df_new.rename(columns={"variable":"Date","value":"confirmed_cases"},inplace=True)

  confirmed_per_day = df_new.groupby("Date")['confirmed_cases'].sum()

  confirmed_per_day = confirmed_per_day.reset_index()

  print(confirmed_per_day)

  confirmed_per_day = confirmed_per_day[['Date','confirmed_cases']]

  return confirmed_per_day



confirmed_cases = confirmed_US_case()
confirmed_cases.tail()
#Since we would be using prophet to forecast number of cases, we need to convert data types of our columns.



confirmed_cases.rename(columns={"Date":"ds","confirmed_cases":"y"},inplace=True)

confirmed_cases['ds'] = pd.to_datetime(confirmed_cases['ds'])

confirmed_cases.sort_values(by='ds',inplace=True)





#Plotting number of cases with day

plt_confirmed = confirmed_cases.reset_index()['y'].plot(title="#Confirmed Cases Vs Day");

plt_confirmed.set(xlabel="Date", ylabel="#Confirmed Cases");
#Doing tran test split

train = confirmed_cases[:-4]

test = confirmed_cases[-4:]



test = test.set_index("ds")

test = test['y']
# Model Initialize

from fbprophet import Prophet

m = Prophet()

m.fit(train)

future_dates = m.make_future_dataframe(periods=10)

# Prediction

forecast =  m.predict(future_dates)

pd.plotting.register_matplotlib_converters()

ax = forecast.plot(x='ds',y='yhat',label='Predicted confirmed cases',legend=True,figsize=(12,8))

test.plot(y='y',label='Actual Confirmed case counts',legend=True)


