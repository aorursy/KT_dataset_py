#import libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from pylab import rcParams

from scipy.signal import find_peaks_cwt

import plotly.express as px

import plotly.io
#set some display modifications 

%matplotlib inline

rcParams['figure.figsize'] = 8,11

#plt.style.use('seaborn-whitegrid')

pd.set_option('display.max_rows', 162)
#dailydata = pd.read_csv('Covid-19/TurkeyCovid19Dataset.csv')

datatr = pd.read_csv('../input/covid19-in-turkey/covid_19_data_tr.csv')

test_numbers = pd.read_csv('../input/covid19-in-turkey/test_numbers.csv')

#confirmed = pd.read_csv('Covid-19/time_series_covid_19_confirmed_tr.csv')

intubated = pd.read_csv('../input/covid19-in-turkey/time_series_covid_19_intubated_tr.csv')

#Drop unnecesary columns

intubated.drop(['Province/State', 'Lat','Long','Country/Region'], axis=1, inplace=True)

#Convert ıntubated dataset from wider to longer for add general dataset

intubated = intubated.unstack().reset_index()

intubated = intubated.rename(columns={0:'Intubated'})

intubated.drop(['level_1','level_0'],axis=1, inplace=True)

intubated.tail()
#Drop unnecesary columns

test_numbers.drop(['Province/State', 'Lat','Long','Country/Region'], axis=1, inplace=True)

#Convert test numbers dataset from wider to longer for add general dataset

test_numbers = test_numbers.unstack().reset_index()

test_numbers = test_numbers.rename(columns={0:'Test_numbers'})

test_numbers.drop(['level_1','level_0'],axis=1, inplace=True)

test_numbers.tail()
#Drop unnecesary columns

datatr.drop(['Province/State','Country/Region'], axis=1, inplace=True)

datatr.tail()
#Change columns order

cols = ['Test_numbers', 'Confirmed','Deaths','Recovered','Intubated','Last_Update']

df = pd.concat([datatr, test_numbers, intubated], axis=1)

df = df[cols]

df.tail()
#Date was string, convert it into DateTime and set index

df.set_index('Last_Update', inplace=True)

df.index = pd.to_datetime(df.index)

df.tail()
#In dataset value in Confirmed, Deaths and Recovered are aggregated, not daily

#In this piece of codes make Daily value for each day

df['DailyConfirmed'] = pd.DataFrame(df['Confirmed'].diff().fillna(0).astype(int))

df['DailyDeaths'] = pd.DataFrame(df['Deaths'].diff().fillna(0).astype(int))

df['DailyRecovered'] = pd.DataFrame(df['Recovered'].diff().fillna(0).astype(int))
#Added new value until 31 September

# yeniveri = pd.read_csv("Dataset/güncelveri.txt")

# yeniveri.set_index('Last_Update', inplace=True)

# yeniveri.index = pd.to_datetime(yeniveri.index)
#concatted old dataframe

#df = pd.concat([df,yeniveri])
#Intubated number for each day but when increase

#below 0 is eliminated

df['DailyIntubated'] = pd.DataFrame(df['Intubated'].diff().fillna(0).astype(int))

df['DailyIntubated'] = df['DailyIntubated'].apply(lambda x: 0 if x<0 else x)
#Difference of Intubated number

df['General DailyIntubated'] = pd.DataFrame(df['Intubated'].diff().fillna(0).astype(int))
#with diff function first row in DailyConfirmed was eliminated

#it was fixed

df['DailyConfirmed'][0] = 1
#Made additional columns

df['Total Death/Recovered'] = pd.DataFrame(((df['Deaths'] / df['Recovered']) ).replace([np.inf, -np.inf], np.nan).fillna(0))

df['Total Death/Recovered'] = pd.DataFrame(((df['Deaths'] / df['Recovered']) ).replace([np.inf, -np.inf], np.nan).fillna(0))

df['Total Confirmed/Recovered'] = pd.DataFrame(((df['Confirmed'] / df['Recovered']) ).replace([np.inf, -np.inf], np.nan).fillna(0))

df['Daily Death/Recovered'] = pd.DataFrame(((df['DailyDeaths'] / df['DailyRecovered']) ).replace([np.inf, -np.inf], np.nan).fillna(0))

df['Daily Confirmed/Recovered'] = pd.DataFrame(((df['DailyConfirmed'] / df['DailyRecovered'])).replace([np.inf, -np.inf], np.nan).fillna(0))

df['Daily Intubated/Recovered'] = pd.DataFrame(((df['DailyIntubated'] / df['DailyRecovered'])).replace([np.inf, -np.inf], np.nan).fillna(0))

df['Daily Death/Intubated'] = pd.DataFrame(((df['DailyDeaths'] / df['DailyIntubated'])).replace([np.inf, -np.inf], np.nan).fillna(0))

df['Daily Confirmed/Test'] = pd.DataFrame(((df['DailyConfirmed'] / df['Test_numbers'])).replace([np.inf, -np.inf], np.nan).fillna(0))
df = pd.read_csv("../input/covid19turkey-dataset/31SeptData.csv")

df.set_index('Last_Update', inplace=True)

df.index = pd.to_datetime(df.index)

df.tail(10)
#Quick overview all parameters by date

df.plot(subplots=True)

plt.show()
#Plotting confirmed number

confirmed_fig = px.line(x=df.index,y=df['DailyConfirmed'], color=px.Constant("Daily Confirmed"),

             labels=dict(x="Date", y="# of Daily Confirmed", color="Time Period"),

            title = 'Daily Confirmed Trend').update_layout(plot_bgcolor='aqua')

confirmed_fig.show()
#Plotting deaths number

death_fig = px.line(x=df.index,y=df['DailyDeaths'], color=px.Constant("Daily Deaths"),

             labels=dict(x="Date", y="# of Daily Deaths", color="Time Period"),

            title = 'Daily Deaths Trend')

death_fig.add_scatter(x=df.index,y=df['Daily Death/Intubated'], name="Daily Death/Intubated")

death_fig.show()
#Confirmed-Recovered ratio and Deaths-Intubated ratio is important

#It shows us how was going pandemic what is trend

#for to say pandemic under control Confirmed-Recovered ratio should be under one is required

daily_con_rec_death_int = px.line(x=df.index,y=df['Daily Confirmed/Recovered'], color=px.Constant("Daily Confirmed/Recovered"),

             labels=dict(x="Date", y="Daily Ratio", color="Time Period"),

            title = 'Daily Confirmed/Recovered and Death/Intubated Ratio Trend')

daily_con_rec_death_int.add_scatter(x=df.index,y=df['Daily Death/Intubated'], name="Daily Death/Intubated")

daily_con_rec_death_int.show()
#plotting number of intubated for each day which has increase

int_num_fig = px.line(x=df.index,y=df['DailyIntubated'], color=px.Constant("# of Intubated"),

             labels=dict(x='Date', y="# of intubated", color="Time Period"),

                title = '# of Intubated')

int_num_fig.show()
#plotting number of test and number of confirmed patients 

fig = px.line(x=df.index,y=df['DailyConfirmed'], color=px.Constant("Daily Confirmed"),

             labels=dict(x="Date", y="# of Test and Confirmed", color="Time Period"),

            title = 'Number of test and confirmed')

fig.add_bar(x=df.index,y=df['Test_numbers'], name="Test Numbers")

fig.show()
#Plotting intubated patients number day by day

int_fig = px.bar(x=df.index,y=df['General DailyIntubated'], color=px.Constant("# of Intubated"),

             labels=dict(x='Last_Update', y="Intubated Trend", color="Time Period"),

                title = 'Intubated Trend day by day').update_layout(plot_bgcolor = 'gray')

int_fig.show()
#Plotting intubated patients number day by day

dailyTestConf = px.line(x=df.index,y=df['Daily Confirmed/Test']*100,

             labels=dict(x='Last_Update', y="Daily Confirmed/Test Ratio"),

                title = 'Daily Confirmed/Test Ratio Trend').update_layout(plot_bgcolor = 'cornsilk')

dailyTestConf.show()
#Türkiye GeoJSON dosyası https://github.com/cihadturhan/tr-geojson adresinden alınmıştır.

import json

with open('../input/trcitiesgeojson/tr-cities-utf8.json', encoding="utf-8") as response:

    counties = json.load(response)

counties["features"][0]
eachcitydf = pd.read_csv('../input/number-of-cases-in-the-city-covid19-turkey/number_of_cases_in_the_city.csv')

#Listede 'id' sütunu yoktu Geojson ile eşleştirme yapmak için gerekli.

eachcitydf["id"] = list(range(1,82))

eachcitydf["id"] = eachcitydf["id"].astype(str)
#Turkey map

#Şehir Şehir korona virüs vaka sayısı güncel değildir.

#Sadece 'choropleth' ile etkileşimli harita nasıl yapılır gösterilmek için yapılmıştır.

import plotly.express as px

fig = px.choropleth(eachcitydf, geojson=counties, color='Number of Case',

                            color_continuous_scale="Viridis_R",            

                            range_color=(0,300),

                            locations = "id", #GeoJson ile eşleşecek anahtar kelime

                            scope="asia",

                            labels={'Number of Case':'# of Case'},

                            #featureidkey="properties.name",

                            center = {"lat": 39.925533, "lon": 32.866287},

                            projection="mercator",)



fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.update_geos(fitbounds="locations", visible=False) #Komşu ülkeler görünmez hale getirilip lokasyonlar odak hale getirildi.

fig.show()
#* 23/08/2020 tarihi dahil toplam sayı, ** 17/08/2020 – 23/08/2020 tarihleri arasındaki toplam sayı

regiondf = pd.read_csv('../input/ibbs1-covid19-turkey/blgeverileri.txt')

regiondf
#Sütun adlarındaki boşlukları silmek için.

regiondf.rename(columns=lambda x: x.strip(), inplace=True)
regiondf[['Toplam Vaka Sayısı*', 'Vaka/ 100.000 Nüfus',

       'Son 7 Gün Yeni Vaka Sayısı**', '7 Gün İnsidansı (100.000 Nüfusta)',

       'Önceki Haftaya Göre Değişim (%)']].apply(pd.to_numeric)
#plotting number of test and number of confirmed patients 

fig = px.bar(x=regiondf["İBBS-1"],y=regiondf["Toplam Vaka Sayısı*"],

             labels=dict(x="Regions", y="Total # of Confirmed", color="İBBS-1"),

            title = 'Total # of Confirmed')

fig.update_layout(xaxis = {'categoryorder':'total descending'}, plot_bgcolor='lightgray',)

fig.show()
#plotting number of test and number of confirmed patients 

fig = px.bar(x=regiondf["İBBS-1"],y=regiondf['Son 7 Gün Yeni Vaka Sayısı**'],

             labels=dict(x="Regions", y="# of Confirmed", color="İBBS-1"),

            title = 'New cases in Last 7 Days for each region**',)

fig.update_layout(xaxis = {'categoryorder':'total descending'}, plot_bgcolor='goldenrod')

fig.show()
#plotting number of test and number of confirmed patients 

fig = px.bar(x=regiondf["İBBS-1"],y=regiondf['Önceki Haftaya Göre Değişim (%)'],

             labels=dict(x="Regions", y="Percentage of Change"),

            title = 'Change Compared to the Previous Week (%)',

            color_discrete_sequence=px.colors.qualitative.G10)

fig.update_layout(xaxis = {'categoryorder':'total descending'}, plot_bgcolor='white')

fig.show()