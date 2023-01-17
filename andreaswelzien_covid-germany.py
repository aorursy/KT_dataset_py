import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.dates as mdates
def plot(plottype,dataframe, Title, Y_Lable, column):

    fig, ax = plt.subplots(figsize=(15,7))

    if plottype == 'line':

        ax.plot(dataframe['date'], dataframe[column], color='tab:blue', marker='o')

    elif plottype == 'bar':

        ax.bar(dataframe['date'], dataframe[column])

    ax.xaxis.set_major_locator(mdates.WeekdayLocator())

    ax.set(xlabel="Datum",

           ylabel= Y_Lable,           title = Title)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    plt.show()

    
df = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv", parse_dates = ['Date'])
df.info()
print('Der erste Eintrag datiert vom:',df['Date'].iloc[0])

print('Der letzte Eintrag datiert vom:',df['Date'].iloc[-1])
#Renaming the coulmns for easy usage

df.rename(columns={'Date': 'date', 

                     'Province/State':'state',

                     'Country/Region':'country',

                     'Lat':'lat', 'Long':'long',

                     'Confirmed': 'confirmed',

                     'Deaths':'deaths',

                     'Recovered':'recovered'

                    }, inplace=True)



df['active'] = df['confirmed'] - df['deaths'] - df['recovered']

df['mortality'] = df['deaths'] / df['confirmed'] * 100
germany =  df[df.country == 'Germany']

germany = germany.groupby(by = 'date')['recovered', 'deaths', 'confirmed', 'active', 'mortality'].sum().reset_index()

germany
plot('line',germany,'Gesamtsumme der bestätigten Fälle', '', 'confirmed')
germany['new_confirmed'] = germany['confirmed'].shift(-1) - germany['confirmed'] 

plot('bar',germany,'Neu identifizierte Infektionen pro Tag', '', 'new_confirmed')
plot('line',germany,'Summe der Toten', '', 'deaths')

germany['new_deaths'] = germany['deaths'].shift(-1) - germany['deaths'] 

plot('bar',germany,'Neu Todesfälle pro Tag', '', 'new_deaths')
print(f'Die Sterblichkeitsraten in Deutschland beträgt {germany["mortality"].iloc[-1]:.2f}%')
lebensbedrohlich = 5 #in %

zeitdelta_lebensbedrohlich = 6

behandlungsdauer = 10

germany['lebensbedrohlich'] = germany['confirmed'].shift(zeitdelta_lebensbedrohlich) * lebensbedrohlich / 100

germany['benötigte Betten'] = germany['lebensbedrohlich'] - germany['lebensbedrohlich'].shift(behandlungsdauer) 

plot('line',germany,'Benötigte Intensivbetten', '', 'benötigte Betten')
bev_altersklassen = [7760, 7537, 9573, 10940, 10105, 13331, 10741, 7456, 5074, 832, 0]

mort_altersklassen = [0, 0, 0, 0.1, 0.1, 0.5, 1.6, 6.3, 11.6, 11.6]

anz_sterbefälle = [x*y/100 for x,y in zip(bev_altersklassen, mort_altersklassen)]

_, ax = plt.subplots(figsize=(15,7))

ax.bar(range(10), anz_sterbefälle)

ax.set(xlabel="Altersklasen",title = 'Prognostizierte Todesfälle je Altersklassen')

for i, v in enumerate(anz_sterbefälle):

   ax.text(i-.25, v+5, f'{v*0.7*1000:,.0f}', fontsize=12, color='black')

plt.show()