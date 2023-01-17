# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

plt.style.use('fivethirtyeight')
# importing datasets





full_table = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates = ['ObservationDate'])
#world = full_table.groupby(['ObservationDate','Country/Region'])['Confirmed','Deaths'].sum().to_frame().reset_index()

#world['Death Rate'] = (world['Deaths']/world['Confirmed'])*100

#world['Recovery Rate'] = (world['Recovered']/world['Confirmed'])*100

#world.sort_values(['Confirmed'],inplace=True,ascending=False)

#world.head(40)
#plt.plot(world['ObservationDate'],world['Confirmed'],label='Death Rate in United States')
italy = pd.DataFrame(full_table[full_table['Country/Region']=='Italy'])

france = pd.DataFrame(full_table[full_table['Country/Region']=='France'])

germany = pd.DataFrame(full_table[full_table['Country/Region']=='Germany'])

uk = pd.DataFrame(full_table[full_table['Country/Region']=='UK'])

spain = pd.DataFrame(full_table[full_table['Country/Region']=='Spain'])

turkey = pd.DataFrame(full_table[full_table['Country/Region']=='Turkey'])
italy.head()
france.sample(10)
fig, ax = plt.subplots(figsize=(15,7))



ax.set_xlabel("Observation Date")

ax.set_ylabel("Count of Confirmed Positive Cases, Deaths, and Recoveries")



italy[['ObservationDate','Confirmed','Deaths','Recovered']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus in Italy")

fig, ax = plt.subplots(figsize=(15,7))

ax.set_xlabel("Observation Date")

ax.set_ylabel("Count of Confirmed Positive Cases, Deaths, and Recoveries")



germany[['ObservationDate','Confirmed','Deaths','Recovered']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus in Germany")

fig, ax = plt.subplots(figsize=(15,7))

ax.set_xlabel("Observation Date")

ax.set_ylabel("Count of Confirmed Positive Cases, Deaths, and Recoveries")



spain[['ObservationDate','Confirmed','Deaths','Recovered']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus in Spain")



fig, ax = plt.subplots(figsize=(15,7))

ax.set_xlabel("Observation Date")

ax.set_ylabel("Count of Confirmed Positive Cases, Deaths, and Recoveries")



italy[['ObservationDate','Confirmed','Deaths']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus cases, deaths in Italy, Germany and Spain")

germany[['ObservationDate','Confirmed','Deaths']].plot(x='ObservationDate',kind='line',ax=ax)

spain[['ObservationDate','Confirmed','Deaths']].plot(x='ObservationDate',kind='line',ax=ax)

ax.legend(['Confirmed Cases in Italy','Confirmed Deaths in Italy',

           'Confirmed Cases in Germany','Confirmed Deaths in Germany',

          'Confirmed Cases in Spain','Confirmed Deaths in Spain'])

fig, ax = plt.subplots(figsize=(15,7))

ax.set_xlabel("Observation Date")

ax.set_ylabel("Count of Confirmed Positive Cases, Deaths, and Recoveries")



turkey[['ObservationDate','Confirmed','Deaths','Recovered']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus in Turkey")
italy['Death Rate in Italy'] = ((italy['Deaths']/italy['Confirmed'])*100)

germany['Death Rate in Germany'] = (germany['Deaths']/germany['Confirmed'])*100

spain['Death Rate in Spain'] = (spain['Deaths']/spain['Confirmed'])*100

turkey['Death Rate in Turkey'] = (turkey['Deaths']/turkey['Confirmed'])*100

#Recoveries

italy['Recovery Rate in Italy'] = ((italy['Recovered']/italy['Confirmed'])*100)

germany['Recovery Rate in Germany'] = (germany['Recovered']/germany['Confirmed'])*100

spain['Recovery Rate in Spain'] = (spain['Recovered']/spain['Confirmed'])*100

turkey['Recovery Rate in Turkey'] = (turkey['Recovered']/turkey['Confirmed'])*100



#europe = pd.DataFrame(italy[['ObservationDate','Death Rate in Italy']]) 

#europe[]

#print(europe)
uk_agg=pd.pivot_table(uk, index=['ObservationDate'],values=['Confirmed','Deaths','Recovered'],aggfunc=np.sum)

#uk_agg

fig, ax = plt.subplots(figsize=(15,7))

plt.plot(uk_agg.index,uk_agg.values)

plt.legend(['Confirmed','Deaths','Recovered'])

plt.title("Tracking Corona Virus in United Kingdom")

plt.ylabel('Count of Confirmed Positive Cases, Deaths, and Recoveries')

plt.xticks(rotation=90)
uk_agg['Death Rate in United Kingdom'] = (uk_agg['Deaths']/uk_agg['Confirmed'])*100

uk_agg['Recovery Rate in United Kingdom'] = (uk_agg['Recovered']/uk_agg['Confirmed'])*100

uk_agg.tail()
uk_agg.tail(10)
uk2= uk_agg.unstack()

uk2
#uk2[['ObservationDate','Death Rate in United Kingdom']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus in Italy, Germany and Spain")
fig, ax = plt.subplots(figsize=(15,7))

# Set the x-axis label

ax.set_xlabel("Observation Date")

ax.set_ylabel("Death Rate(%)")

italy[['ObservationDate','Death Rate in Italy']].plot(x='ObservationDate',kind='line',ax=ax)

#fig, ax = plt.subplots(figsize=(15,7))

#ax.set_xlabel("Observation Date")

#ax.set_ylabel("Death Rate(%)")



germany[['ObservationDate','Death Rate in Germany']].plot(x='ObservationDate',kind='line',ax=ax)

#fig, ax = plt.subplots(figsize=(15,7))

#ax.set_xlabel("Observation Date")

#ax.set_ylabel("Death Rate(%)")



#uk_agg[['ObservationDate','Death Rate in United Kingdom']].plot(x='ObservationDate',kind='line',ax=ax)



spain[['ObservationDate','Death Rate in Spain']].plot(x='ObservationDate',kind='line',ax=ax, title="Comparing Corona Virus in Italy, Germany and Spain")



turkey[['ObservationDate','Death Rate in Turkey']].plot(x='ObservationDate',kind='line',ax=ax, title="Comparing Corona Virus in Italy, Germany, Turkey and Spain")

fig, ax = plt.subplots(figsize=(15,7))

# Set the x-axis label

ax.set_xlabel("Observation Date")

ax.set_ylabel("Recovery Rate(%)")

italy[['ObservationDate','Recovery Rate in Italy']].plot(x='ObservationDate',kind='line',ax=ax)

#fig, ax = plt.subplots(figsize=(15,7))

#ax.set_xlabel("Observation Date")

#ax.set_ylabel("Death Rate(%)")



germany[['ObservationDate','Recovery Rate in Germany']].plot(x='ObservationDate',kind='line',ax=ax)

#fig, ax = plt.subplots(figsize=(15,7))

#ax.set_xlabel("Observation Date")

#ax.set_ylabel("Death Rate(%)")



#uk_agg[['ObservationDate','Death Rate in United Kingdom']].plot(x='ObservationDate',kind='line',ax=ax)



spain[['ObservationDate','Recovery Rate in Spain']].plot(x='ObservationDate',kind='line',ax=ax, title="Comparing Corona Virus in Italy, Germany and Spain")



turkey[['ObservationDate','Recovery Rate in Turkey']].plot(x='ObservationDate',kind='line',ax=ax, title="Comparing Corona Virus in Italy, Germany, Turkey and Spain")

france_agg=pd.pivot_table(france, index=['ObservationDate'],values=['Confirmed','Deaths','Recovered'],aggfunc=np.sum)

fig, ax = plt.subplots(figsize=(15,7))

plt.plot(france_agg.index,france_agg.values)

#france_agg

plt.legend(['Confirmed','Deaths','Recovered'])

plt.title("Tracking Corona Virus in France")

plt.ylabel('Count of Confirmed Positive Cases, Deaths, and Recoveries')

plt.xticks(rotation=90)
france_agg['Death Rate in France'] = (france_agg['Deaths']/france_agg['Confirmed'])*100

france_agg['Recovery Rate in France'] = (france_agg['Recovered']/france_agg['Confirmed'])*100

france_agg.tail()
us = full_table[full_table['Country/Region']=='US']

#us.sample(10)

us_agg=pd.pivot_table(us, index=['ObservationDate'],values=['Confirmed','Deaths','Recovered'],aggfunc=np.sum)

fig, ax = plt.subplots(figsize=(15,10))

plt.plot(us_agg.index,us_agg.values)

plt.legend(['Confirmed','Deaths','Recovered'])

plt.title("Tracking Corona Virus in United States")

plt.ylabel('Count')

plt.xticks(rotation=90)
# Italy

italy_agg=pd.pivot_table(italy, index=['ObservationDate'],values=['Confirmed','Deaths','Recovered'],aggfunc=np.sum)



# Spain

spain_agg=pd.pivot_table(spain, index=['ObservationDate'],values=['Confirmed','Deaths','Recovered'],aggfunc=np.sum)



# Plotting

fig, ax = plt.subplots(figsize=(15,7))

plt.plot(italy_agg.index,italy_agg['Confirmed'],label='Confirmed Cases in Italy')

plt.plot(spain_agg.index,spain_agg['Confirmed'],label='Confirmed Cases in Spain')

plt.plot(us_agg.index,us_agg['Confirmed'],label='Confirmed Cases in United States')

plt.plot(uk_agg.index,uk_agg['Confirmed'],label='Confirmed Cases in United Kingdom')

#plt.plot(spain_agg.index,spain_agg['Confirmed'],label='Confirmed Cases in Spain')

plt.plot(france_agg.index,france_agg['Confirmed'],label='Confirmed Cases in France')



plt.title("Tracking Confirmed Cases in United States, Italy, France, Spain and United Kingdom")

plt.ylabel('Confirmed Cases')

plt.xlabel('Observation Date')

ax.legend()

plt.xticks(rotation=90)

# United States

us_agg['Death Rate in United States'] = (us_agg['Deaths']/us_agg['Confirmed'])*100

us_agg['Recovery Rate in United States'] = (us_agg['Recovered']/us_agg['Confirmed'])*100

# Italy

italy_agg['Death Rate in Italy'] = (italy_agg['Deaths']/italy_agg['Confirmed'])*100

italy_agg['Recovery Rate in Italy'] = (italy_agg['Recovered']/italy_agg['Confirmed'])*100

# Spain

spain_agg['Death Rate in Spain'] = (spain_agg['Deaths']/spain_agg['Confirmed'])*100

spain_agg['Recovery Rate in Spain'] = (spain_agg['Recovered']/spain_agg['Confirmed'])*100



us_agg.tail()


#us_agg[['ObservationDate','Death Rate in United States']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus in United States")

fig, ax = plt.subplots(figsize=(15,7))

plt.plot(us_agg.index,us_agg['Death Rate in United States'],label='Death Rate in United States')

plt.plot(france_agg.index,france_agg['Death Rate in France'],label='Death Rate in France')

plt.plot(uk_agg.index,uk_agg['Death Rate in United Kingdom'],label='Death Rate in United Kingdom')

plt.plot(italy_agg.index,italy_agg['Death Rate in Italy'],label='Death Rate in Italy')

plt.plot(spain_agg.index,spain_agg['Death Rate in Spain'],label='Death Rate in Spain')

#plt.legend('Death Rate in United States')

plt.title("Tracking Death Rates(%) due to Corona Virus in United States, Italy, France, Spain and United Kingdom")

plt.ylabel('Death Rate(%)')

plt.xlabel('Observation Date')

ax.legend()

plt.xticks(rotation=90)
fig, ax = plt.subplots(figsize=(15,7))

plt.plot(us_agg.index,us_agg['Recovery Rate in United States'],label='Recovery Rate in United States')

plt.plot(france_agg.index,france_agg['Recovery Rate in France'],label='Recovery Rate in France')

plt.plot(uk_agg.index,uk_agg['Recovery Rate in United Kingdom'],label='Recovery Rate in United Kingdom')

plt.plot(spain_agg.index,spain_agg['Recovery Rate in Spain'],label='Recovery Rate in Spain')

plt.plot(italy_agg.index,italy_agg['Recovery Rate in Italy'],label='Recovery Rate in Italy')

#plt.legend('Death Rate in United States')

plt.title("Recovery Rates(%) for United States, France, Spain, Italy and United Kingdom")

plt.ylabel('Recovery Rate(%)')

plt.xlabel('Observation Date')

ax.legend()

plt.xticks(rotation=90)
pak = pd.DataFrame(full_table[full_table['Country/Region']=='Pakistan'])

bang = pd.DataFrame(full_table[full_table['Country/Region']=='Bangladesh'])

india = pd.DataFrame(full_table[full_table['Country/Region']=='India'])
pak['Death Rate in Pakistan'] = (pak['Deaths']/pak['Confirmed'])*100

bang['Death Rate in Bangladesh'] = (bang['Deaths']/bang['Confirmed'])*100

india['Death Rate in India'] = (india['Deaths']/india['Confirmed'])*100

#Recovery

pak['Recovery Rate in Pakistan'] = (pak['Recovered']/pak['Confirmed'])*100

bang['Recovery Rate in Bangladesh'] = (bang['Recovered']/bang['Confirmed'])*100

india['Recovery Rate in India'] = (india['Recovered']/india['Confirmed'])*100
fig, ax = plt.subplots(figsize=(15,7))



ax.set_xlabel("Observation Date")

ax.set_ylabel("Death Rate(%)")



pak[['ObservationDate','Death Rate in Pakistan']].plot(x='ObservationDate',kind='line',ax=ax)



#fig, ax = plt.subplots(figsize=(15,7))

#ax.set_xlabel("Observation Date")

#ax.set_ylabel("Death Rate(%)")



bang[['ObservationDate','Death Rate in Bangladesh']].plot(x='ObservationDate',kind='line',ax=ax)



#fig, ax = plt.subplots(figsize=(15,7))

#ax.set_xlabel("Observation Date")

#ax.set_ylabel("Death Rate(%)")



india[['ObservationDate','Death Rate in India']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus related Deaths in India, Pakistan and Bangladesh")



fig, ax = plt.subplots(figsize=(15,7))



ax.set_xlabel("Observation Date")

ax.set_ylabel("Recovery Rate(%)")



pak[['ObservationDate','Recovery Rate in Pakistan']].plot(x='ObservationDate',kind='line',ax=ax)





bang[['ObservationDate','Recovery Rate in Bangladesh']].plot(x='ObservationDate',kind='line',ax=ax)





india[['ObservationDate','Recovery Rate in India']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus related Recoveries in India, Pakistan and Bangladesh")

asia = full_table[full_table['Country/Region'].isin(['India','Pakistan','Bangladesh'])]

asia['Death Rate'] = (asia['Deaths']/asia['Confirmed'])*100

asia


fig, ax = plt.subplots(figsize=(15,7))

ax.set_xlabel("Observation Date")

ax.set_ylabel("Count of Confirmed Positive Cases, Deaths, and Recoveries")

pak[['ObservationDate','Confirmed','Deaths','Recovered']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus in Pakistan")



fig, ax = plt.subplots(figsize=(15,7))

ax.set_xlabel("Observation Date")

ax.set_ylabel("Count of Confirmed Positive Cases, Deaths, and Recoveries")



bang[['ObservationDate','Confirmed','Deaths','Recovered']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus in Bangladesh")

fig, ax = plt.subplots(figsize=(15,7))

ax.set_xlabel("Observation Date")

ax.set_ylabel("Count of Confirmed Positive Cases, Deaths, and Recoveries")



india[['ObservationDate','Confirmed','Deaths','Recovered']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus in India")
iran = pd.DataFrame(full_table[full_table['Country/Region']=='Iran'])

iraq = pd.DataFrame(full_table[full_table['Country/Region']=='Iraq'])

turkey = pd.DataFrame(full_table[full_table['Country/Region']=='Turkey'])



fig, ax = plt.subplots(figsize=(15,7))



ax.set_xlabel("Observation Date")

ax.set_ylabel("Count of Confirmed Positive Cases, Deaths, and Recoveries")





iran[['ObservationDate','Confirmed','Deaths','Recovered']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus in Iran")

## Iraq

fig, ax = plt.subplots(figsize=(15,7))

ax.set_xlabel("Observation Date")

ax.set_ylabel("Count of Confirmed Positive Cases, Deaths, and Recoveries")





iraq[['ObservationDate','Confirmed','Deaths','Recovered']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus in Iraq")



#Turkey



fig, ax = plt.subplots(figsize=(15,7))

ax.set_xlabel("Observation Date")

ax.set_ylabel("Count of Confirmed Positive Cases, Deaths, and Recoveries")





turkey[['ObservationDate','Confirmed','Deaths','Recovered']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus in Turkey")

iran['Death Rate in Iran'] = pd.DataFrame((iran['Deaths']/iran['Confirmed'])*100)

iraq['Death Rate in Iraq'] = pd.DataFrame((iraq['Deaths']/iraq['Confirmed'])*100)

turkey['Death Rate in Turkey'] = pd.DataFrame((turkey['Deaths']/turkey['Confirmed'])*100)
#Iran

fig, ax = plt.subplots(figsize=(15,7))

ax.set_xlabel("Observation Date")

ax.set_ylabel("Death Rate(%)")





iran[['ObservationDate','Death Rate in Iran']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus Related Deaths in Iran, Iraq and Turkey")



#Iraq

#fig, ax = plt.subplots(figsize=(15,7))

ax.set_xlabel("Observation Date")

ax.set_ylabel("Death Rate(%)")



iraq[['ObservationDate','Death Rate in Iraq']].plot(x='ObservationDate',kind='line',ax=ax)



# Turkey

#fig, ax = plt.subplots(figsize=(15,7))

ax.set_xlabel("Observation Date")

ax.set_ylabel("Death Rate(%)")



turkey[['ObservationDate','Death Rate in Turkey']].plot(x='ObservationDate',kind='line',ax=ax)

malaysia = full_table[full_table['Country/Region']=='Malaysia']

indonesia = full_table[full_table['Country/Region']=='Indonesia']

#china = full_table[full_table['Country/Region']=='China']
fig, ax = plt.subplots(figsize=(15,7))

ax.set_xlabel("Observation Date")

ax.set_ylabel("Count of Confirmed Positive Cases, Deaths, and Recoveries")





malaysia[['ObservationDate','Confirmed','Deaths','Recovered']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus in Malaysia")



fig, ax = plt.subplots(figsize=(15,7))

ax.set_xlabel("Observation Date")

ax.set_ylabel("Count of Confirmed Positive Cases, Deaths, and Recoveries")



indonesia[['ObservationDate','Confirmed','Deaths','Recovered']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus in Indonesia")



#fig, ax = plt.subplots(figsize=(15,7))

#ax.set_xlabel("Observation Date")

#ax.set_ylabel("Count of Confirmed Positive Cases, Deaths, and Recoveries")



#china[['ObservationDate','Confirmed','Deaths','Recovered']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus in China")

fig, ax = plt.subplots(figsize=(15,7))

ax.set_xlabel("Observation Date")

ax.set_ylabel("Count of Confirmed Positive Cases, Deaths, and Recoveries")



malaysia[['ObservationDate','Confirmed','Deaths']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus cases, deaths in Malaysia, Indonesia, India and Pakistan")

indonesia[['ObservationDate','Confirmed','Deaths']].plot(x='ObservationDate',kind='line',ax=ax)

pak[['ObservationDate','Confirmed','Deaths']].plot(x='ObservationDate',kind='line',ax=ax)

india[['ObservationDate','Confirmed','Deaths']].plot(x='ObservationDate',kind='line',ax=ax)

ax.legend(['Confirmed Cases in Malaysia','Confirmed Deaths in Malaysia',

           'Confirmed Cases in Indonesia','Confirmed Deaths in Indonesia',

          'Confirmed Cases in Pakistan','Confirmed Deaths in Pakistan',

          'Confirmed Cases in India','Confirmed Deaths in India'])

## As Per Media Reports quoting Zafar Mirza



import matplotlib.pyplot as plt

classes = ['Misc - Mostly Secondary Cases', 'Iranian Origin(Zaireen) Positive Cases','Positive Cases from Other Countries']

pop = [420,857,191]



plt.pie(pop,labels=classes,autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('Breakdown of Pakistans Corona Virus Cases')



#Show

plt.show()
china = full_table[full_table['Country/Region']=='Mainland China']

china.tail()
china_agg=pd.pivot_table(china, index=['ObservationDate'],values=['Confirmed','Deaths','Recovered'],aggfunc=np.sum)

fig, ax = plt.subplots(figsize=(15,7))

plt.plot(china_agg.index,china_agg.values)

plt.legend(['Confirmed','Deaths','Recovered'])

plt.title("Tracking Corona Virus in China")

plt.ylabel('Number of Positive Cases, Deaths and Recoveries')

plt.xticks(rotation=90)
china_agg['Death Rate in China'] = (china_agg['Deaths']/china_agg['Confirmed'])*100

china_agg['Recovery Rate in China'] = (china_agg['Recovered']/china_agg['Confirmed'])*100

china_agg.tail()
china_agg['Death Rate in China'] = (china_agg['Deaths']/china_agg['Confirmed'])*100

#us_agg[['ObservationDate','Death Rate in United States']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus in United States")

fig, ax = plt.subplots(figsize=(15,7))

plt.plot(china_agg.index,china_agg[['Death Rate in China','Recovery Rate in China']])

#plt.legend('Death Rate in United St')

plt.title("Tracking Death/Recovery Rates(%) Corona Virus in China")

plt.ylabel('Death/Recovery Rate(%)')

plt.xlabel('Observation Date')

ax.legend(['Death Rate in China','Recovery Rate in China'])

plt.xticks(rotation=90)
fig, ax = plt.subplots(figsize=(15,7))

plt.plot(china_agg.index,china_agg['Death Rate in China'],label='Death Rate in China')

#plt.plot(us_agg.index,us_agg['Death Rate in USA'])

plt.plot(us_agg.index,us_agg['Death Rate in United States'],label='Death Rate in USA')



#plt.legend('Death Rate in United St')

plt.title("Tracking Death Rates(%) Corona Virus in China and United States")

plt.ylabel('Death Rate(%)')

plt.xlabel('Observation Date')

ax.legend()

plt.xticks(rotation=90)
brazil = full_table[full_table['Country/Region']=='Brazil']

brazil.tail()
fig, ax = plt.subplots(figsize=(15,7))

ax.set_xlabel("Observation Date")

ax.set_ylabel("Count of Confirmed Positive Cases, Deaths, and Recoveries")



brazil[['ObservationDate','Confirmed','Recovered','Deaths']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus cases, deaths in Brazil")

plt.xticks(rotation=90)
mexico = full_table[full_table['Country/Region']=='Mexico']

mexico.tail()
fig, ax = plt.subplots(figsize=(15,7))

ax.set_xlabel("Observation Date")

ax.set_ylabel("Count of Confirmed Positive Cases, Deaths, and Recoveries")



mexico[['ObservationDate','Confirmed','Recovered','Deaths']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus cases, deaths in Mexico")

plt.xticks(rotation=90)
canada = full_table[full_table['Country/Region']=='Canada']

#us.sample(10)

canada_agg=pd.pivot_table(canada, index=['ObservationDate'],values=['Confirmed','Deaths','Recovered'],aggfunc=np.sum)

fig, ax = plt.subplots(figsize=(15,7))

plt.plot(canada_agg.index,canada_agg.values)

plt.legend(['Confirmed','Deaths','Recovered'])

plt.title("Tracking Corona Virus in Canada")

plt.ylabel('Count')

plt.xticks(rotation=90)
canada_agg['Death Rate in Canada'] = (canada_agg['Deaths']/canada_agg['Confirmed'])*100

canada_agg['Recovery Rate in Canada'] = (canada_agg['Recovered']/canada_agg['Confirmed'])*100

fig, ax = plt.subplots(figsize=(15,7))

plt.plot(canada_agg.index,canada_agg['Death Rate in Canada'],label='Death Rate in Canada')

#plt.plot(us_agg.index,us_agg['Death Rate in USA'])

plt.plot(canada_agg.index,canada_agg['Recovery Rate in Canada'],label='Recovery Rate in Canada')



#plt.legend('Death Rate in United St')

plt.title("Tracking Death Rates(%) Corona Virus in Canada")

plt.ylabel('Death/Recovery Rate(%)')

plt.xlabel('Observation Date')

ax.legend()

plt.xticks(rotation=90)
aus = full_table[full_table['Country/Region']=='Australia']

aus.tail()
aus_agg=pd.pivot_table(aus, index=['ObservationDate'],values=['Confirmed','Deaths','Recovered'],aggfunc=np.sum)

fig, ax = plt.subplots(figsize=(15,7))

plt.plot(aus_agg.index,aus_agg.values)

plt.legend(['Confirmed','Deaths','Recovered'])

plt.title("Tracking Corona Virus in Australia")

plt.ylabel('Number of Positive Cases, Deaths and Recoveries')

plt.xticks(rotation=90)
aus_agg['Death Rate in Australia'] = (aus_agg['Deaths']/aus_agg['Confirmed'])*100

aus_agg['Recovery Rate in Australia'] = (aus_agg['Recovered']/aus_agg['Confirmed'])*100

#us_agg[['ObservationDate','Death Rate in United States']].plot(x='ObservationDate',kind='line',ax=ax, title="Tracking Corona Virus in United States")

fig, ax = plt.subplots(figsize=(15,7))

plt.plot(aus_agg.index,aus_agg[['Death Rate in Australia','Recovery Rate in Australia']])

#plt.legend('Death Rate in United St')

plt.title("Tracking Death Rates(%) Corona Virus in Australia")

plt.ylabel('Death Rate(%)')

plt.xlabel('Date')

ax.legend(['Death Rate in Australia','Recovery Rate in Australia'])

plt.xticks(rotation=90)