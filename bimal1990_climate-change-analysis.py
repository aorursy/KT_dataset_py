import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output



#Load in files

tempbycity = pd.read_csv("../input/GlobalLandTemperaturesByCity.csv",index_col='dt',parse_dates=[0])

tempbycountry = pd.read_csv("../input/GlobalLandTemperaturesByCountry.csv",index_col='dt',parse_dates=[0])

tempbymajorcity = pd.read_csv("../input/GlobalLandTemperaturesByMajorCity.csv",index_col='dt',parse_dates=[0])

tempbystate = pd.read_csv("../input/GlobalLandTemperaturesByState.csv",index_col='dt',parse_dates=[0])

temp = pd.read_csv("../input/GlobalTemperatures.csv",index_col='dt',parse_dates=[0])
temps = pd.read_csv('../input/GlobalTemperatures.csv')



temps = temps[['dt', 'LandAverageTemperature']]



#Set the time to an easier format for graphing

temps['dt'] = pd.to_datetime(temps['dt'])

temps['year'] = temps['dt'].map(lambda x: x.year)

temps['month'] = temps['dt'].map(lambda x: x.month)



#split years into seasons

def findseason(month):

    if month >= 3 and month <= 5:

        return 'spring'

    elif month >= 6 and month <= 8:

        return 'summer'

    elif month >= 9 and month <= 11:

        return 'fall'

    else:

        return 'winter'



#min and max years

yearmin = temps['year'].min()

yearmax = temps['year'].max()

years = range(yearmin, yearmax +1)



#apply seasons attribute to global temp file

temps['season'] = temps['month'].apply(findseason)



springtemp = []

summertemp = []

falltemp = []

wintertemp = []



for year in years:

    currentyeardata = temps[temps['year'] == year]

    springtemp.append(currentyeardata[currentyeardata['season'] == 'spring']['LandAverageTemperature'].mean())

    summertemp.append(currentyeardata[currentyeardata['season'] == 'summer']['LandAverageTemperature'].mean())

    falltemp.append(currentyeardata[currentyeardata['season'] == 'fall']['LandAverageTemperature'].mean())

    wintertemp.append(currentyeardata[currentyeardata['season'] == 'winder']['LandAverageTemperature'].mean())

   
countries = tempbycountry['Country'].unique()

averagetemp = []



# calc mean temperature

for country in countries:

    averagetemp.append(tempbycountry[tempbycountry['Country'] == country]['AverageTemperature'].mean())

    

# nan cleaning

avgtempres = []

rescountries = []



for i in range(len(averagetemp)):

    if not np.isnan(averagetemp[i]):

        avgtempres.append(averagetemp[i])

        rescountries.append(countries[i])

        

# sorting

avgtempres, rescountries = (list(x) for x in zip(*sorted(zip(avgtempres, rescountries), key=lambda pair: pair[0])))
#Plot the average temperature year-round of every country in the list

f, ax = plt.subplots(figsize=(8, 54))

sns.barplot(x=avgtempres, y=rescountries, palette=sns.color_palette("RdBu_r", len(avgtempres)), ax=ax)



texts = ax.set(ylabel="", xlabel="Temperature (Celsius)", title="Average temperature in each country")
#A quick chart of the five hottest cities in 2010

tempbymajorcity[tempbymajorcity.index.year == 2010][['City', 'Country', 'AverageTemperature']].groupby(['City', 'Country']).mean().sort_values('AverageTemperature',ascending=False).head(n=10)
#Another quick chart of the hottest cities in 1980

tempbymajorcity[tempbymajorcity.index.year == 1980][['City', 'Country', 'AverageTemperature']].groupby(['City', 'Country']).mean().sort_values('AverageTemperature',ascending=False).head(n=10)