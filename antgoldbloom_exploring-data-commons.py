#install the data commons package

!pip install -U git+https://github.com/google/datacommons.git@stable-1.x
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datacommons as dc
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

dcKey = user_secrets.get_secret("dcKey")

dc.set_api_key(dcKey)


city_dcids = dc.get_places_in(['country/USA'], 'City')['country/USA']

#city_dcids = city_dcids[0:400]
data = pd.DataFrame({'CityId': city_dcids})



data['PopId'] = dc.get_populations(data['CityId'],'Person')

data['population'] = dc.get_observations(data['PopId'], 'count','measuredValue','2017', measurement_method='CensusACS5yrSurvey')



#only look at cities with populations of 100K+

data = data[data['population'] > 100000]



data['population'] = data['population'].astype(int)



# Create the Pandas DataFrame 



data['median_age'] = dc.get_observations(data['PopId'], 'age','medianValue','2017', measurement_method='CensusACS5yrSurvey')

data['unemployment_rate'] = dc.get_observations(data['PopId'],'unemploymentRate','measuredValue','2017',observation_period='P1Y',measurement_method='BLSSeasonallyUnadjusted')



# Get the name of each city

data['City'] = dc.get_property_values(data['CityId'], 'name')

data = data.explode('City')



#Get the name of each state

data['State'] = dc.get_property_values(data['CityId'].str[:8], 'name')

data = data.explode('State')



data.index = data[['City','State']]



data['ObesityId'] = dc.get_populations(data['CityId'],'Person',constraining_properties={'age': 'Years18Onwards','healthBehavior': 'Obesity'})

data['obesity_rate'] = dc.get_observations(data['ObesityId'],'percent','measuredValue','2015','P1Y',measurement_method='CrudePrevalence')



data['ViolentCrimeId'] = dc.get_populations(data['CityId'],'CriminalActivities',constraining_properties={'crimeType': 'ViolentCrime'})

data['violent_crime_rate'] = round(dc.get_observations(data['ViolentCrimeId'],'count','measuredValue','2017','P1Y')/data['population']*100,2)

CitySample = ['New York','Los Angeles','Chicago','Houston','Philadelphia','Austin','Denver','Seattle','San Francisco', 'Washington DC','Boston','Miami','Charleston']

data[data['City'].isin(CitySample)].sort_values(by='population',ascending=False)[['population','unemployment_rate','obesity_rate','violent_crime_rate']][:11]
