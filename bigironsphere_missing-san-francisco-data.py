import numpy as np 
import pandas as pd 

df = pd.read_csv('../input/ca.csv') #place California data in our DataFrame, df
print(df.columns)
df.head()
cities = df.CITY.value_counts()
cities[:15] #top city regions by number of addresses
print('REGION value counts: \n', df.REGION.value_counts(), '\n')
print('DISTRICT value counts: \n', df.DISTRICT.value_counts())
test = df[df.CITY == 'UNASSIGNED']
long = test.LON.mean()
lat = test.LAT.mean()

print('Average UNASSIGNED location is at Longitude: %f Latitude: %f' % (long, lat))
SF_ZIP = ['94102', '94104', '94103', '94105', '94108', '94107', '94110', '94109', '94112', '94111', '94115', '94114', '94117', '94116', 
          '94118', '94121', '94123', '94122', '94124', '94127', '94126', '94129', '94131', '94133', '94132', '94134', '94139', '94143',
          '94146', '94151', '94159', '94158', '94188', '94177']

SF_addresses = df[df['POSTCODE'].isin(SF_ZIP)]
print(SF_addresses)