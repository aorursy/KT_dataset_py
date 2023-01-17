# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_dir = '../input/'
cities=pd.read_csv(data_dir+'WCities.csv')

print(cities.shape)
print(cities.columns)
cities.head()
gcity=pd.read_csv(data_dir+'WGameCities.csv')

print(gcity.shape)
print(gcity.columns)
gcity.head()
city=pd.merge(left=gcity,right=cities,how='inner',on=['CityID'])
city.head()
tcompactresult=pd.read_csv(data_dir+'WNCAATourneyCompactResults.csv')

print(tcompactresult.shape)
print(tcompactresult.columns)
tcompactresult.head()
tcompactresult.WLoc.value_counts()
tseeds=pd.read_csv(data_dir+'WNCAATourneySeeds.csv')

print(tseeds.shape)
print(tseeds.columns)
tseeds.head()
tslots=pd.read_csv(data_dir+'WNCAATourneySlots.csv')

print(tslots.shape)
print(tslots.columns)
tslots.head()
rseasoncr=pd.read_csv(data_dir+'WRegularSeasonCompactResults.csv')

print(rseasoncr.shape)
print(rseasoncr.columns)
rseasoncr.head()
season=pd.read_csv(data_dir+'WSeasons.csv')

print(season.shape)
print(season.columns)
season
subb=pd.read_csv(data_dir+'WSampleSubmissionStage1.csv')

print(subb.shape)
print(subb.columns)
subb.head()
subb.tail()