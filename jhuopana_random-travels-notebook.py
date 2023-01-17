# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import urllib, json

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
# Any results you write to the current directory are saved as output.
borders1 = np.array([	[48.3	,-124.7],
						[48.17	,-123.13],
						[48.98	,-123.09],
						[49.01	,-95.18],
						[47.9	,-89.64],
						[45.06	,-82.48],
						[41.64	,-82.78],
						[47.34	,-68.68],
						[44.46	,-66.93],
						[42.55	,-71.05],
						[41.46	,-70.36],
						[37.32	,-75.94],
						[35.27	,-75.55],
						[30.74	,-81.57],
						[26.85	,-79.9],
						[25.04	,-80.77],
						[26.81	,-82.44],
						[29.64	,-83.72],
						[29.52	,-94.09],
						[27.71	,-97.46],
						[25.89	,-97.11],
						[26.46	,-99.24],
						[29.71	,-101.39],
						[29.26	,-103.89],
						[31.72	,-106.4],
						[31.4	,-108.27],
						[31.27	,-110.92],
						[32.58	,-115.49],
						[32.43	,-117.08],
						[34.48	,-120.54],
						[40.43	,-124.37],
						[45.9	,-123.95],
						[48.46	,-124.83]])
#Lat to lon 
borders1[:,[0, 1]] = borders1[:,[1, 0]]
#Let's see if we done this already
if os.path.exists('../input/cities-with-location/WCities_mod.csv'):
	city_data = pd.read_csv('../input/cities-with-location/WCities_mod.csv')
else:
	#Read the city name file
	city_data = pd.read_csv('../input/womens-machine-learning-competition-2018/WCities.csv')

	#Use Google API to get lat and lon from Google Geocode
	url_0 = "https://maps.googleapis.com/maps/api/geocode/json?address="
	#You need to get an api key from https://developers.google.com/
	url_apikey = "YOUR API KEY"

	#Creating some empty columns
	city_data['Lat'] = 0.0
	city_data['Lng'] = 0.0
	
	#Reading JSON input from Google Geocode and 
	for  index, row in city_data.iterrows():
		url = url_0 + str(row['City'])+','+str(row['State']) + url_apikey 
		response = urllib.request.urlopen(url)
		data = json.loads(response.read())
		lat = float(data['results'][0]['geometry']['location']['lat'])
		lng = float(data['results'][0]['geometry']['location']['lng'])
		city_data.iloc[[index],[city_data.columns.get_loc("Lat")]] = lat
		city_data.iloc[[index],[city_data.columns.get_loc("Lng")]] = lng
		print(str(row['City']),'Lat ',lat,' ','Lon ',lng)
	#Write the modified file
	city_data.to_csv('../input/cities-with-location/WCities_mod.csv', sep=',', encoding='utf-8', index=False)

print(city_data.head())
#Read the game data
game_data = pd.read_csv('../input/womens-machine-learning-competition-2018/WGameCities.csv')
name_data = pd.read_csv('../input/womens-machine-learning-competition-2018/WTeams.csv')

teams = []

#Pick a random season
season = np.random.choice(np.unique(game_data['Season'].values))
#Creating a smaller dataset
game_data_season = game_data.loc[game_data['Season'] == season]
#Pick 30 random teams
teams_playing = np.random.choice(np.unique(game_data_season['WTeamID'].values), 30)
fig, ax = plt.subplots()
#Plot the map
patches = []
polygon = Polygon(borders1, True)
patches.append(polygon)
p = PatchCollection(patches, alpha=0.2)
ax.add_collection(p)

for team in teams_playing:
	#Pick a team
	temp_name = name_data[name_data['TeamID']==team]['TeamName'].values
	teams.append(temp_name[0])
	
	#Data for selected season and team
	game_data_season_team = game_data_season.loc[(game_data_season['WTeamID'] == team) | (game_data_season['LTeamID'] == team)]

	travel = np.zeros((game_data_season_team.shape[0],2))
	i=0
	for index, row in game_data_season_team.iterrows():
		travel[i][0] = city_data[city_data['CityID'] == row['CityID']]['Lat'].values
		travel[i][1] = city_data[city_data['CityID'] == row['CityID']]['Lng'].values
		i += 1
	ax.plot(travel[:,1],travel[:,0], alpha=0.3)

ax.axis([-160, -62, 15, 63])
lgd = ax.legend(teams, ncol=2, loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title('Team travels during season '+str(season))
ax.grid()
fig.set_size_inches(7,4,forward=True)
fig.savefig('Travel_figure.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
