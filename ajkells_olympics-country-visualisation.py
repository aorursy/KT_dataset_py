import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # nice package for statistical plots
import matplotlib.pyplot as plt # the standard library for plotting in python
athlete_data = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')
region_data = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv')
gdp_data = pd.read_csv('../input/gdp-data/UN_Gdp_data.csv')
athlete_data.head()
region_data.head()
gdp_data.head()

athlete_data = athlete_data.merge(region_data,left_on = 'NOC',right_on = 'NOC',how = 'left')
athlete_data.head()
countries_in_athelete_not_gdp = set(athlete_data['region'])-set(gdp_data['Country or Area'])
print(list(countries_in_athelete_not_gdp))
countries_in_gdp_not_athlete = set(gdp_data['Country or Area'])-set(athlete_data['region'])
print(list(countries_in_gdp_not_athlete))
gdp_data['Country or Area']=gdp_data['Country or Area'].str.replace( "China, People's Republic of","China")
gdp_data['Country or Area']=gdp_data['Country or Area'].str.replace( 'Iran, Islamic Republic of',"Iran")
gdp_data['Country or Area']=gdp_data['Country or Area'].str.replace( "Côte d'Ivoire","Ivory Coast")
gdp_data['Country or Area']=gdp_data['Country or Area'].str.replace( 'Russian Federation',"Russia")
gdp_data['Country or Area']=gdp_data['Country or Area'].str.replace( 'Republic of Korea',"South Korea")
gdp_data['Country or Area']=gdp_data['Country or Area'].str.replace('United Kingdom of Great Britain and Northern Ireland',"UK")
gdp_data['Country or Area']=gdp_data['Country or Area'].str.replace( 'United States',"US")
gdp_data['Country or Area']=gdp_data['Country or Area'].str.replace( 'Viet Nam',"Vietnam")
gdp_data['Country or Area']=gdp_data['Country or Area'].str.replace( 'Bolivia (Plurinational State of)',"Boliva")
athlete_data = athlete_data.merge(gdp_data,left_on = ['Year','region'],right_on = ['Year','Country or Area'],how = 'left')
athletes_with_gdp = athlete_data.dropna(subset=['Value'])
athletes_with_gdp.head()
athletes_with_gdp['Medal']=athletes_with_gdp['Medal'].fillna(0)
athletes_with_gdp['Medal_win']=athletes_with_gdp['Medal'].replace({'Gold':int(1), 'Silver':int(1), 'Bronze':int(1)})
athletes_with_gdp.head()
athletes_with_gdp['Value']=athletes_with_gdp['Value']/athletes_with_gdp.\
groupby('Year')['Value'].transform('max')
medal_tally_agnostic = athletes_with_gdp.\
groupby(['Year', 'Team'])[['Medal_win','Value']].\
agg({'Medal_win':'sum','Value':'mean'}).reset_index()
medal_tally_agnostic.plot.scatter('Value','Medal_win',logx='True')
clean_athlete_data = athlete_data.dropna(subset=['Country or Area'])
clean_athlete_data = clean_athlete_data.dropna(subset=['Age'])
clean_athlete_data = clean_athlete_data.dropna(subset=['Weight'])
clean_athlete_data = clean_athlete_data.dropna(subset=['Height'])
clean_athlete_data = clean_athlete_data.dropna(subset=['Value'])
clean_athlete_data = clean_athlete_data[clean_athlete_data['Season']=='Summer']


clean_athlete_data=clean_athlete_data.drop(['Name','Country or Area','City','Games','Item','notes'],axis=1)
non_winners_data=clean_athlete_data[clean_athlete_data['Medal'].isnull()]
winners_data = clean_athlete_data.dropna(subset=['Medal'])
clean_athlete_data.head()

fig=plt.figure(figsize=(18, 7), dpi= 80, facecolor='w', edgecolor='k')
plt.subplot(1, 2, 1)
plt.title('Distribution of ages among men')
# mens plot
sns.distplot(non_winners_data[non_winners_data['Sex']=='M']['Age'],bins=np.arange(1,100))
sns.distplot(winners_data[winners_data['Sex']=='M']['Age'],bins=np.arange(1,100))
plt.legend(['Non-winners','Winners'])


plt.subplot(1, 2, 2)
plt.title('Distribution of ages among women')
# womens plot
sns.distplot(non_winners_data[non_winners_data['Sex']=='F']['Age'],bins=np.arange(1,100))
sns.distplot(winners_data[winners_data['Sex']=='F']['Age'],bins=np.arange(1,100))
plt.legend(['Non-winners','Winners'])
fig, ax = plt.subplots(1, 1)
non_winners_data.plot.scatter(x='Height',y='Weight',c='DarkBlue',ax=ax)
winners_data.plot.scatter(x='Height',y='Weight',c='Red',ax=ax)
