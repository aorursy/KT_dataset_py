import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)
df = pd.read_csv('../input/atpdata/ATP.csv')
df.head()
df.info()
winners = df[['winner_name', 'winner_hand', 'winner_ioc','winner_rank']]
winners.columns = ['name', 'hand', 'country', 'rank']
losers = df[['loser_name', 'loser_hand', 'loser_ioc', 'loser_rank']]
losers.columns = ['name', 'hand', 'country', 'rank']
frames = [winners, losers]
plays = pd.concat(frames)
#get the best ranking of each player
plays = plays.groupby(['name', 'hand', 'country']).agg('min').reset_index().sort_values(['rank','name'])
#create flag for the players who got into top 10
plays['top10'] = plays['rank'] <= 10
plays
inf = pd.read_csv('../input/atp-players-overviews/player_overviews_UNINDEXED.csv', 
                  names=['player_id','player_slug','first_name','last_name','player_url','flag_code','residence', 'birthplace','birthdate','birth_year','birth_month',
                         'birth_day','turned_pro','weight_lbs', 'weight_kg','height_ft','height_inches','height_cm','hand','backhand'])
inf.head()
#create full name so we can join with plays dataset later
inf['name'] = inf['first_name'] + ' ' + inf['last_name']
#replace zero values with nan
inf['turned_pro'] = inf['turned_pro'].replace(0, np.nan)
inf['height_cm'] = inf['height_cm'].replace(0, np.nan)
inf['weight_kg'] = inf['weight_kg'].replace(0, np.nan)
#calculate age of the player he turned pro
inf['turned_pro_age'] = inf['turned_pro'] - inf['birth_year']
#change birthdate data type to date
inf['birthdate'] = pd.to_datetime(inf['birthdate'], format='%Y.%m.%d')
#select only relevant columns
inf = inf[['name', 'residence', 'birthplace', 'birthdate', 'birth_year','birth_month', 'birth_day', 'turned_pro_age', 'weight_kg', 'height_cm', 'hand', 
           'backhand']]
inf.info()
inf.head()
#create dataset with zodiacs and its dates
data = {'d_start':  ['19', '21', '21', '21', '23', '23', '23', '23', '22', '22', '20', '19'],
        'm_start':  ['3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '1', '2'],
        'd_end':  ['20', '20', '20', '22', '22', '22', '22', '21','21', '19', '18', '20'],        
        'm_end':  ['4', '5', '6', '7', '8', '9', '10', '11','12', '1', '2', '3'],
        'zodiac': ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
        }
zdt = pd.DataFrame(data, columns = ['d_start', 'm_start', 'd_end', 'm_end', 'zodiac'])
#ensure correct data types
zdt[['d_start', 'm_start', 'd_end', 'm_end']] = zdt[['d_start', 'm_start', 'd_end', 'm_end']].apply(pd.to_numeric)
zdt.head()
#generate years
yDF = pd.DataFrame({"Year": pd.Series(range(1900,2020))})
#cross join dataset with years only and with month, day and zodiac dataset (created in previous step)
yDF = pd.merge(yDF.assign(key=0), zdt.assign(key=0), on='key').drop('key', axis=1)
#create full start and end date for each zodiac
yDF['Date_start'] = pd.to_datetime(yDF.Year*10000+yDF.m_start*100+yDF.d_start,format='%Y%m%d')
yDF['Date_end'] = pd.to_datetime(yDF.Year*10000+yDF.m_end*100+yDF.d_end,format='%Y%m%d')
#select only relevant columns
zodiacs = yDF[['Date_start', 'Date_end', 'zodiac']]
zodiacs.head()
#create artifical key to cross join every player information with every zodiac
inf['key'] = 0
zodiacs['key'] = 0
infAll = inf.merge(zodiacs)
#filter each player with corresponding zodiac
infAll = infAll[(infAll.birthdate >= infAll.Date_start) & (infAll.birthdate <= infAll.Date_end)]
#drop useless columns
infAll.drop(['key', 'Date_start', 'Date_end'], axis=1, inplace=True)
infAll.info()
infAll.head()
#perform join with plays and player information
out = plays.merge(infAll, on='name')
out
import plotly.express as px
fig = px.histogram(out, x="top10")
fig.show()
print(out[out['top10']].sort_values('name').name.values)
fig = px.histogram(out, x="zodiac", color='zodiac').update_xaxes(categoryorder = 'total descending')
fig.show()
fig = px.histogram(out[out['top10'] == True], x="zodiac", color='zodiac').update_xaxes(categoryorder = 'total descending')
fig.show()
fig = px.histogram(out, x="hand_y", color='top10')
fig.show()
fig = px.histogram(out[out['top10'] == True], x="top10", color='hand_x')
fig.show()
fig = px.histogram(out, x="backhand", color='top10')
fig.show()
fig = px.histogram(out, x="country", color='top10').update_xaxes(categoryorder = 'total descending')
fig.show()
fig = px.histogram(out[out['top10'] == True], x="country").update_xaxes(categoryorder = 'total descending')
fig.show()
fig = px.histogram(out, x="height_cm", color='top10', nbins = 20)
fig.show()
fig = px.histogram(out, x="weight_kg", color='top10', nbins = 20)
fig.show()
fig = px.histogram(out, x="turned_pro_age", color='top10', nbins = 20)
fig.show()
!pip install googlemaps
import googlemaps
gmaps_key = googlemaps.Client(key="AIzaSyDREugpyCcRUjp_3KvPl6oOUzip6mm7NRY")
#create columns for latitude and longtitude
out['lon'] = None
out['lat'] = None
#get coordinates for birthplace using google maps api
for i in range(len(out)):
    geo_result = gmaps_key.geocode(out.loc[i, 'birthplace'])
    try:
        lon = geo_result[0]["geometry"]["location"]["lng"]
        lat = geo_result[0]["geometry"]["location"]["lat"]
        out.loc[i, 'lon'] = lon
        out.loc[i, 'lat'] = lat
    except:
        lat = None
        lon = None
fig = px.scatter_geo(out[out['top10'] == False], lon="lon", lat = 'lat', color_discrete_sequence=["black"],hover_name='birthplace', opacity= 0.2)
fig.show()
fig = px.scatter_geo(out[out['top10'] == True], lon="lon", lat = 'lat', color='top10', color_discrete_sequence=["green"],hover_name='birthplace', opacity=0.5)
fig.show()
#output the complete dataset
out.to_csv('ATP_players_info_full.csv', index = False)