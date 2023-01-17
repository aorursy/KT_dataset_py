# Import Libraries #
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # matplotlib for plotting
import seaborn as sns  # seaborn to help with visualizations
from subprocess import check_output  # check files in directory
print(check_output(["ls", "../input"]).decode("utf8"))
# Data Upload #
ufo_db = '../input/scrubbed.csv'  # Dataset file path
ufo_data = pd.read_csv(ufo_db, low_memory=False)  # DATA LOADED INTO ufo_data
ufo_data.columns
ufo_data.shape
ufo_data.head(5)
ufo_data.tail(5)
nulvals = ufo_data.isnull().sum()
nulpct = (nulvals / len(ufo_data))*100
print(' Null Values (% of entries):')
print(round(nulpct.sort_values(ascending=False),2))
# CLEAN Null values from dataset
ufo_data = pd.read_csv(ufo_db, 
                        low_memory = False, 
                        na_values = ['UNKNOWN','UNK'], 
                        na_filter = True, 
                        skip_blank_lines = True)  # load dataset without NA values

# Choose columns to work with based on our initial questions
ufo_subcols = ['datetime', 'city', 'state', 'country', 'shape', 'duration (seconds)',
        'comments', 'date posted', 'latitude',
       'longitude ']

# After tidying data and choosing what to work with, create dataframe to work with
ufo_data = pd.DataFrame(data=ufo_data, columns=ufo_subcols)

# ...drop null values
ufo_data = ufo_data.dropna(thresh=8)

#...reset the index
ufo_data = ufo_data.reset_index(drop=True)

# EXTRACT LATITUDES #
ufo_data['latitude'] = pd.to_numeric(ufo_data['latitude'],errors = 'coerce')  # latitudes as numerics
ufo_data['longitude '] = pd.to_numeric(ufo_data['longitude '], errors='coerce')

# CHANGE VARIABLES UFO_DATE IN SCRIPT 1-1-18
ufo_date = ufo_data.datetime.str.replace('24:00', '00:00')  # clean illegal values
ufo_date = pd.to_datetime(ufo_date, format='%m/%d/%Y %H:%M')  # now in datetime

ufo_data['datetime'] = ufo_data.datetime.str.replace('24:00', '00:00')
ufo_data['datetime'] = pd.to_datetime(ufo_data['datetime'], format='%m/%d/%Y %H:%M')
ufo_data.shape

ufo_yr = ufo_date.dt.year  # series with the year exclusively

## Set axes ##
years_data = ufo_yr.value_counts()
years_index = years_data.index  # x ticks
years_values = years_data.get_values()

## Create Bar Plot ##
plt.figure(figsize=(15,8))
plt.xticks(rotation = 60)
plt.title('UFO Sightings by Year')

years_plot = sns.barplot(x=years_index[:60],y=years_values[:60], palette = "GnBu")
country_sightings = ufo_data.country.value_counts()  # num ufo sightings per country 

explode = (0, 0, 0, 0., 0.05)
colors = ['lightblue','gold','yellowgreen','lightcoral','orange']
country_sightings.plot(kind = 'pie', fontsize = 0, title='UFO Sightings by Country', colors=colors,
                       autopct='%1.1f%%',shadow=True, explode=explode,figsize=(8,8))
plt.legend(labels=['United States','Canada','United Kingdom','Australia','Germany'], loc="best")
plt.tight_layout()
ufo_data['country'].value_counts().head(10)
# Filter US Values to analyze US state sightings #
usa_filter = ufo_data['country']=='us'  # filter non-usa country
us_data = ufo_data[usa_filter]  # DF ufo_data of only US sightings - includes puerto rico and dc

# Get x and y axes for states bar viz #
states_sights = us_data.state.value_counts()  # State Data
state_names = states_sights.index  # x axis ticks
state_freq = states_sights.get_values()  # y axis values

# States Frequency Pareto Chart #
plt.figure(figsize=(15,8))
plt.xticks(rotation = 60)
plt.title('Total UFO Sightings by State')
states_plot = sns.barplot(x=state_names,y=state_freq, palette="GnBu_r")
plt.show()
print('Top 10 States for Total UFO Sightings:')
print(states_sights[:10].sort_values(ascending=False))
statespop = {'al':4872725.,'ak':746079.,'az':7044577.,'ar':2998643.,'ca':39506094.,
            'co':5632271.,'ct':3568174.,'de':960054.,'dc':691963.,'fl':20979964.,
            'ga':10421344.,'hi':1431957.,'id':1713452.,'il':12764031.,'in':6653338.,
            'ia':3147389.,'ks':2907857.,'ky':4449337.,'la':4694372.,'me':1333505.,
            'md':6037911.,'ma':6839318.,'mi':9938885.,'mn':5557469.,'ms':2988062.,
            'mo':6109796.,'mt':1052967.,'ne':1920467.,'nv':2996358,'nh':1339479.,
            'nj':8953517.,'nm':2081702.,'ny':19743395.,'nc':10258390.,'nd':759069.,
            'oh':11623656.,'ok':3939708.,'or':4162296.,'pa':12776550.,'pr':3661538.,
            'ri':1057245.,'sc':5027404.,'sd':872989.,'tn':6707332.,'tx':28295553.,
            'ut':3111802.,'vt':623100.,'va':8456029.,'wa':7415710.,'wv':1821151.,
            'wi':5789525.,'wy':584447.} 
states_pop = pd.Series(statespop)  # turn dict into series type


state_propsight = (states_sights / states_pop)*100 # prop data series for viz, scaled for style
state_propsight = state_propsight.sort_values(ascending=False) 

# Visualize it
us_namesp = state_propsight.index  # x ticks
us_sightsp = state_propsight.get_values()  # y values

plt.figure(figsize=(15,8))
plt.xticks(rotation=60)
plt.title('State UFO Sightings Relative to Population')
sns.barplot(x = us_namesp[:50], y = us_sightsp[:50], palette="GnBu_r")
plt.show()
print('States with Highest Proportion of UFO Sightings:')
print(round(state_propsight[:10],2))
m_cts = (ufo_data['datetime'].dt.month.value_counts()).sort_index()
m_ctsx = m_cts.index
m_ctsy = m_cts.get_values()
f, ax = plt.subplots(figsize=(15,8))

sns.barplot(x=m_ctsx, y=m_ctsy, palette="YlGnBu")
ax.set_title('Global UFO Sightings by Month')
ax.set_xlabel('Month')
ax.set_ylabel('# Sightings')
plt.xticks(rotation=45)
plt.show()
# Add Season Column to ufo_date #
# Given a datetime, return the season that it's in #
ufo_datem = ufo_date.dt.month
spring = range(5,7)
summer = range(7,10)
fall = range(10,12)
seasons = []

for st_date in ufo_datem:
    # Conversion Process # 
    if st_date in spring:
        seasons.append('Spring')
    elif st_date in summer:
        seasons.append('Summer')
    elif st_date in fall:
        seasons.append('Fall')
    else:
        seasons.append('Winter')
ufo_data['season'] = pd.Series(seasons, index=ufo_data.index)
# Add Hemisphere Column to ufo_date #
hemis = []
for st_loc in ufo_data['latitude']:
    if st_loc >= 0 :
        hemis.append('Northern Hemisphere')
    else:
        hemis.append('Southern Hemisphere')
ufo_data['hemisphere'] = (pd.Series(hemis, index=ufo_data.index)).astype('category')
ufo_data['season'].value_counts()
plt.figure(figsize=(15,8))
sns.countplot(x='season', hue='hemisphere', data=ufo_data)
print(max(ufo_data['latitude']))
print(min(ufo_data['latitude']))
resp_n = ufo_data[ufo_data['hemisphere'] == 'Northern Hemisphere']
resp_s = ufo_data[ufo_data['hemisphere'] == 'Southern Hemisphere']
nsperc = resp_n['season'].value_counts() / len(resp_n) * 100
ssperc = resp_s['season'].value_counts() / len(resp_s) * 100

pos = list(range(len(nsperc)))
width = 0.25
fig, ax = plt.subplots(figsize = (15,8))

plt.bar(pos, nsperc, width, alpha = .7, color='#0064A9')
plt.bar([p + width*1.05 for p in pos], ssperc, width, alpha = .65, color='#E1E066')

ax.set_title('UFO Sightings by Season')
ax.set_xlabel('Season')
ax.set_ylabel('% of UFO Sightings')
ax.set_xticks([p + .5 * width for p in pos])
ax.set_xticklabels(nsperc.index)
plt.xticks(rotation=45)

plt.ylim([0, 60])
plt.legend(['Northern Hemisphere','Southern Hemisphere'], loc='upper left')
plt.show()
print('Northern Hemisphere:\n', nsperc)
print('Southern Hemisphere:\n',ssperc)
n_mon = resp_n['datetime'].dt.month
s_mon = resp_s['datetime'].dt.month

# N.MONTH #
monn_cts = n_mon.value_counts().sort_index()
monn_in = monn_cts.index 
monn_val = monn_cts.get_values()

# S.MONTH #
mons_cts = s_mon.value_counts().sort_index()
mons_in = mons_cts.index
mons_val = mons_cts.get_values()

plt.figure(figsize=(15,7))
plt.xticks(rotation = 60)
plt.title('UFO Sightings by Month - Northern Hemisphere')
sns.barplot(x=monn_in,y=monn_val, palette="GnBu_d")
plt.show()

plt.figure(figsize=(15,7))
plt.xticks(rotation = 60)
plt.title('UFO Sighting by Month - Southern Hemisphere')
sns.barplot(x=mons_in, y=mons_val, palette="OrRd_d")
plt.show()

print('Top Months for UFO Sightings in N. Hemishpere:')
print(n_mon.value_counts()[:6])
print('Top Months for UFO Sightings in S. Hemisphere:')
print(s_mon.value_counts()[:6])