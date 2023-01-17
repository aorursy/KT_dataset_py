import pandas as pd
import requests
import json
import time
client_id=45454
client_secret='e8fe6327eb96f53484558706ca72c043488049e1'
## Get the tokens from file to connect to Strava
with open('/kaggle/input/strava-token/strava_tokens.json') as json_file:
    strava_tokens = json.load(json_file)
## If access_token has expired then use the refresh_token to get the new access_token

if strava_tokens['expires_at'] < time.time():
#Make Strava auth API call with current refresh token
    response = requests.post(
                        url = 'https://www.strava.com/oauth/token',
                        data = {
                                'client_id': client_id,
                                'client_secret': client_secret,
                                'grant_type': 'refresh_token',
                                'refresh_token': strava_tokens['refresh_token']
                                }
                    )
#Save response as json in new variable
    new_strava_tokens = response.json()
# Save new tokens to file
    with open('/kaggle/input/strava-token/strava_tokens.json', 'w') as outfile:
        json.dump(new_strava_tokens, outfile)
#Use new Strava tokens from now
    strava_tokens = new_strava_tokens
#Loop through all activities
page = 1
url = "https://www.strava.com/api/v3/activities"
access_token = strava_tokens['access_token']
## Create the dataframe ready for the API call to store your activity data

columns = [
            "name",
            "start_date_local",
            "type",
            "distance",
            "moving_time",
            "elapsed_time",
            "total_elevation_gain",
            "average_speed",
            "average_heartrate",
            "max_heartrate"
    ]

activities = pd.DataFrame(
    columns = columns
)
while True:
    
    # get page of activities from Strava
    r = requests.get(url + '?access_token=' + access_token + '&per_page=200' + '&page=' + str(page))
    r = r.json()
# if no results then exit loop
    if (not r):
        break
    
    # otherwise add new data to dataframe
    for x in range(len(r)):
        for col in columns:
            activities.loc[x + (page-1)*200,col] = r[x][col]
        activities.loc[x + (page-1)*200,'summary_polyline'] = r[x]['map']['summary_polyline']

# increment page
    page += 1
activities.to_csv('strava_activities.csv')
#Load activities
#activities=pd.read_csv('/kaggle/input/anton-sozykin-strava-activities/strava_activities.csv')
activities['start_date_local'] = pd.to_datetime(activities['start_date_local'].str[:11])
activities.sample(20)
import polyline
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

summary_polyline=activities.loc[1,'summary_polyline']


coordinates = polyline.decode(summary_polyline)

ride_longitudes = [coordinate[1] for coordinate in coordinates]
ride_latitudes = [coordinate[0] for coordinate in coordinates]


plt.figure(figsize=(16,9))

bonds=0.005

m = Basemap(
    llcrnrlon=min(ride_longitudes) - bonds,
    llcrnrlat=min(ride_latitudes) - bonds,
    urcrnrlon=max(ride_longitudes) + bonds, 
    urcrnrlat=max(ride_latitudes) + bonds,
    epsg=2205
)

m.arcgisimage(service='World_Imagery', verbose=True)
x, y = m(ride_longitudes, ride_latitudes)
m.plot(x, y, 'r-')

plt.show()
activities.drop(columns=['summary_polyline'],axis=1, inplace=True)
# Count types of training activities
print(activities['type'].value_counts())

activities.isnull().sum()
import datetime
runs=activities[activities['type']=='Run'].copy()
runs['average_pace_sec']=(runs['moving_time'])/(runs['distance']/1000)
runs['average_speed']=runs['average_speed']*3600/1000
runs['average_pace']=pd.to_timedelta(runs['average_pace_sec'], unit='s')
runs['distance']=runs['distance']/1000
runs.head(5)
import datetime

def SecToMin(sec):
    timeS=str(datetime.timedelta(seconds=sec))
    return timeS[-5:]

runs.plot(x='start_date_local' ,y=['distance','average_heartrate','max_heartrate','average_speed'],subplots=True,
                           sharex=False,
                           figsize=(16,16),
                           linestyle='none',
                           marker='o',
                           markersize=5,
                          )
y_range=[i for i in range (260,400,10)]
y_labels=[SecToMin(i) for i in y_range]

fig = plt.figure(figsize=(16, 8))
plt.plot(runs['start_date_local'], runs['average_pace_sec'],linestyle='none',marker='o')
plt.yticks(y_range,y_labels)
plt.title('Pace')
plt.show()

runs_time=runs.set_index('start_date_local').copy()
print(runs_time.info())
runs_time['distance']=runs_time['distance'].astype('float')
runs_time['average_heartrate']=runs_time['average_heartrate'].astype('int64')
runs_time['max_heartrate']=runs_time['max_heartrate'].astype('int64')
runs_time['average_speed']=runs_time['average_speed'].astype('int64')
#runs_time['average_pace_sec']=runs_time['average_pace_sec'].astype('int64')
print(runs_time.info())
print('How my average run looks')
display(runs_time.resample('M').mean())
print('How my average run looks:')
display(runs_time.resample('M').mean().mean())
print('How many trainings I had every month:')
display(runs_time.resample('M')['distance'].count())
print('How many trainings per month I had on average:')
display(runs_time.resample('M')['distance'].count().mean())
import numpy as np
from datetime import date, timedelta

todayM = date.today()+ timedelta(days=60)
Months_List=np.arange('2019-03', todayM, dtype='datetime64[M]')
Months_Labels=[pd.to_datetime(mm).strftime( "%B-%Y") for mm in list(Months_List)]
# Check target - 100 km per months
run_month = runs_time['distance'].resample('M').sum()
# Create plot
fig = plt.figure(figsize=(10, 5))
# Plot and customize
ax = run_month.plot(marker='v', markersize=10, linewidth=0, color='blue')
ax.set(ylim=[0, 140],
       ylabel='Distance (km)',
       xlabel='Months',
       title='Monthly totals for distance')
ax.axhspan(100, 140, color='green', alpha=0.4)
ax.axhspan(80, 100, color='yellow', alpha=0.3)
ax.axhspan(0, 80, color='red', alpha=0.2)
ax.set_xticks(ticks=Months_List)
ax.set_xticklabels(labels=Months_Labels, rotation = 75 )

plt.show()
# Prepare data
hr_zones = [127, 138, 151, 165, 176, 187]
zone_names = ['Easy', 'Moderate', 'Hard', 'Very hard', 'Maximal']
zone_colors = ['green', 'yellow', 'orange', 'tomato', 'red']
run_hr_all = runs_time['average_heartrate']

# Create plot
fig, ax = plt.subplots(figsize = (10,5))

# Plot and customize
n, bins, patches = ax.hist(run_hr_all, bins=hr_zones, alpha=0.5)
for i in range(0, len(patches)):
    patches[i].set_facecolor(zone_colors[i])

ax.set(title='Distribution of HR', ylabel='Number of runs')
ax.xaxis.set(ticks=hr_zones)
#ax.set_xticklabels(labels = zone_names,rotation = -30, ha = 'left' )

# Show plot
plt.show()
import statsmodels.api as sm

def BuildTrend(col='distance'):
    # Prepare data
    run_wkly = runs_time.resample('W')[col].bfill()
    decomposed = sm.tsa.seasonal_decompose(run_wkly, extrapolate_trend=1, period=52)

    # Create plot
    fig = plt.figure(figsize=(12, 5))

    # Plot and customize
    ax = decomposed.trend.plot(label='Trend', linewidth=2)
    ax = decomposed.observed.plot(label='Observed', linewidth=0.5)
    ax.legend()
    ax.set_title(col)
    plt.show()
BuildTrend(col='distance')
BuildTrend(col='average_heartrate')
BuildTrend(col='max_heartrate')
