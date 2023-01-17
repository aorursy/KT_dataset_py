import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

import folium

from sklearn.cluster import KMeans

!pip install geocoder

import geocoder

import json

import time

import datetime

!pip install wget

import wget



from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error
try:

    with open('daily.csv') as us_states_covid19_daily:

        df=pd.read_csv(us_states_covid19_daily)

except IOError: 

    url='https://covidtracking.com/api/v1/states/daily.csv'

    df=pd.read_csv(url)  
df.head()
df.sort_values(['state','date'],inplace=True,ascending=(True,False))

df.head()
states=df['state'].unique()
df=df.drop(['dateChecked','hash','pending','negative','deathIncrease','hospitalizedIncrease','negativeIncrease','positiveIncrease','totalTestResultsIncrease','fips','total','totalTestResults','posNeg'],axis=1)

df.head()
df_dict={}

for i in range(len(states)):

    df_dict[states[i]]=[]

    for j in range(len(df)):

        if df['state'].iloc[j]==states[i]:

            df_dict[states[i]].append(df.iloc[j])
for i in range(len(df_dict)):

    df_dict[states[i]]=pd.DataFrame(df_dict[states[i]])
df_dict['MN']
dates_mn=df_dict['MN']['date']

dates_mn=pd.to_datetime(dates_mn, format='%Y%m%d')

day_num_mn=[i for i in range(len(dates_mn))]

day_num_mn=np.array(day_num_mn).reshape(-1,1)

#mn_dates=pd.Index(df_dict['MN']['date'].unique())
cases_mn=np.array(df_dict['MN']['positive'].tolist())

cases_mn=np.flip(cases_mn).reshape(-1,1)
plt.plot(day_num_mn,cases_mn)

plt.title('Number of People Tested Positive for COVID-19 in MN')

plt.xlabel('Date')

plt.xticks(rotation=90)

plt.ylabel('Number of People')

plt.show()
days_in_future = 10

future_forcast_mn = np.array([i for i in range(len(dates_mn)+days_in_future)]).reshape(-1, 1)
X_train_mn, X_test_mn, y_train_mn, y_test_mn = train_test_split(day_num_mn, cases_mn, test_size=0.1, shuffle=False) 
# use this to find the optimal parameters for SVR

#c = [0.01, 0.1, 1]

#gamma = [0.01,0.015,0.02] #I had to alter this line because my system would crash with larger values

#epsilon = [0.01, 0.1, 1]

#shrinking = [True, False]

#degree = [3, 4, 5, 6, 7]

#

#svm_grid = {'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking, 'degree': degree}

#

#svm = SVR(kernel='poly')

#svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=30, verbose=1)

#svm_search.fit(X_train_mn, np.ravel(y_train_mn))

#

#svm_search.best_params_



#svm_confirmed_mn = svm_search.best_estimator_
#svm_confirmed_mn = svm_search.best_estimator_

svm_confirmed_mn = SVR(shrinking=False, kernel='poly',gamma=0.01, epsilon=0.1,degree=3, C=1)

svm_confirmed_mn.fit(X_train_mn, np.ravel(y_train_mn))

svm_pred_mn = svm_confirmed_mn.predict(future_forcast_mn)
MAE=[]

for i in range(1,10):

    svm_confirmed_degree_test = SVR(shrinking=False, kernel='poly',gamma=0.01, epsilon=0.1,degree=i, C=1)

    svm_confirmed_degree_test.fit(X_train_mn, np.ravel(y_train_mn))

    svm_pred_degree_test = svm_confirmed_degree_test.predict(future_forcast_mn)

    svm_test_pred_degree_test = svm_confirmed_degree_test.predict(X_test_mn)

    MAE.append(mean_absolute_error(svm_test_pred_degree_test, y_test_mn))

degree_num=list(range(1,10))

plt.plot(degree_num,MAE)

plt.title('Tuning Degree Parameter')

plt.ylabel('SVM Error')

plt.xlabel('Degree Number')

plt.show()
plt.plot(day_num_mn, cases_mn)

plt.plot(future_forcast_mn, svm_pred_mn, linestyle='dashed', color='purple')

plt.title('Predicted Increase in Number of Cases')

plt.legend(['Confirmed Cases','SVM Prediction'])

plt.xlabel('Day')

plt.ylabel('Number of Confirmed Cases')

plt.show()
not_states=['AS','GU','DC','PR','MP','VI']

for i in range(len(not_states)):

    del df_dict[not_states[i]]

key_list=list(df_dict.keys())
states_list=[]

for i in range(len(states)):

    if states[i] not in not_states:

        states_list.append(states[i])
dates=[]

for i in range(len(states_list)):

    dates.append(pd.Index(df_dict[states_list[i]]['date'].unique()))

day_num=[]

for i in range(len(dates)):

    day_num.append([j for j in range(len(dates[i]))])

    day_num[i]=np.array(day_num[i]).reshape(-1,1)
day_num_length_list=[]

for i in range(len(day_num)):

    day_num_length_list.append(len(day_num[i]))

    

min_day_num=min(day_num_length_list)
cases=[]

for i in range(len(df_dict)):

    cases.append(np.array(df_dict[states_list[i]]['positive'].tolist()))

    cases[i]=np.flip(cases[i]).reshape(-1,1)
#cases_us=[]

#for day in list(df['date'].unique()):

#    cases_us.append(df[df['date']==day]['positive'].sum())

#cases_us=np.array(cases_us)

#cases_us=np.flip(cases_us).reshape(-1,1)

days_in_future=10

future_forcast=[]

for i in range(len(dates)):

    future_forcast.append(np.array([j for j in range(len(dates[i])+days_in_future)]).reshape(-1, 1))
svm_confirmed = svm_confirmed_mn



X_train=[0 for i in range(len(cases))]

X_test=[0 for i in range(len(cases))]

y_train=[0 for i in range(len(cases))]

y_test=[0 for i in range(len(cases))]

for i in range(len(cases)):

    X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(day_num[i], cases[i], test_size=0.1, shuffle=False)
svm_pred=[0 for i in range(len(cases))]

for i in range(len(cases)):

        svm_confirmed.fit(X_train[i], np.ravel(y_train[i]))

        svm_pred[i] = svm_confirmed.predict(future_forcast[i])
plt.figure(figsize=(8,8),dpi=100)

for i in range(50):

    plt.subplot(10,5,i+1)    

    plt.plot(day_num[i],cases[i])

    plt.title(key_list[i],loc='center',pad=-10,fontsize=10)

    plt.tick_params(

    axis='x',          # changes apply to the x-axis

    which='both',      # both major and minor ticks are affected

    bottom=False,      # ticks along the bottom edge are off

    top=False,         # ticks along the top edge are off

    labelbottom=False)

    plt.tick_params(

    axis='y',          # changes apply to the x-axis

    which='both',      # both major and minor ticks are affected  

    direction='in',

    labelsize=5,

    pad=-20)
plt.figure(figsize=(8,8),dpi=100)

for i in range(50):

    plt.subplot(10,5,i+1)    

    plt.plot(day_num[i],cases[i])

    plt.plot(future_forcast[i], svm_pred[i], linestyle='dashed', color='purple')

    plt.title(states_list[i],loc='center',pad=-10,fontsize=10)

    plt.legend(['Actual','Predicted'],fontsize=3)

    plt.tick_params(

    axis='x',          # changes apply to the x-axis

    which='both',      # both major and minor ticks are affected

    bottom=False,      # ticks along the bottom edge are off

    top=False,         # ticks along the top edge are off

    labelbottom=False)

    plt.tick_params(

    axis='y',         

    which='both',      

    direction='in',

    labelsize=5,

    pad=-20)
df_cum=df.sort_index()

df_cum.head()
df_cum=df_cum.iloc[0:56]
#get rid of rows that are us territories, not states: AS, GU, DC, PR, MP, VI

drop_ind=[3,8,12,27,42,50]

df_cum.drop(index=drop_ind,inplace=True)
df_cum.reset_index(drop=True,inplace=True)

#df_cum['positive'].astype('int',inplace=True)
state_names=['Alaska','Alabama','Arkansas','Arizona','California','Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Iowa', 'Indiana','Idaho', 'Illinois', 'Kansas', 'Kentucky', 'Louisiana', 'Massachusetts', 'Maryland','Maine','Michigan', 'Minnesota', 'Missouri','Mississippi', 'Montana', 'North Carolina', 'North Dakota','Nebraska', 'New Hampshire','New Jersey','New Mexico','Nevada','New York', 'Ohio','Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina','South Dakota', 'Tennessee', 'Texas', 'Utah','Virginia','Vermont', 'Washington', 'Wisconsin','West Virginia', 'Wyoming']
df_cum['State_Names']=pd.DataFrame(state_names)
df_cum=df_cum[['date','state','State_Names','positive','hospitalizedCurrently','inIcuCurrently','inIcuCumulative','onVentilatorCurrently','onVentilatorCumulative','recovered','death','hospitalized']]
df_cum
try:

    with open('us-states.json') as jsonfile:

        json_path=jsonfile.name

except IOError:

    #import wget

    us_states_json_url='https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json'

    us_states_json=wget.download(us_states_json_url)

    json_path=us_states_json
us_map = folium.Map(location=[37,-96], zoom_start=4)
bins = list(df_cum['positive'].quantile([0, 0.25, 0.5, 0.75, 1]))

bins
folium.Choropleth(

    geo_data=json_path,

    data=df_cum,

    columns=['State_Names', 'positive'],

    key_on='feature.properties.name',

    fill_color='YlGn', 

    fill_opacity=0.7, 

    line_opacity=0.2,

    legend_name='COVID-19',

    bins=bins,

    reset=True

).add_to(us_map)





# display map

us_map
try:

    with open('time_series_covid19_confirmed_US.csv') as timeseries:

        df_counties=pd.read_csv(timeseries)

except IOError:

    counties_url='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'

    df_counties=pd.read_csv(counties_url)
df_counties.head(10)
df_counties=df_counties.drop(['UID','iso2','iso3','code3','FIPS'],axis=1)

df_counties.head(10)
most_recent_day='4/11/20'

df_counties_small=df_counties[['Lat','Long_',most_recent_day]]

df_counties_small.head()
kclusters=100

kmeans=KMeans(n_clusters=kclusters,random_state=0).fit(df_counties_small)

centers=kmeans.cluster_centers_
lats=pd.Series(centers[:,0])

longs=pd.Series(centers[:,1])

weights=centers[:,2]
cluster_map=folium.Map(location=[37,-96], zoom_start=4)

for lat,long in zip(lats,longs):

    folium.CircleMarker(

            [lat,long],

            radius=5).add_to(cluster_map)



cluster_map
try:

    with open('us-counties.csv') as uscounties:

        df_counties2=pd.read_csv(uscounties)

except IOError:

    counties2_url='https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'

    df_counties2=pd.read_csv(counties2_url)

df_counties2.head()
for i in range(len(df_counties2)):

    if df_counties2['county'][i]!='Unknown':

        df_counties2.at[i,'county, state']=df_counties2['county'][i]+', '+df_counties2['state'][i]

    elif df_counties2['county'][i]=='Unknown':

        df_counties2.at[i,'county, state']=df_counties2['state'][i]

df_counties2.tail()
county_state_list=list(df_counties2['county, state'].unique())

county_state_list

len(county_state_list)
API_KEY='Put Your Api key here'

#g_total=[]

#for name in county_state_list[0:1000]:

#    g_current = geocoder.google(name,key=API_KEY)

#    g_total.append(g_current)
#for name in county_state_list[1000:2000]:

#    g_current = geocoder.google(name,key=API_KEY)

#    g_total.append(g_current)
#for name in county_state_list[2573:len(county_state_list)]:

#    g_current = geocoder.google(name,key=API_KEY)

#    g_total.append(g_current)
#g_dict={}

#g_dict['counties']=[]

#for i in range(len(g_total)):

 #   if type(g_total[i])==type(g_current):

  #      g_dict['counties'].append(g_total[i].json)

   # else:

    #    g_dict['counties'].append(g_total[i])
#with open('county_lat_lng.txt', 'w') as outfile:

 #   json.dump(g_dict, outfile)
#for i in range(len(g_total)):

 #   if type(g_total[i])!=dict:

  #      print(i,type(g_total[i]))
jsonpath='../input/county-lat-lng-json/county_lat_lng.json'

with open(jsonpath,"r") as json_file:

    g_dict = json.load(json_file)
g_total=g_dict['counties']
# g_current = geocoder.google('Minneasota',key=API_KEY)

g_current=0

lat_long=[]

for i in range(len(g_total)):

    if type(g_total[i])==type(g_current):

        lat_long.append([g_total[i].lat, g_total[i].lng])

    else:

        lat_long.append([g_total[i]['lat'], g_total[i]['lng']])

    
#changing lat_long to a numpy array since it's easier to get columns and rows out

lat_long=np.array([[i[0] for i in lat_long],[i[1] for i in lat_long]])
county_dict={}

for i in range(len(county_state_list)):

    county_dict.update({county_state_list[i]:tuple(lat_long[:,i])})
lat_series=[]

long_series=[]

for i in range(len(df_counties2)):

    lat_series.append(county_dict[df_counties2['county, state'][i]][0])

    long_series.append(county_dict[df_counties2['county, state'][i]][1])
df_counties2['latitude']=lat_series

df_counties2['longitude']=long_series

df_counties2.tail()
most_recent_day2='2020-04-11'

df_counties_curr=df_counties2[df_counties2['date']==most_recent_day2]

df_counties_curr.head()

heatmap_list=df_counties_curr[['latitude','longitude','cases']].values.tolist()
heatmap_list
from folium.plugins import HeatMap

base_map=folium.Map(location=[37,-96], zoom_start=4)

HeatMap(data=heatmap_list,radius=10).add_to(base_map)

base_map    
heatmap_list_time=[]

for date in df_counties2['date'].unique():

    df_counties_curr=df_counties2[df_counties2['date']==date]

    heatmap_list_time.append(df_counties_curr[['latitude','longitude','cases']].values.tolist())
from folium.plugins import HeatMapWithTime

base_map=folium.Map(location=[37,-96], zoom_start=4)

HeatMapWithTime(data=heatmap_list_time, radius=10,auto_play=True).add_to(base_map)

base_map