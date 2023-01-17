#from importlib import reload

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

matplotlib.rcParams['figure.figsize']=(10,18)

%matplotlib inline

from datetime import datetime

from datetime import date

import xgboost as xgb

from sklearn.cluster import MiniBatchKMeans

import seaborn as sns # plot beautiful charts

import warnings

sns.set()

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv', parse_dates=['pickup_datetime'])# `parse_dates` will recognize the column is date time

test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv', parse_dates=['pickup_datetime'])

data.head(3)
data.info()
for df in (data,test):

    df['year']  = df['pickup_datetime'].dt.year

    df['month'] = df['pickup_datetime'].dt.month

    df['day']   = df['pickup_datetime'].dt.day

    df['hr']    = df['pickup_datetime'].dt.hour

    df['minute']= df['pickup_datetime'].dt.minute

    df['store_and_fwd_flag'] = 1 * (df.store_and_fwd_flag.values == 'Y')
test.head(3)
data = data.assign(log_trip_duration = np.log(data.trip_duration+1))
from datetime import datetime

holiday = pd.read_csv('../input/nyc2016holidays/NYC_2016Holidays.csv',sep=';')

holiday['Date'] = holiday['Date'].apply(lambda x: x + ' 2016')

holidays = [datetime.strptime(holiday.loc[i,'Date'], '%B %d %Y').date() for i in range(len(holiday))]
time_data = pd.DataFrame(index = range(len(data)))

time_test = pd.DataFrame(index = range(len(test)))
from datetime import date

def restday(yr,month,day,holidays):

    '''

    Output:

        is_rest: a list of Boolean variable indicating if the sample occurred in the rest day.

        is_weekend: a list of Boolean variable indicating if the sample occurred in the weekend.

    '''

    is_rest    = [None]*len(yr)

    is_weekend = [None]*len(yr)

    i=0

    for yy,mm,dd in zip(yr,month,day):        

        is_weekend[i] = date(yy,mm,dd).isoweekday() in (6,7)

        is_rest[i]    = is_weekend[i] or date(yy,mm,dd) in holidays 

        i+=1

    return is_rest,is_weekend
rest_day,weekend = restday(data.year,data.month,data.day,holidays)

time_data = time_data.assign(rest_day=rest_day)

time_data = time_data.assign(weekend=weekend)



rest_day,weekend = restday(test.year,test.month,test.day,holidays)

time_test = time_test.assign(rest_day=rest_day)

time_test = time_test.assign(weekend=weekend)
time_data = time_data.assign(pickup_time = data.hr+data.minute/60)#float value. E.g. 7.5 means 7:30 am

time_test = time_test.assign(pickup_time = test.hr+test.minute/60)
time_data.head()
fastrout1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv',

                        usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps','step_direction'])

fastrout2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv',

                        usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps','step_direction'])

fastrout = pd.concat((fastrout1,fastrout2))

fastrout.head()
right_turn = []

left_turn = []

right_turn+= list(map(lambda x:x.count('right')-x.count('slight right'),fastrout.step_direction))

left_turn += list(map(lambda x:x.count('left')-x.count('slight left'),fastrout.step_direction))
osrm_data = fastrout[['id','total_distance','total_travel_time','number_of_steps']]

osrm_data = osrm_data.assign(right_steps=right_turn)

osrm_data = osrm_data.assign(left_steps=left_turn)

osrm_data.head(3)
data = data.join(osrm_data.set_index('id'), on='id')

data.head(3)
osrm_test = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv')

right_turn= list(map(lambda x:x.count('right')-x.count('slight right'),osrm_test.step_direction))

left_turn = list(map(lambda x:x.count('left')-x.count('slight left'),osrm_test.step_direction))



osrm_test = osrm_test[['id','total_distance','total_travel_time','number_of_steps']]

osrm_test = osrm_test.assign(right_steps=right_turn)

osrm_test = osrm_test.assign(left_steps=left_turn)

osrm_test.head(3)
test = test.join(osrm_test.set_index('id'), on='id')
osrm_test.head()
osrm_data = data[['total_distance','total_travel_time','number_of_steps','right_steps','left_steps']]

osrm_test = test[['total_distance','total_travel_time','number_of_steps','right_steps','left_steps']]
def haversine_array(lat1, lng1, lat2, lng2):

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371  # in km

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h



def dummy_manhattan_distance(lat1, lng1, lat2, lng2):

    a = haversine_array(lat1, lng1, lat1, lng2)

    b = haversine_array(lat1, lng1, lat2, lng1)

    return a + b



def bearing_array(lat1, lng1, lat2, lng2):

    lng_delta_rad = np.radians(lng2 - lng1)

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    y = np.sin(lng_delta_rad) * np.cos(lat2)

    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)

    return np.degrees(np.arctan2(y, x))
List_dist = []

for df in (data,test):

    lat1, lng1, lat2, lng2 = (df['pickup_latitude'].values, df['pickup_longitude'].values, 

                              df['dropoff_latitude'].values,df['dropoff_longitude'].values)

    dist = pd.DataFrame(index=range(len(df)))

    dist = dist.assign(haversind_dist = haversine_array(lat1, lng1, lat2, lng2))

    dist = dist.assign(manhattan_dist = dummy_manhattan_distance(lat1, lng1, lat2, lng2))

    dist = dist.assign(bearing = bearing_array(lat1, lng1, lat2, lng2))

    List_dist.append(dist)

Other_dist_data,Other_dist_test = List_dist
coord_pickup = np.vstack((data[['pickup_latitude', 'pickup_longitude']].values,                  

                          test[['pickup_latitude', 'pickup_longitude']].values))

coord_dropoff = np.vstack((data[['dropoff_latitude', 'dropoff_longitude']].values,                  

                           test[['dropoff_latitude', 'dropoff_longitude']].values))
coords = np.hstack((coord_pickup,coord_dropoff))# 4 dimensional data

sample_ind = np.random.permutation(len(coords))[:500000]

kmeans = MiniBatchKMeans(n_clusters=10, batch_size=10000).fit(coords[sample_ind])

for df in (data,test):

    df.loc[:, 'pickup_dropoff_loc'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude',

                                                         'dropoff_latitude','dropoff_longitude']])
kmean10_data = data[['pickup_dropoff_loc']]

kmean10_test = test[['pickup_dropoff_loc']]
plt.figure(figsize=(16,16))

N = 500

for i in range(10):

    plt.subplot(4,3,i+1)

    tmp_data = data[data.pickup_dropoff_loc==i]

    drop = plt.scatter(tmp_data['dropoff_longitude'][:N], tmp_data['dropoff_latitude'][:N], s=10, lw=0, alpha=0.5,label='dropoff')

    pick = plt.scatter(tmp_data['pickup_longitude'][:N], tmp_data['pickup_latitude'][:N], s=10, lw=0, alpha=0.4,label='pickup')    

    plt.xlim([-74.05,-73.75]);plt.ylim([40.6,40.9])

    plt.legend(handles = [pick,drop])

    plt.title('clusters %d'%i)

#plt.axes().set_aspect('equal')
weather = pd.read_csv('../input/knycmetars2016/KNYC_Metars.csv', parse_dates=['Time'])

weather.head(3)
print('The Events has values {}.'.format(str(set(weather.Events))))
weather['snow']= 1*(weather.Events=='Snow') + 1*(weather.Events=='Fog\n\t,\nSnow')

weather['year'] = weather['Time'].dt.year

weather['month'] = weather['Time'].dt.month

weather['day'] = weather['Time'].dt.day

weather['hr'] = weather['Time'].dt.hour

weather = weather[weather['year'] == 2016][['month','day','hr','Temp.','Precip','snow','Visibility']]
weather.head()
data = pd.merge(data, weather, on = ['month', 'day', 'hr'], how = 'left')

test = pd.merge(test, weather, on = ['month', 'day', 'hr'], how = 'left')
weather_data = data[['Temp.','Precip','snow','Visibility']]

weather_test = test[['Temp.','Precip','snow','Visibility']]
weather_data.head()
outliers=np.array([False]*len(data))
print('There are %d rows that have missing values'%sum(data.isnull().any(axis=1)))
y = np.array(data.log_trip_duration)

plt.subplot(131)

plt.plot(range(len(y)),y,'.');plt.ylabel('log_trip_duration');plt.xlabel('index');plt.title('val vs. index')

plt.subplot(132)

sns.boxplot(y=data.log_trip_duration)

plt.subplot(133)

sns.distplot(y,bins=50, color="m");plt.yticks([]);plt.xlabel('log_trip_duration');plt.title('data');plt.ylabel('frequency')

#plt.hist(y,bins=50);
outliers[y>12]=True

print('There are %d entries that have trip duration too long'% sum(outliers))
kph = osrm_data.total_distance/1000/data.trip_duration*3600

plt.plot(range(len(kph)),kph,'.');plt.ylabel('kph');plt.xlabel('index');plt.show()
fig=plt.figure(figsize=(10, 8))

for i,loc in enumerate((['pickup_longitude','dropoff_longitude'],['pickup_latitude','dropoff_latitude'])):

    plt.subplot(1,2,i+1)

    sns.boxplot(data=data[outliers==False],order=loc);#plt.title(loc)
outliers[data.pickup_longitude<-110]=True

outliers[data.dropoff_longitude<-110]=True

outliers[data.pickup_latitude>45]=True

print('There are total %d entries of ouliers'% sum(outliers))
for i,feature in enumerate(['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude']):

    plt.subplot(2,2,i+1)

    data[outliers==False][feature].hist(bins=100)

    plt.xlabel(feature);plt.ylabel('frequency');plt.yticks([])

#plt.show();plt.close()
import folium # goelogical map

map_1 = folium.Map(location=[40.767937,-73.982155 ],tiles='OpenStreetMap',

 zoom_start=12)

#tile: 'OpenStreetMap','Stamen Terrain','Mapbox Bright','Mapbox Control room'

for each in data[:1000].iterrows():

    folium.CircleMarker([each[1]['pickup_latitude'],each[1]['pickup_longitude']],

                        radius=3,

                        color='red',

                        popup=str(each[1]['pickup_latitude'])+','+str(each[1]['pickup_longitude']),

                        fill_color='#FD8A6C'

                        ).add_to(map_1)

map_1
print('I assign %d points as outliers'%sum(outliers))
Out = pd.DataFrame()

Out = Out.assign(outliers=outliers)
tmpdata = data

tmpdata = pd.concat([tmpdata,time_data],axis = 1)

tmpdata = tmpdata[outliers==False]
fig=plt.figure(figsize=(18, 8))

sns.boxplot(x="hr", y="log_trip_duration", data=tmpdata)
sns.violinplot(x="month", y="log_trip_duration", hue="rest_day", data=tmpdata, split=True,inner="quart")
tmpdata.head(3)
sns.violinplot(x="pickup_dropoff_loc", y="log_trip_duration",

               hue="rest_day", 

               data=tmpdata[np.array(tmpdata.pickup_time>7) & np.array(tmpdata.pickup_time<9)], 

               split=True,inner="quart")
mydf = data[['vendor_id','passenger_count','pickup_longitude', 'pickup_latitude',

       'dropoff_longitude', 'dropoff_latitude','store_and_fwd_flag']]



testdf = test[['vendor_id','passenger_count','pickup_longitude', 'pickup_latitude',

       'dropoff_longitude', 'dropoff_latitude','store_and_fwd_flag']]
print('There are %d samples in the test data'%len(testdf))
kmean_data= pd.get_dummies(kmean10_data.pickup_dropoff_loc,prefix='loc', prefix_sep='_')    

kmean_test= pd.get_dummies(kmean10_test.pickup_dropoff_loc,prefix='loc', prefix_sep='_')    
mydf  = pd.concat([mydf  ,time_data,weather_data,osrm_data,Other_dist_data,kmean_data],axis=1)

testdf= pd.concat([testdf,time_test,weather_test,osrm_test,Other_dist_test,kmean_test],axis=1)
if np.all(mydf.keys()==testdf.keys()):

    print('Good! The keys of training feature is identical to those of test feature')

else:

    print('Oops, something is wrong, keys in training and testing are not matching')    
X = mydf[Out.outliers==False]

z = data[Out.outliers==False].log_trip_duration.values
if np.all(X.keys()==testdf.keys()):

    print('Good! The keys of training feature is identical to those of test feature.')

    print('They both have %d features, as follows:'%len(X.keys()))

    print(list(X.keys()))

else:

    print('Oops, something is wrong, keys in training and testing are not matching')
import xgboost as xgb
X = X[:50000]

z = z[:50000]
from sklearn.model_selection import train_test_split



#%% split training set to validation set

Xtrain, Xval, Ztrain, Zval = train_test_split(X, z, test_size=0.2, random_state=0)

Xcv,Xv,Zcv,Zv = train_test_split(Xval, Zval, test_size=0.5, random_state=1)

data_tr  = xgb.DMatrix(Xtrain, label=Ztrain)

data_cv  = xgb.DMatrix(Xcv   , label=Zcv)

evallist = [(data_tr, 'train'), (data_cv, 'valid')]
parms = {'max_depth':8, #maximum depth of a tree

         'objective':'reg:linear',

         'eta'      :0.3,

         'subsample':0.8,#SGD will use this percentage of data

         'lambda '  :4, #L2 regularization term,>1 more conservative 

         'colsample_bytree ':0.9,

         'colsample_bylevel':1,

         'min_child_weight': 10,

         'nthread'  :3}  #number of cpu core to use



model = xgb.train(parms, data_tr, num_boost_round=1000, evals = evallist,

                  early_stopping_rounds=30, maximize=False, 

                  verbose_eval=100)



print('score = %1.5f, n_boost_round =%d.'%(model.best_score,model.best_iteration))
data_test = xgb.DMatrix(testdf)

ztest = model.predict(data_test)
ytest = np.exp(ztest)-1

print(ytest[:10])
submission = pd.DataFrame({'id': test.id, 'trip_duration': ytest})

# submission.to_csv('submission.csv', index=False)
fig =  plt.figure(figsize = (15,15))

axes = fig.add_subplot(111)

xgb.plot_importance(model,ax = axes,height =0.5)

plt.show();plt.close()