# importing the required libraries 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
from datetime import datetime # to tackle the date time features
from datetime import date
from sklearn.cluster import MiniBatchKMeans
import seaborn as sns #pretty plots
import warnings
sns.set()


#reading the training and testing data and storing them as data and test respectively.
data =  pd.read_csv('../input/new-york-city-taxi-with-osrm/train.csv', parse_dates =['pickup_datetime'])
dates=['pickup_datetime']
test = pd.read_csv('../input/new-york-city-taxi-with-osrm/test.csv',parse_dates = ['pickup_datetime'])
data.head(5)
# checking the types of columns and entries in the data and test.
data.info()
test.info()
#using the describe() function to get and idea about different features and their stats
data.describe()
#getting individual components of date time
#converting store_and_fwd_flag to a 0 and 1 from a categorical data as only 2 possible values.
for df in (data,test):
    df['year'] = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day
    df['hr'] = df['pickup_datetime'].dt.hour
    df['minute'] = df['pickup_datetime'].dt.minute
    #converting flag into 0 or 1
    df['store_and_fwd_flag'] = 1 * (df.store_and_fwd_flag.values== 'Y')

 
print(data[data['pickup_longitude']<-74.2].shape[0]) #points with longitude out of range, possible outliers
print((data['vendor_id']==2).mean())  #Distribution by vendor id
print((data['store_and_fwd_flag']==1).mean()) #distribution by store_and_fwd_flag
print(data[data['store_and_fwd_flag']==1].shape[0])


#converting trip duration to log(trip_duration +1) for rmsle
data = data.assign(log_trip_duration = np.log(data.trip_duration+1)) 

#since ride duration will depend on holiday or workday etc importing nyc 2016 holiday dataset
holiday = pd.read_csv('../input/nyc2016holidays/NYC_2016Holidays.csv', sep = ';') # secondary dataset for restdays
holiday['Date'] = holiday['Date'].apply(lambda x: x+ '2016')
holidays = [datetime]
#storing restday features into new data frames (will merge them later)
time_data = pd.DataFrame(index = range(len(data)))  
time_test = pd.DataFrame(index = range(len(test)))
# we will list out the weekends and rest days(holidays)
# will return if its a holiday or weekend

from datetime import date
def find_holiday(yr, month, day, holidays):
    holiday =  [None]*len(yr)
    weekend =  [None]*len(yr)
    i = 0 
    for yy,mm,dd in zip(yr, month, day):
        #checking for saturday sunday(6,7), date.isoweekday() returns the day
        weekend[i] = date(yy,mm,dd).isoweekday() in (6,7)
        holiday[i] = weekend[i] or date(yy,mm,dd)  in holidays
        i+=1
    return holiday, weekend
    
#for training data, creating rest_day and weekend columns in a new dataframe (will merge later)
rest_day,weekend = find_holiday(data.year, data.month, data.day, holidays)
time_data = time_data.assign(rest_day = rest_day)
time_data = time_data.assign(weekend = weekend)

#replicating for test data
rest_day,weekend = find_holiday(test.year, test.month, test.day, holidays)
time_test = time_test.assign(rest_day = rest_day)
time_test = time_test.assign(weekend = weekend)
#changing time into floats 
time_data = time_data.assign(pickup_time = data.hr+data.minute/60)
time_test = time_test.assign(pickup_time = test.hr+ test.minute/60)

#using OSRM feature of fastroute (its split into 2 files)
fastrout1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv', usecols=['id','total_distance','total_travel_time','number_of_steps','step_direction'])
fastrout2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv', usecols=['id','total_distance','total_travel_time','number_of_steps','step_direction'])
#merging data from the 2 fastrout files
fastrout = pd.concat((fastrout1,fastrout2))
fastrout.head()
#from the data calculating the number for strict lefts and rights
right_turn = []
left_turn = []
right_turn+= list(map(lambda x:x.count('right')-x.count('slight right'), fastrout.step_direction))
left_turn+= list(map(lambda x:x.count('left')-x.count('slight left'), fastrout.step_direction))
# storing the osrm features in a new dataframe (will merge later)
osrm_data = fastrout[['id', 'total_distance', 'total_travel_time','number_of_steps']]
osrm_data = osrm_data.assign(right_steps = right_turn)
osrm_data = osrm_data.assign(left_steps = left_turn)
osrm_data.head(3)
# making osrm data compatible to use by mapping ids
data = data.join(osrm_data.set_index('id'), on = 'id')
data.head(3)


#replicating for test data
osrm_test = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv')
right_turn= list(map(lambda x:x.count('right')-x.count('slight right'),osrm_test.step_direction))
left_turn = list(map(lambda x:x.count('left')-x.count('slight left'),osrm_test.step_direction))

osrm_test = osrm_test[['id','total_distance','total_travel_time','number_of_steps']]
osrm_test = osrm_test.assign(right_steps=right_turn)
osrm_test = osrm_test.assign(left_steps=left_turn)
osrm_test.head(3)
test = test.join(osrm_test.set_index('id'), on='id')
osrm_test.head(3)
# final osrm features
osrm_data = data[['total_distance','total_travel_time','number_of_steps','right_steps','left_steps']]
osrm_test = test[['total_distance','total_travel_time','number_of_steps','right_steps','left_steps']]
# DISTANCE METRICS
#haversine manhattan and bearing distance
def haversine_array(lat1, lng1, lat2, lng2):
    lat1,lng1,lat2,lng2 = map(np.radians, (lat1,lng1,lat2,lng2))
    earth_radius = 6371 #km
    lat = lat2-lat1
    lng = lng2-lng1
    d = np.sin(lat*0.5) **2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng *0.5)**2
    h = 2 * earth_radius * np.arcsin(np.sqrt(d))
    return h

def manhattan_distance(lat1, lat2, lng1, lng2):
    a = haversine_array(lat1,lng1,lat1,lng2)
    b = haversine_array(lat1,lng1,lat2,lng1)
    return a+b

def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))
    
    
#calculating and saving distance features in a new dataframe

List_dist = []
for df in (data,test):
    lat1, lng1, lat2, lng2 = (df['pickup_latitude'].values, df['pickup_longitude'].values, 
                              df['dropoff_latitude'].values,df['dropoff_longitude'].values)
    dist = pd.DataFrame(index=range(len(df)))
    dist = dist.assign(haversine_dist = haversine_array(lat1, lng1, lat2, lng2))
    dist = dist.assign(manhattan_dist = manhattan_distance(lat1, lng1, lat2, lng2))
    dist = dist.assign(bearing = bearing_array(lat1, lng1, lat2, lng2))
    List_dist.append(dist)
Other_dist_data,Other_dist_test = List_dist
##### using k means to make clusters of locations
coord_pickup = np.vstack((data[['pickup_latitude', 'pickup_longitude']].values, test[['pickup_latitude','pickup_longitude']].values))
coord_dropoff = np.vstack((data[['dropoff_latitude', 'dropoff_longitude']].values, test[['dropoff_latitude','dropoff_longitude']].values))

# creating a 4d data for k means
coords = np.hstack((coord_pickup,coord_dropoff))
sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=10, batch_size =10000).fit(coords[sample_ind])
for df in (data,test):
    df.loc[:,'pickup_dropoff_loc'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude',
                                                         'dropoff_latitude','dropoff_longitude']])
# saving location clusters
kmean_data = data[['pickup_dropoff_loc']]
kmean_test = test[['pickup_dropoff_loc']]

# WEATHER DATA
weather = pd.read_csv('../input/knycmetars2016/KNYC_Metars.csv', parse_dates=['Time'])
weather.head(5)
#possible events
print(set(weather.Events))
#assuming snow to play a role with or without fog 
weather['snow'] = 1*(weather.Events == 'Snow') + 1*(weather.Events == 'Fog\n\t,\nSnow')
weather['year'] = weather['Time'].dt.year
weather['month'] = weather['Time'].dt.month
weather['day'] = weather['Time'].dt.day
weather['hr'] = weather['Time'].dt.hour
weather = weather[weather['year']== 2016][['month','day','hr','Temp.','Precip','snow', 'Visibility']]
weather.head(3)
# merging weather data by mapping the datetime stamps
data = pd.merge(data, weather, on=['month', 'day', 'hr'], how = 'left')
test = pd.merge(test, weather, on=['month', 'day', 'hr'], how = 'left')
#weather features stored in a new data frame
weather_data = data[['Temp.','Precip','snow','Visibility']]
weather_test = test[['Temp.','Precip','snow','Visibility']]
weather_data.head()
#DATA Visualization

# Calculate the manhattan distance and speed
manhattan = manhattan_distance(data['pickup_latitude'],data['dropoff_latitude'],data['pickup_longitude'],data['dropoff_longitude'])
manhattan_speed = manhattan / data['trip_duration']

# Plot speed against the time of day
sns.set(font_scale=1.2)
plt.figure(figsize=(12, 5))
sns.boxplot(data['hr'], manhattan_speed, showfliers=False)
plt.title('Average speeds vs. time of day')
plt.ylabel('Average Speed')
plt.xlabel('Hour')
plt.show()
# distributionof trip duration
sns.set(font_scale=1.2)
plt.figure(figsize=(6, 5))
plt.hist(sorted(data['log_trip_duration']), bins=100, normed=True)
plt.title('Distribution curve of rides w.r.t Duration(in log)')
plt.ylabel('Ride Distribution')
plt.xlabel('log_trip_duration')
plt.show()
#Workhours time
tempdata = data
tempdata = pd.concat([tempdata, time_data], axis=1)
#fig = plt.figure(figsize = (18,8))
sns.set(font_scale=1.2)
plt.figure(figsize=(18, 8))
sns.boxplot(x = "hr", y="log_trip_duration", data = data, showfliers = False)
plt.title("log_trip_duration vs hr of the day")
#month vs trip_duration
sns.violinplot(x = "month", y="log_trip_duration", hue = "rest_day", data=tempdata, split=True, inner = "quart")
plt.title("Comparision of log_trip_duration vs month for rest_day and working day")

#THE XGBOOST
# dropping off the target variables
mydf = data[['vendor_id','passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','store_and_fwd_flag']]
testdf = test[['vendor_id','passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','store_and_fwd_flag']]
#converting kmeans data into one hot encoding for XGBOOSTING
kmeans_data = pd.get_dummies(kmean_data.pickup_dropoff_loc, prefix='loc', prefix_sep = '_')
kmeans_test = pd.get_dummies(kmean_test.pickup_dropoff_loc, prefix='loc', prefix_sep = '_')
#merging the entire feature space
mydf  = pd.concat([mydf  ,time_data,weather_data,osrm_data,Other_dist_data,kmeans_data],axis=1)
testdf= pd.concat([testdf,time_test,weather_test,osrm_test,Other_dist_test,kmeans_test],axis=1)
# checking if test and training sets are similar
print(np.all(mydf.keys()==testdf.keys()))

X = mydf
z = data['log_trip_duration'].values

# verifying if the keys are similar
print(X.keys() == testdf.keys())
import xgboost as xgb
from sklearn.model_selection import train_test_split
# 80-20 split for training and validation
Xtrain, Xval, Ztrain, Zval = train_test_split(X, z, test_size=0.2, random_state =42)
#50-50 validation set split
Xcv,Xv,Zcv,Zv = train_test_split(Xval, Zval, test_size=0.5, random_state=1)
#DMatrix conversion for xgb
data_tr  = xgb.DMatrix(Xtrain, label=Ztrain)
data_cv  = xgb.DMatrix(Xcv   , label=Zcv)
evallist = [(data_tr, 'train'), (data_cv, 'valid')]

parms = {'max_depth':15, #maximum depth of a tree 8 12
         'objective':'reg:linear',
         'eta'      :0.05, #0.3
         'subsample':0.9,#SGD will use this percentage of data 0.8 0.99
         'lambda '  :3, #L2 regularization term,>1 more conservative 4 
         'colsample_bytree ':0.6, #0.9
         'colsample_bylevel':0.7, #1 0.7
         'min_child_weight': 0.5, #10 0.5
         #'nthread'  :3 ... default is max cores
         'eval_metric':'rmse'}  #number of cpu core to use
# running for 2k iterations 
model = xgb.train(parms, data_tr, num_boost_round=2000, evals = evallist,
                  early_stopping_rounds=50, maximize=False, 
                  verbose_eval=100)


# The optimization function replacing GridSearchCV implemented with the help of the kaggle kernel of beluga
'''                  

Flag = True
xgb_pars = []
for min_child_weight in [0.5, 1, 2, 4]:
    for ETA in [0.05, 0.1, 0.15]:
        for colsample_bytree in [0.5,0.6,0.7,0.8,0.9]:
            for max_depth in [6, 8, 10, 12, 15]:
                for subsample in [0.5, 0.6, 0.7, 0.8, 0.9]:
                    for lambda in [0.5, 1., 1.5,  2., 3.,4.]:
                        xgb_pars.append({'min_child_weight': min_child_weight, 'eta': ETA,
                                         'colsample_bytree': colsample_bytree, 'max_depth': max_depth,
                                         'subsample': subsample, 'lambda': lambda, 
                                         'eval_metric': 'rmse','silent': 1, 'objective': 'reg:linear'})

while Flag:
    xgb_par = np.random.choice(xgb_pars, 1)[0]
    print(xgb_par)
    model = xgb.train(xgb_par, data_tr, 2000, evallist, early_stopping_rounds=50,maximize=False, verbose_eval=100)
    print('Modeling RMSLE %.6f' % model.best_score)
'''
#the final score achieved by the set parameters and the iterations required for achieveing it (early stopping) 

print('score = %1.5f, n_boost_round =%d.'%(model.best_score,model.best_iteration))
# Feature imporatnce graph using the xgb.plot_importance
fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
xgb.plot_importance(model,ax = axes,height =0.5)
plt.show();plt.close()
# prediction using the model from the test data
data_test = xgb.DMatrix(testdf)
ztest = model.predict(data_test)


ytest = np.exp(ztest)-1
print(ytest[:10])
#creating submission for kaggle
submission = pd.DataFrame({'id': test.id, 'trip_duration': ytest})
submission.to_csv('submission.csv', index=False)
# model evaluation predicted vs actual values on validation set
pred_val = model.predict(xgb.DMatrix(Xval))

sns.set(font_scale=1)
plt.figure(figsize=(7, 7))
plt.scatter(Zval, pred_val, s=3, color='red', alpha=0.025)
plt.plot([1,9],[1,9], color='blue')
plt.title('Predicted values vs actual values on the validation set')
plt.xlabel('log(trip_duration+1)')
plt.ylabel('Model predictions')
axes = plt.gca()
axes.set_xlim([2, 9])
axes.set_ylim([2, 9])
plt.show()
plt.show()
