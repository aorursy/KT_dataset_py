# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt # to handle dates and time

from datetime import datetime, timedelta, date



from functools import reduce



# data visualization

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

#import cufflinks as cf

import seaborn as sns

sns.set_style('darkgrid')

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# sklearn

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline, FeatureUnion



from sklearn.preprocessing import Imputer, OneHotEncoder, LabelBinarizer, StandardScaler



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV



from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

import xgboost as xgb
def mean_cross_score(x):

    print("Accuracy: %0.3f +/- %0.3f" % (np.mean(x), np.std(x)))
class DataFrameSelector(BaseEstimator, TransformerMixin): # to select dataframes in a pipeline.

    def __init__(self, attributes): 

        self.attributes = attributes

    def fit(self, df, y=None): 

        return self

    def transform(self, df):

        return df[self.attributes].values
def RemoveOutliers(df,cols,n_sigma): # keep only instances that are within p\m n_sigma in columns cols

    new_df = df.copy()

    for col in cols:

        new_df = new_df[np.abs(new_df[col]-new_df[col].mean())<=(n_sigma*new_df[col].std())]

    print('%i instances have been removed' %(df.shape[0]-new_df.shape[0]))

    return new_df
def my_pipeline(df, func_list):

    new_df = df.copy()

    return reduce(lambda x, func: func(x), func_list, new_df)
train_df = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')

test_df = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')



holidays_df = pd.read_csv('../input/nyc2016holidays/NYC_2016Holidays.csv', sep=';')
# Geo locations taken from Google maps

NYC = np.array([-74.0059,40.7128]) # google maps coordinates of NYC



fifth_ave = np.array([0.58926996811979,0.8079362008674332]) # versor of Fifth Av. digitized from google maps

ort_fifth_ave = np.array([-0.8079362008674332,0.58926996811979]) # orthogonal versor



EastRiver = np.array([-73.955921,40.755157])

HudsonRiver = np.array([-74.012226,40.755677])

LeftBound = np.array([-74.020485,40.701463])

RightBound = np.array([-73.932614,40.818593])
train_df.info()
train_df['id'].value_counts().shape
train_df['store_and_fwd_flag'].value_counts()
train_df.describe()
plt.figure(figsize=(12,4))



plt.subplot(1,2,1)

data = train_df['vendor_id'].value_counts().sort_index()

data.plot(kind='bar')

plt.xlabel('vendor_id')

plt.ylabel('events')



plt.subplot(1,2,2)

data = train_df['passenger_count'].value_counts().sort_index()

data.plot(kind='bar')

plt.xlabel('passenger_count')



plt.show()
plt.figure(figsize=(10,10))



plt.scatter(x=train_df['pickup_longitude'].values,y=train_df['pickup_latitude'].values, marker='^',s=1,alpha=.3)

plt.xlim([-74.1,-73.7])

plt.ylim([40.6, 40.9])

plt.axis('off')



#plt.scatter(x=train_df['dropoff_longitude'].values,y=train_df['dropoff_latitude'].values, marker='v',s=1,alpha=.1)

#plt.xlim([-74.05,-73.75])

#plt.ylim([40.6, 40.9])

#plt.axis('off')



plt.show()
clean_att = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']

train_df_clean = RemoveOutliers(train_df,clean_att,5)
train_df_clean['trip_duration'] = np.log(train_df_clean['trip_duration'])
plt.figure(figsize=(8,6))



plt.hist(train_df_clean['trip_duration'], bins=100)

#plt.yscale('log', nonposy='clip')

plt.xlabel('Trip duration (log)')

plt.ylabel('events')



plt.show()
clean_att = ['trip_duration']

train_df_clean = RemoveOutliers(train_df_clean,clean_att,5)
plt.figure(figsize=(8,6))



plt.hist(train_df_clean['trip_duration'], bins=100)

#plt.yscale('log', nonposy='clip')

plt.xlabel('Trip duration (log)')

plt.ylabel('events')



plt.show()
num_att = [f for f in train_df_clean.columns if train_df_clean.dtypes[f] != 'object']

cat_att = [f for f in train_df_clean.columns if train_df_clean.dtypes[f] == 'object']

print("-"*10+" numerical attributes "+"-"*10)

print(num_att)

print('')

print("-"*10+" categorical attributes "+"-"*10)

print(cat_att)
train_df_clean['pickup_datetime'] = pd.to_datetime(train_df_clean['pickup_datetime'])

train_df_clean['dropoff_datetime'] = pd.to_datetime(train_df_clean['dropoff_datetime'])



delta_t = np.log((train_df_clean['dropoff_datetime']-train_df_clean['pickup_datetime']).dt.total_seconds())

print("Number of wrong trip durations: %i" %train_df_clean[np.round(delta_t,5)!=np.round(train_df_clean['trip_duration'],5)].shape[0])
X_train = train_df_clean.drop(['id','dropoff_datetime','trip_duration'], axis=1)

Y_train = train_df_clean['trip_duration'].copy()



X_test = test_df.drop(['id'], axis=1)

X_test_id = test_df['id'].copy()





cat_att = ['store_and_fwd_flag']

date_att = ['pickup_datetime']

coord_att = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
def FlagEncoder(df):

    new_df = df.copy()

    new_df['store_and_fwd_flag'] = (new_df['store_and_fwd_flag']=='Y')*1

    return new_df
holidays = (pd.to_datetime('2016 '+ holidays_df['Date']))
def DateAttributes(df):

    new_df = df.copy()

    new_df['pickup_datetime'] = pd.to_datetime(new_df['pickup_datetime'])

    new_df['day'] = new_df['pickup_datetime'].dt.weekday

    new_df['weekend'] = 1*((new_df['day']>=5)|(new_df['pickup_datetime'].dt.date.isin(holidays.dt.date.values)))

    new_df['time'] = np.round(new_df['pickup_datetime'].dt.time.apply(lambda x: x.hour + x.minute/60.0),1)

    return new_df
def distance(coords): #  L1 and L2 distances (in arbitrary units)

    units = np.array([np.cos(np.radians(NYC[1])),1]) # multiply by 111.2 to get km 

    picks = np.split(coords.transpose(),2)[0].transpose()*units

    drops = np.split(coords.transpose(),2)[1].transpose()*units

    x1 = np.dot(picks,fifth_ave*units)

    y1 = np.dot(picks,ort_fifth_ave*units)    

    x2 = np.dot(drops,fifth_ave*units)

    y2 = np.dot(drops,ort_fifth_ave*units)

    dist_L1 = abs(x1-x2) + abs(y1-y2)

    dist_L2 = np.sqrt((x1-x2)**2 + (y1-y2)**2)

    return [dist_L1,dist_L2]
def DistanceAttribute(df):

    new_df = df.copy()

    coords = new_df[coord_att].values

    new_df['dist_L1'] = distance(coords)[0]

    new_df['dist_L2'] = distance(coords)[1]

    return new_df
# Adding attributes

pipe_list = [FlagEncoder, DateAttributes, DistanceAttribute]



X_train_prepared = my_pipeline(X_train,pipe_list)

X_test_prepared = my_pipeline(X_test,pipe_list)
X=X_train_prepared



c1 = (X['pickup_longitude']>LeftBound[0])&(X['pickup_longitude']<RightBound[0])

c2 = (X['pickup_longitude']>LeftBound[0])&(X['dropoff_longitude']<RightBound[0])

c3 = ((X['pickup_longitude']-EastRiver[0])*ort_fifth_ave[0]+(X['pickup_latitude']-EastRiver[1])*ort_fifth_ave[1])>0

c4 = ((X['pickup_longitude']-HudsonRiver[0])*ort_fifth_ave[0]+(X['pickup_latitude']-HudsonRiver[1])*ort_fifth_ave[1])<0

c5 = ((X['dropoff_longitude']-EastRiver[0])*ort_fifth_ave[0]+(X['dropoff_latitude']-EastRiver[1])*ort_fifth_ave[1])>0

c6 = ((X['dropoff_longitude']-HudsonRiver[0])*ort_fifth_ave[0]+(X['dropoff_latitude']-HudsonRiver[1])*ort_fifth_ave[1])<0



Manhattan_df = X[c1&c2&c3&c4&c5&c6]

Y_Manhattan = Y_train.values[c1&c2&c3&c4&c5&c6]

print('Percentage of trips within Manhattan: %.2f' %(1.*Manhattan_df.shape[0]/X.shape[0])) 
JFK = np.array([-73.779148,40.653416])

LaGuardia = np.array([-73.873890,40.775341])
X=X_train_prepared



c1 = (X['pickup_longitude']>(JFK[0]-0.05))&(X['pickup_longitude']<(JFK[0]+0.05))

c2 = (X['pickup_latitude']>(JFK[1]-0.05))&(X['pickup_latitude']<(JFK[1]+0.05))

c3 = (X['dropoff_longitude']>(JFK[0]-0.05))&(X['dropoff_longitude']<(JFK[0]+0.05))

c4 = (X['dropoff_latitude']>(JFK[1]-0.05))&(X['dropoff_latitude']<(JFK[1]+0.05))



JFK_df = X[(c1&c2)|(c3&c4)]

Y_JFK = Y_train.values[(c1&c2)|(c3&c4)]

print('Percentage of trips to/from JFK: %.2f' %(1.*JFK_df.shape[0]/X.shape[0])) 
X=X_train_prepared



c1 = (X['pickup_longitude']>(LaGuardia[0]-0.02))&(X['pickup_longitude']<(LaGuardia[0]+0.02))

c2 = (X['pickup_latitude']>(LaGuardia[1]-0.02))&(X['pickup_latitude']<(LaGuardia[1]+0.02))

c3 = (X['dropoff_longitude']>(LaGuardia[0]-0.02))&(X['dropoff_longitude']<(LaGuardia[0]+0.02))

c4 = (X['dropoff_latitude']>(LaGuardia[1]-0.02))&(X['dropoff_latitude']<(LaGuardia[1]+0.02))



LaGuardia_df = X[(c1&c2)|(c3&c4)]

Y_LaGuardia = Y_train.values[(c1&c2)|(c3&c4)]

print('Percentage of trips to/from La Guardia: %.2f' %(1.*LaGuardia_df.shape[0]/X.shape[0])) 
# Dataframe consolidation

Manhattan_df = DistanceAttribute(Manhattan_df)

JFK_df = DistanceAttribute(JFK_df)

LaGuardia_df = DistanceAttribute(LaGuardia_df)
plt.figure(figsize=(8,8))



plt.scatter(x=X_train_prepared['dist_L2'].values,y=np.exp(Y_train).values,s=1,alpha=0.1)

plt.xlim([0,0.3])

plt.ylim([0, 6000])

#plt.axis('off')

plt.xlabel('dist_L2')

plt.ylabel('Trip duration (log)')



plt.show()
plt.figure(figsize=(8,8))



plt.scatter(x=X_train_prepared['dist_L2'].values,y=np.exp(Y_train),s=1,alpha=0.1)

plt.scatter(x=Manhattan_df['dist_L2'].values,y=np.exp(Y_Manhattan),s=1,alpha=0.1)

plt.scatter(x=JFK_df['dist_L2'].values,y=np.exp(Y_JFK),s=1,alpha=0.1)

plt.scatter(x=LaGuardia_df['dist_L2'].values,y=np.exp(Y_LaGuardia),s=1,alpha=0.1)



plt.xlim([0,0.3])

plt.ylim([0, 6000])

plt.xlabel('dist_L2')

plt.ylabel('Trip duration')



plt.show()
c1 = X_train_prepared['dist_L1']>0.0001

c2 = X_train_prepared['dist_L2']>0.0001



X_train = X_train[(c1&c2)]

Y_train = Y_train[c1&c2]
def RideScope(df):

    X = df.copy()

    

    # Manhatthan only

    c1 = (X['pickup_longitude']>LeftBound[0])&(X['pickup_longitude']<RightBound[0])

    c2 = (X['dropoff_longitude']>LeftBound[0])&(X['dropoff_longitude']<RightBound[0])

    c3 = ((X['pickup_longitude']-EastRiver[0])*ort_fifth_ave[0]+(X['pickup_latitude']-EastRiver[1])*ort_fifth_ave[1])>0

    c4 = ((X['pickup_longitude']-HudsonRiver[0])*ort_fifth_ave[0]+(X['pickup_latitude']-HudsonRiver[1])*ort_fifth_ave[1])<0

    c5 = ((X['dropoff_longitude']-EastRiver[0])*ort_fifth_ave[0]+(X['dropoff_latitude']-EastRiver[1])*ort_fifth_ave[1])>0

    c6 = ((X['dropoff_longitude']-HudsonRiver[0])*ort_fifth_ave[0]+(X['dropoff_latitude']-HudsonRiver[1])*ort_fifth_ave[1])<0



    X['M&M'] = (c1&c2&c3&c4&c5&c6)*1

    

    # JFK

    c1 = (X['pickup_longitude']>(JFK[0]-0.05))&(X['pickup_longitude']<(JFK[0]+0.05))

    c2 = (X['pickup_latitude']>(JFK[1]-0.05))&(X['pickup_latitude']<(JFK[1]+0.05))

    c3 = (X['dropoff_longitude']>(JFK[0]-0.05))&(X['dropoff_longitude']<(JFK[0]+0.05))

    c4 = (X['dropoff_latitude']>(JFK[1]-0.05))&(X['dropoff_latitude']<(JFK[1]+0.05))

    

    X['JFK'] = ((c1&c2)|(c3&c4))*1

    

    #LaGuardia

    c1 = (X['pickup_longitude']>(LaGuardia[0]-0.02))&(X['pickup_longitude']<(LaGuardia[0]+0.02))

    c2 = (X['pickup_latitude']>(LaGuardia[1]-0.02))&(X['pickup_latitude']<(LaGuardia[1]+0.02))

    c3 = (X['dropoff_longitude']>(LaGuardia[0]-0.02))&(X['dropoff_longitude']<(LaGuardia[0]+0.02))

    c4 = (X['dropoff_latitude']>(LaGuardia[1]-0.02))&(X['dropoff_latitude']<(LaGuardia[1]+0.02))



    X['LaG'] = ((c1&c2)|(c3&c4))*1

    

    return X
pipe_list = pipe_list + [RideScope]



X_train_prepared = my_pipeline(X_train,pipe_list)

X_test_prepared = my_pipeline(X_test,pipe_list)
plt.figure(figsize=(8,8))



sc = plt.scatter(JFK_df['dist_L2'].values,np.exp(Y_JFK),c=JFK_df['time'],s=1,cmap=plt.get_cmap('jet'),alpha=0.5)

plt.xlim([0,0.3])

plt.ylim([0, 6000])

cb = plt.colorbar(sc)

plt.xlabel('dist_L2')

plt.ylabel('Trip duration')

cb.set_label('Time of the day')



plt.show()
def Speeds(df,Y):

    new_df = df.copy()

    new_df['speed_L1'] = new_df['dist_L1']/np.exp(Y)

    new_df['speed_L2'] = new_df['dist_L2']/np.exp(Y)

    return new_df
speed_df = Speeds(X_train_prepared,Y_train)



weekday_speed = speed_df.groupby(['weekend','time']).agg(['mean','std'])[['speed_L1','speed_L2']].loc[0].reset_index()

weekend_speed = speed_df.groupby(['weekend','time']).agg(['mean','std'])[['speed_L1','speed_L2']].loc[1].reset_index()
plt.figure(figsize=(6,6))



plt.errorbar(weekday_speed['time'],weekday_speed['speed_L2']['mean'],yerr=0, label='weekday')

plt.errorbar(weekend_speed['time'],weekend_speed['speed_L2']['mean'],yerr=0, label='weekend')

plt.ylim([0, 0.000125])



#plt.errorbar(weekday_speed['time'],weekday_speed['speed_L2']['mean'],yerr=weekday_speed['speed_L2']['std'])

#plt.errorbar(weekend_speed['time'],weekend_speed['speed_L2']['mean'],yerr=weekend_speed['speed_L2']['std'])

#plt.ylim([0, 0.000125])

plt.ylim([0, 0.0001])

plt.xlabel('time')

plt.ylabel('speed_L2')

plt.legend()



plt.show()
speeds = speed_df.groupby(['weekend','time']).mean()[['speed_L1','speed_L2']].reset_index()
def SpeedAttribute(df):

    new_df = df.copy()

    new_df = pd.merge(new_df, speeds, how='left', on=['weekend','time'])

    return new_df
pipe_list = pipe_list + [SpeedAttribute]



X_train_prepared = my_pipeline(X_train,pipe_list)

X_test_prepared = my_pipeline(X_test,pipe_list)
X_train_prepared.columns
num_att = ['passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','dist_L1','dist_L2','speed_L1','speed_L2','time']

OneHot = ['weekend','M&M','JFK','LaG']
num_pipeline = Pipeline([

    ('selector', DataFrameSelector(num_att)),

    ('imputer', Imputer(strategy="median")),

    ('std_scaler', StandardScaler()),

])



OneHot_pipeline = Pipeline([

    ('selector', DataFrameSelector(OneHot)),

])
full_pipeline = FeatureUnion(transformer_list=[

           ("num_pipeline", num_pipeline),

           ("cat_pipeline", OneHot_pipeline),

])
X_train_scaled = full_pipeline.fit_transform(X_train_prepared)

X_test_scaled = full_pipeline.transform(X_test_prepared)
#tree_reg = DecisionTreeRegressor()



#scores = cross_val_score(tree_reg, X_train_scaled, Y_train, cv=3, scoring='neg_mean_squared_error')

#mean_cross_score(-scores)
rnd_reg = RandomForestRegressor()



scores = cross_val_score(rnd_reg, X_train_scaled, Y_train, cv=3, scoring='neg_mean_squared_error')

mean_cross_score(-scores)
rnd_reg = RandomForestRegressor()

rnd_reg.fit(X_train_scaled, Y_train)
Y_test_pred = rnd_reg.predict(X_test_scaled)



submission = pd.DataFrame({

        "id": X_test_id,

        "trip_duration": np.exp(Y_test_pred)

    })

submission.to_csv('submission2.csv', index=False)