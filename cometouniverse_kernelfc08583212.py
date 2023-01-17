import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#train=pd.read_csv("/kaggle/input/train.csv",nrows=1000000)

train=pd.read_csv("/kaggle/input/train.csv",index_col='pickup_datetime',parse_dates=True,nrows=1000000)
test=pd.read_csv("/kaggle/input/test.csv",index_col='pickup_datetime',parse_dates=True)

#test=pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')
train.info()
train.head(3)
#train['pickup_datetime'] =  pd.to_datetime(train['pickup_datetime'])
test.head(3)
test.info()
#test['pickup_datetime'] =  pd.to_datetime(test['pickup_datetime'])
pd.options.display.float_format = '{:.6f}'.format

train.describe()
test_desc=test.describe()

test_desc
lon_min=min(test_desc['pickup_longitude']['min'],test_desc['dropoff_longitude']['min']);

lon_max=max(test_desc['pickup_longitude']['max'],test_desc['dropoff_longitude']['max']);

lat_min=min(test_desc['pickup_latitude']['min'],test_desc['dropoff_latitude']['min']);

lat_max=min(test_desc['pickup_latitude']['max'],test_desc['dropoff_latitude']['max']);

lon_min,lon_max,lat_min,lat_max
train[(train['pickup_longitude']>lon_max) | (train['pickup_longitude'] <lon_min) | (train['dropoff_longitude'] > lon_max) | (train['dropoff_latitude'] < lat_min)]
train=train[~((train['pickup_longitude']>lon_max) | (train['pickup_longitude'] <lon_min) | (train['dropoff_longitude'] > lon_max) | (train['dropoff_longitude'] < lon_min)

             | (train['pickup_latitude']>lat_max) | (train['pickup_latitude'] <lat_min) | (train['dropoff_latitude'] > lat_max) | (train['dropoff_latitude'] < lat_min))]

train.head(3)
train=train.dropna()
sns.boxplot(x='passenger_count',data=train)
train.passenger_count.value_counts()
train[train['passenger_count']==0]
train=train[(train['passenger_count']>=1) & (train['passenger_count']<=6)]

train
"""Referenced from (https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points (by Michael Dunn))"""



from math import radians, cos, sin, asin, sqrt



def distance(lat1,lon1,lat2,lon2):

    # convert decimal degrees to radians 

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])



    # haversine formula 

    dlon = lon2 - lon1 

    dlat = lat2 - lat1 

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2

    c = 2 * asin(sqrt(a)) 

    r = 6371 # Radius of earth in kilometers. Use 3956 for miles

    return c * r
train['dist'] =  np.vectorize(distance)(train['pickup_latitude'],train['pickup_longitude'],train['dropoff_latitude'],train['dropoff_longitude'])

#from haversine import haversine, Unit #(lat, lon)

#train['dist']=train.apply(lambda a:haversine((a.pickup_latitude,a.pickup_longitude),(a.dropoff_latitude,a.dropoff_longitude), unit='mi'),axis=1)
train.head(3)
f, ax = plt.subplots(figsize=(16, 6))

sns.distplot(train.dist)
sns.heatmap(train[['fare_amount','dist']].corr(),annot=True)
train_a=train[train.fare_amount == 0]

train_b=train[train.fare_amount < 0]

print(train_a.shape[0] ,',',train_b.shape[0])
train_a.dist.hist()
train_b.dist.hist()
train[train.fare_amount < 0] = 0

train[train.fare_amount < 0].shape[0]
train_f=train.fare_amount

#train_f.plot()
train_f.resample('Y').mean().plot()
train_f.groupby(train_f.index.year).describe()
train['year']=train.index.year

train.head(1)
import scipy.stats as sp

fig,ax=plt.subplots(2,4,figsize=(10,10))

for j in range(2):

    for i in range(4):

        if j==1 and i==3 :

            break

        year=2009+i+(j*4)

        sp.probplot(train[train['year']==year].fare_amount, plot=ax[j,i])

'''Referenced from ( https://stackoverflow.com/questions/51632900/pandas-apply-kruskal-wallis-to-numeric-columns )'''

sp.mstats.kruskalwallis(*[group["fare_amount"].values for name, group in train.groupby("year")])
monthly=train_f.resample('M').mean()

monthly.rolling(3,center=True).sum().plot(style=['--'])

plt.grid(True)
train['month']=train.index.month

train.head(1)
fig,ax=plt.subplots(3,4,figsize=(10,10))

for j in range(3):

    for i in range(4):

        month=1+i+(j*4)

        sp.probplot(train[train['month']==month].fare_amount, plot=ax[j,i])
'''Referenced from ( https://stackoverflow.com/questions/51632900/pandas-apply-kruskal-wallis-to-numeric-columns )'''

sp.mstats.kruskalwallis(*[group["fare_amount"].values for name, group in train.groupby("month")])
by_time=train_f.groupby(train_f.index.hour).mean()

by_time.plot(style=['--'])
train['evening']=train.index.hour

train['evening'].head(10)
train['evening']=np.vectorize(lambda x: 1 if (x<=6 or x>=20 ) else 0)(train.evening)
train['evening'].value_counts()
fig,ax=plt.subplots(2,figsize=(10,10))

for j in range(2):

    sp.probplot(train[train['evening']==j].fare_amount, plot=ax[j])
sp.mstats.kruskalwallis(*[group["fare_amount"].values for name, group in train.groupby("evening")])
by_weekday= train_f.groupby(train_f.index.dayofweek).mean()

by_weekday.index = ['Mon','Tues','Wed','Thurs','Fri','Sat','Sun']

by_weekday.plot(style=['--'])
by_date=train_f.groupby(train_f.index.day).mean()

by_date.plot(style=['--'])

plt.grid(True)
train['dayofweek'] = train.index.dayofweek

train.dayofweek = train.dayofweek.map({0:'Mon',1:'Tues',2:'Wed',3:'Thurs',4:'Fri',5:'Sat',6:'Sun'})

train.head(5)
fig,ax=plt.subplots(2,4,figsize=(10,10));dayofweek=['Mon','Tues','Wed','Thurs','Fri','Sat','Sun']

count=0

for j in range(2):

    for i in range(4):

        if count <=6:

            sp.probplot(train[train['dayofweek']==dayofweek[count]].fare_amount, plot=ax[j,i])

            count+=1
sp.mstats.kruskalwallis(*[group["fare_amount"].values for name, group in train.groupby("dayofweek")])
sns.distplot(train[train.fare_amount <= 12.5].fare_amount)
train.groupby(['passenger_count']).fare_amount.mean()
LgAirLog=-73.874044 ; LgAirLat=40.776967 ; JfkAirLog=-73.783051 ; JfkAirLat = 40.648433
train['LgAirPickDist']=np.vectorize(distance)(train['pickup_latitude'],train['pickup_longitude'], LgAirLat , LgAirLog)

train['LgAirDropDist']=np.vectorize(distance)(train['dropoff_latitude'],train['dropoff_longitude'], LgAirLat , LgAirLog)

train['JfkAirPickDist']=np.vectorize(distance)(train['pickup_latitude'],train['pickup_longitude'], JfkAirLat , JfkAirLog)

train['JfkAirDropDist']=np.vectorize(distance)(train['dropoff_latitude'],train['dropoff_longitude'], JfkAirLat , JfkAirLog)
train.head(3)
train_a=train[(train.JfkAirPickDist <= 2) | (train.JfkAirDropDist <= 2)]
train_a[['dist','fare_amount']].describe()
sns.distplot(train_a.dist)
sns.distplot(train_a.fare_amount)
train_b=train[(train.LgAirDropDist <= 2) | (train.LgAirPickDist <= 2)]
sns.distplot(train_b.dist)
sns.distplot(train_b.fare_amount)
#Referenced from ( https://thenotes.tistory.com/entry/Ttest-in-python [NOTES] )

from scipy import stats



tTest_Result= stats.ttest_ind(train[train.JfkAirDropDist <= 2].fare_amount,train[train.JfkAirDropDist > 2].fare_amount, equal_var=False) 

print("The t-statistic and p-value assuming unequal variances is %.3f and %.3f." % tTest_Result)

tTest_Result= stats.ttest_ind(train[train.JfkAirPickDist <= 2].fare_amount,train[train.JfkAirPickDist > 2].fare_amount, equal_var=False) 

print("The t-statistic and p-value assuming unequal variances is %.3f and %.3f." % tTest_Result)

tTest_Result= stats.ttest_ind(train[train.LgAirDropDist <= 2].fare_amount,train[train.LgAirDropDist > 2].fare_amount, equal_var=False) 

print("The t-statistic and p-value assuming unequal variances is %.3f and %.3f." % tTest_Result)

tTest_Result= stats.ttest_ind(train[train.LgAirPickDist <= 2].fare_amount,train[train.LgAirPickDist > 2].fare_amount, equal_var=False) 

print("The t-statistic and p-value assuming unequal variances is %.3f and %.3f." % tTest_Result)

train['LgAirPickDist']=np.vectorize(lambda x: 1 if x<=2 else 0)(train['LgAirPickDist'])

train['LgAirDropDist']=np.vectorize(lambda x: 1 if x<=2 else 0)(train['LgAirDropDist'])

train['JfkAirPickDist']=np.vectorize(lambda x: 1 if x<=2 else 0)(train['JfkAirPickDist'])

train['JfkAirDropDist']=np.vectorize(lambda x: 1 if x<=2 else 0)(train['JfkAirDropDist'])
r1={'south':40.557246 ,'north':40.64994 , 'west':-74.213979 , 'east':-74.053616 } ;

r2={'south':40.496008 ,'north':r1['south'] , 'west': -74.255730 , 'east': -74.101707 };

r3={'south':40.701027 ,'north':40.748349 , 'west': -74.019548 , 'east': -73.967354 };

r4={'south':r3['north'] ,'north':40.766704 , 'west': -74.010367 , 'east': -73.929445 };

r5={'south':r4['north'] ,'north':40.796729 , 'west': -73.997547 , 'east': -73.929445 };

r6={'south':r5['north'] ,'north':40.911176 , 'west': -73.976801 , 'east': -73.929445 };

r7={'south':r5['south'] ,'north':40.915690 , 'west': r6['east'] , 'east': -73.781091 };

r8={'south':40.739446 ,'north':r4['south'] , 'west': r3['east'] , 'east': -73.942672 };

r9={'south': 40.570376 ,'north':r3['south'] , 'west':  -74.041803 , 'east': -73.856216 };

r10={'south': r9['north'] ,'north':r8['south'] , 'west':  r3['east'] , 'east': r8['east'] };

r11={'south': r4['south'] ,'north':r7['south'] , 'west':  r4['east'] , 'east':-73.764818 };

r12={'south': r10['south'] ,'north':r11['south'] , 'west':  r10['east'] , 'east':-73.700318 };

r13={'south': r9['south'] ,'north': r12['south'] , 'west':  r9['east'] , 'east':-73.725710 };

r14={'south': 40.543202 ,'north': r13['south'] , 'west':  -73.940431 , 'east':-73.844121 }; 

areaList=[r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14] 
West = -74.257159 ; East = -73.699215 ; North = 40.915568 ; South = 40.495992 ;

train[((train.pickup_longitude < West) | (train.pickup_longitude > East)) & ((train.pickup_latitude < South) | (train.pickup_latitude > North))]
def CheckArea(lat,lon):

    for num,area in enumerate(areaList):

        if ( (area['west'] <= lon )& (area['east'] > lon)  & (area['south'] <= lat) & (area['north'] > lat) ):

            return str(num+1)

    return 0



# In -> In : 0 ,  In -> Out : 1 , Out -> In : 2 , Out -> Out : 3

def IsBorderChange(pickArea,dropArea):

    if pickArea == '0':

        if dropArea =='0':

            return '3'

        else:

            return '2'

    else:

        if dropArea =='0':

            return '1'

        else:

            return '0'
train['pickupArea'] =  np.vectorize(CheckArea)(train['pickup_latitude'],train['pickup_longitude'])

train['dropoffArea'] =  np.vectorize(CheckArea)(train['dropoff_latitude'],train['dropoff_longitude'])

train['BorderChange'] = np.vectorize(IsBorderChange)(train['pickupArea'],train['dropoffArea'])

train.head(5)
sns.countplot(x="pickupArea",data=train)

#train['pickupArea'].value_counts()
sns.countplot(x="dropoffArea",data=train)

#train['dropoffArea'].value_counts()
'''# In -> In : 0 ,  In -> Out : 1 , Out -> In : 2 , Out -> Out : 3'''

sns.countplot(x="BorderChange",data=train)

train.BorderChange.value_counts()
sns.boxplot(x="BorderChange",y='fare_amount',data=train)
fig,ax=plt.subplots(4)

fig.subplots_adjust(hspace=1.2,wspace=0.4)

for num in range(4):

    obj=str(num)                  

    ax[num].hist(train[train['BorderChange']==obj].fare_amount)

    ax[num].set_title("BorderChange:{}".format(num))



plt.show()
train[train['BorderChange']=='1'].fare_amount.plot(kind='kde')
fig = plt.figure(figsize=(30,12))

sns.heatmap(train.corr(),annot=True)
train.head(3)
train=train.drop(['passenger_count','month','evening','pickupArea','dropoffArea','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','key'],axis=1)
train.head(3)
test['dist'] =  np.vectorize(distance)(test['pickup_latitude'],test['pickup_longitude'],test['dropoff_latitude'],test['dropoff_longitude'])

test['year']=test.index.year

test['dayofweek'] = test.index.dayofweek;test.dayofweek = test.dayofweek.map({0:'Mon',1:'Tues',2:'Wed',3:'Thurs',4:'Fri',5:'Sat',6:'Sun'})



test['LgAirPickDist']=np.vectorize(distance)(test['pickup_latitude'],test['pickup_longitude'], LgAirLat , LgAirLog)

test['LgAirDropDist']=np.vectorize(distance)(test['dropoff_latitude'],test['dropoff_longitude'], LgAirLat , LgAirLog)

test['JfkAirPickDist']=np.vectorize(distance)(test['pickup_latitude'],test['pickup_longitude'], JfkAirLat , JfkAirLog)

test['JfkAirDropDist']=np.vectorize(distance)(test['dropoff_latitude'],test['dropoff_longitude'], JfkAirLat , JfkAirLog)



test['LgAirPickDist']=np.vectorize(lambda x: 1 if x<=2 else 0)(test['LgAirPickDist'])

test['LgAirDropDist']=np.vectorize(lambda x: 1 if x<=2 else 0)(test['LgAirDropDist'])

test['JfkAirPickDist']=np.vectorize(lambda x: 1 if x<=2 else 0)(test['JfkAirPickDist'])

test['JfkAirDropDist']=np.vectorize(lambda x: 1 if x<=2 else 0)(test['JfkAirDropDist'])



test['pickupArea'] =  np.vectorize(CheckArea)(test['pickup_latitude'],test['pickup_longitude'])

test['dropoffArea'] =  np.vectorize(CheckArea)(test['dropoff_latitude'],test['dropoff_longitude'])

test['BorderChange'] = np.vectorize(IsBorderChange)(test['pickupArea'],test['dropoffArea'])
test.head(3)
test=test.drop(['passenger_count','pickupArea','dropoffArea','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1)
test.head(1)
new_train=pd.get_dummies(train[['dist','year','dayofweek','LgAirPickDist','LgAirDropDist','JfkAirPickDist','JfkAirDropDist','BorderChange']],

                         columns=['year','dayofweek','LgAirPickDist','LgAirDropDist','JfkAirPickDist','JfkAirDropDist','BorderChange'])

new_train.head(3)
test=pd.get_dummies(test[['dist','year','dayofweek','LgAirPickDist','LgAirDropDist','JfkAirPickDist','JfkAirDropDist','BorderChange','key']],

                         columns=['year','dayofweek','LgAirPickDist','LgAirDropDist','JfkAirPickDist','JfkAirDropDist','BorderChange'])
test.head(1)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



train_x,test_x,train_y,test_y=train_test_split(new_train,train[['fare_amount']],random_state=10)
print(f'{train_x.shape[0]},{test_x.shape[0]},{train_y.shape[0]},{test_y.shape[0]} ') 
train_y.head(3)
LinearReg=LinearRegression().fit(train_x,train_y)
print(f' coef: {LinearReg.coef_}, Intercept : {LinearReg.intercept_ } , score_train : {LinearReg.score(train_x,train_y)} ,  score_test : {LinearReg.score(test_x,test_y)}') 
from sklearn.tree import DecisionTreeRegressor

tree=DecisionTreeRegressor().fit(train_x,train_y)

print(f' score_train : {tree.score(train_x,train_y)} ,  score_test : {tree.score(test_x,test_y)}') 
from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(max_depth=10,n_jobs=-1).fit(train_x,train_y)

print(f' score_train : {regr.score(train_x,train_y)} ,  score_test : {regr.score(test_x,test_y)}') 
from sklearn.ensemble import GradientBoostingRegressor

regr2 = GradientBoostingRegressor().fit(train_x,train_y)

print(f' score_train : {regr2.score(train_x,train_y)} ,  score_test : {regr2.score(test_x,test_y)}') 
#Referenced From https://github.com/rickiepark/introduction_to_ml_with_python/blob/master/02-supervised-learning.ipynb

def plot_feature_importances(model,n_features):

    plt.figure(figsize=(20,20))

    plt.barh(np.arange(n_features), model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features), train_x.columns)

    plt.xlabel("feature_importances")

    plt.ylabel("features")

    plt.ylim(-1, n_features)

plot_feature_importances(regr2,train_x.shape[1])
#Referenced From ( https://github.com/rickiepark/introduction_to_ml_with_python/blob/master/05-model-evaluation-and-improvement.ipynb )



max_depth=[4,8,12];n_estimators=[5,10,15];max_features=["auto","sqrt","log2"]



best_score = 0

for depth in max_depth:

    for estimators in n_estimators:

        for featrues in max_features:

            regr = RandomForestRegressor(max_depth=depth,n_jobs=-1,n_estimators=estimators,max_features=featrues).fit(train_x,train_y)

            score = regr.score(test_x,test_y)

            if score > best_score:

                best_score = score

                best_parameters = {'max_depth': depth, 'n_estimators': estimators , 'max_features':featrues}

                        

print(f' score_test : {best_score} ,  score_test : {best_parameters}')             
#Referenced from ( https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration )

test_key=test.key;test=test.drop(['key'],axis=1)

estimator = RandomForestRegressor(max_depth= 8, n_estimators= 10, max_features= 'auto').fit(train_x,train_y)

y_pred = estimator.predict(test)



submission = pd.DataFrame(

    {'key': test_key, 'fare_amount': y_pred},

    columns = ['key', 'fare_amount'])

submission.to_csv('submission.csv', index = False)