import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.impute import SimpleImputer
# reading the train.csv 

data=pd.read_csv("../input/train.csv") 

#reading the test data

test=pd.read_csv("../input/test.csv")
data.head()
data.shape
data.describe()
#checking NA's 

data.isna().sum()
# removing rows where fare_amount is 0 as those rows cannot help us to predict the target(fare_amount)

data=data.loc[data.fare_amount>0,]
data['passenger_count'].value_counts()
#removing rows where passenger_count is zero

data=data[data["passenger_count"]>0]
#droping NA's from pickup_longitute,pickup_latitide,dropoff_latittue,dropff_longitute columns

data.dropna(subset = ['pickup_longitude', 'pickup_latitude', 'dropoff_latitude', 'dropoff_longitude'], inplace= True)
#converting pickup_datetime,dropoff_datetime into datetime

data["pickup_datetime"]=pd.to_datetime(data["pickup_datetime"])

data["dropoff_datetime"]=pd.to_datetime(data["dropoff_datetime"])



#adding  new variables called "duration",'hour','day','month','year' 

data['duration']=abs(data['pickup_datetime']-data['dropoff_datetime'])/np.timedelta64(1,'m')



data['hour'] = data.pickup_datetime.dt.hour

data['day'] = data.pickup_datetime.dt.day

data['month'] = data.pickup_datetime.dt.month

data['year'] = data.pickup_datetime.dt.year
#creating a list of unwanted features

cols_drop=['TID','pickup_datetime','dropoff_datetime','new_user','store_and_fwd_flag']
#droping the unwanted features

data.drop(cols_drop,axis=1,inplace=True)
#addind a new feature called "distance",which is extracted from latitudes and langitudes.

def distance(s_lat, s_lng, e_lat, e_lng):

    

    # approximate radius of earth in km

    R = 6373.0

    

    s_lat = s_lat*np.pi/180.0                      

    s_lng = np.deg2rad(s_lng)     

    e_lat = np.deg2rad(e_lat)                       

    e_lng = np.deg2rad(e_lng)  

    

    d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat)*np.cos(e_lat) * np.sin((e_lng - s_lng)/2)**2

    

    return 2 * R * np.arcsin(np.sqrt(d)) 

#adding distance column to the dataset 

data['distance'] = distance(data.pickup_latitude,data.pickup_longitude,data.dropoff_latitude,data.dropoff_longitude)
corr=data.corr()

plt.figure(figsize=(10,8))

sns.heatmap(corr,annot=True,)
sns.distplot(data.loc[data.fare_amount<200,'fare_amount'])

plt.xlabel("fare_amount")

plt.ylabel("Frequency")

plt.title("distribution of fare_amount")
sns.distplot(data.loc[data.distance<200,'distance'])

plt.title("distrtibution of distance travelled")

plt.ylabel("Frequency")
sns.boxplot(x='passenger_count',y='fare_amount',data=data)

plt.ylim(0,40)

plt.title('passenger count vs fare')
plt.scatter(x='distance',y='fare_amount',data=data)

plt.ylim(0,70)

plt.xlim(0,80)

plt.xlabel('distance')

plt.ylabel('fare amount')

plt.title('distance vs fare amount')
plt.scatter(x='duration',y='fare_amount',data=data)

plt.ylim(0,70)

plt.xlim(0,200)

plt.xlabel('duration in minutes')

plt.ylabel('fare_amount')

plt.title('duratoin vs fare amount')
sns.boxplot(x='passenger_count',y='distance',data=data)

plt.ylim(0,10)

plt.title('passenger count vs fare')
#splitting the data into train,validation

y=data.fare_amount

X=data

X.drop('fare_amount',axis=1,inplace=True)
train_X,valid_X,train_y,valid_y=train_test_split(X,y,train_size=0.7,random_state=1)

print(train_X.shape)

print(valid_X.shape)

print(train_y.shape)

print(valid_y.shape)
#creating lists fro categorical and numeric columns

cat_cols = ["vendor_id","payment_type"]

num_cols = train_X.columns.difference(cat_cols)

num_cols
#converting the datatypes 

train_X[cat_cols]=train_X[cat_cols].apply(lambda x:x.astype("category"))

train_X[num_cols]=train_X[num_cols].apply(lambda x:x.astype("float"))



train_num_data = train_X.loc[:,num_cols]

train_cat_data = train_X.loc[:,cat_cols]
# numeric cols imputation

imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_num.fit(train_num_data)

train_num_data = pd.DataFrame(imp_num.transform(train_num_data),columns=num_cols)



# Categorical columns imputation

imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')



train_cat_data = pd.DataFrame(imp_cat.fit_transform(train_cat_data),columns=cat_cols)
#standardizing the train data

stand=StandardScaler()

stand.fit(train_num_data[train_num_data.columns])

train_num_data[train_num_data.columns]=stand.transform(train_num_data[train_num_data.columns])
train_X=pd.concat([train_num_data,train_cat_data],axis=1)
#creating dummies for categorical columns

train_X=pd.get_dummies(train_X,columns=cat_cols)
train_X.head()
# preprocessing on validation set

valid_X[cat_cols]=valid_X[cat_cols].apply(lambda x:x.astype("category"))

valid_X[num_cols]=valid_X[num_cols].apply(lambda x:x.astype("float"))

valid_num_data = valid_X.loc[:,num_cols]

valid_cat_data = valid_X.loc[:,cat_cols]



# numeric cols imputation



valid_num_data = pd.DataFrame(imp_num.transform(valid_num_data),columns=num_cols)



# Categorical columns imputation





valid_cat_data = pd.DataFrame(imp_cat.transform(valid_cat_data),columns=cat_cols)
#standarding the valdiation data

valid_num_data[valid_num_data.columns]=stand.transform(valid_num_data[valid_num_data.columns])
valid_X=pd.concat([valid_num_data,valid_cat_data],axis=1)



valid_X=pd.get_dummies(valid_X,columns=cat_cols)
#preprocessing on test data

#converting pickup_datetime,dropoff_datetime into datetime

test["pickup_datetime"]=pd.to_datetime(test["pickup_datetime"])

test["dropoff_datetime"]=pd.to_datetime(test["dropoff_datetime"])



#adding a new variable called "duration" 

test['duration']=abs(test['pickup_datetime']-test['dropoff_datetime'])/np.timedelta64(1,'m')



test['hour'] = test.pickup_datetime.dt.hour

test['day'] = test.pickup_datetime.dt.day

test['month'] = test.pickup_datetime.dt.month

test['year'] = test.pickup_datetime.dt.year



#droping unwanted columns

cols_drop=['TID','pickup_datetime','dropoff_datetime','new_user','store_and_fwd_flag']

test.drop(cols_drop,axis=1,inplace=True)



def distance(s_lat, s_lng, e_lat, e_lng):

    

    # approximate radius of earth in km

    R = 6373.0

    

    s_lat = s_lat*np.pi/180.0                      

    s_lng = np.deg2rad(s_lng)     

    e_lat = np.deg2rad(e_lat)                       

    e_lng = np.deg2rad(e_lng)  

    

    d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat)*np.cos(e_lat) * np.sin((e_lng - s_lng)/2)**2

    

    return 2 * R * np.arcsin(np.sqrt(d)) 

test['distance'] = distance(test.pickup_latitude,test.pickup_longitude,test.dropoff_latitude,test.dropoff_longitude)
# preprocessing on test set

test[cat_cols]=test[cat_cols].apply(lambda x:x.astype("category"))

test[num_cols]=test[num_cols].apply(lambda x:x.astype("float"))
test_num_data=test.loc[:,num_cols]

test_cat_data=test.loc[:,cat_cols]

# numeric cols imputation



test_num_data = pd.DataFrame(imp_num.transform(test_num_data),columns=num_cols)



# Categorical columns imputation





test_cat_data = pd.DataFrame(imp_cat.transform(test_cat_data),columns=cat_cols)
#standarding the test data

test_num_data[test_num_data.columns]=stand.transform(test_num_data[test_num_data.columns])
test=pd.concat([test_num_data,test_cat_data],axis=1)



test=pd.get_dummies(test,columns=cat_cols)
# MODEL1 LINEAR REGRESSION

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,mean_absolute_error

lir=LinearRegression()

lir.fit(train_X,train_y)



train_preds1=lir.predict(train_X)

valid_preds1=lir.predict(valid_X)
# MODEL1 PREDICTIONS

print("mean_absolute_error on train data:",mean_absolute_error(train_y,train_preds1))

print("mean_absolute_error on validation data:",mean_absolute_error(valid_y,valid_preds1))
# MODEL2 DECISION TREE REGRESSOR 

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import RandomizedSearchCV



dtc=DecisionTreeRegressor()

dtc.fit(train_X,train_y)



train_preds2=dtc.predict(train_X)

valid_preds2=dtc.predict(valid_X)



print("mean_absolute_error on train data:",mean_absolute_error(train_y,train_preds2))

print("mean_absolute_error on validation data:",mean_absolute_error(valid_y,valid_preds2))
# MODEL3 KNN REGRESSOR

from sklearn.neighbors import KNeighborsRegressor

knn=KNeighborsRegressor(n_neighbors=3,algorithm="brute",weights="distance")

knn.fit(train_X,train_y)



train_preds3=knn.predict(train_X)

valid_preds3=knn.predict(valid_X)

print("mean_absolute_error on train data:",mean_absolute_error(train_y,train_preds3))

print("mean_absolute_error on validation data:",mean_absolute_error(valid_y,valid_preds3))
# MODEL 4 XGBOOST

from xgboost import XGBRegressor

xgb = XGBRegressor()

xgb.fit(train_X,train_y)



train_pred4=xgb.predict(train_X)

valid_pred4=xgb.predict(valid_X)



print("mean_absolute_error on train data:",mean_absolute_error(train_y,train_pred4))

print("mean_absolute_error on validation data:",mean_absolute_error(valid_y,valid_pred4))
#hyperparameters tuning

Xgb=XGBRegressor()

n_estimaters=[50,100,150,200]

max_depth=[2,3,5,7]

learnin_rate=[0.05,0.1,0.15,0.20]

min_child_wgt=[1,2,3,4]







hyperparameter={

    "n_estimaters":n_estimaters,

    "max_depth":max_depth,

    "learnin_rate":learnin_rate,

    "min_child_wgt":min_child_wgt,



}



random_cv2=RandomizedSearchCV(estimator=Xgb,param_distributions=hyperparameter,cv=5,n_jobs=-1)
random_cv2.fit(train_X,train_y)
random_cv2.best_estimator_
#XGBOOST WITH BEST ESTIMATER

Xgb2=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0,

             importance_type='gain', learnin_rate=0.05, learning_rate=0.1,

             max_delta_step=0, max_depth=7, min_child_weight=1, min_child_wgt=2,

             missing=None, n_estimaters=50, n_estimators=100, n_jobs=1,

             nthread=None, objective='reg:linear', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,

             subsample=1, verbosity=1)



Xgb2.fit(train_X,train_y)





train_pred6=Xgb2.predict(train_X)

valid_pred6=Xgb2.predict(valid_X)
print("mean_absolute_error on train data:",mean_absolute_error(train_y,train_pred6))

print("mean_absolute_error on validation data:",mean_absolute_error(valid_y,valid_pred6))
#prediction on test data with Xgboost best estimator

test_predictions=Xgb2.predict(test)

test_prediction=pd.DataFrame(test_predictions)
test_prediction