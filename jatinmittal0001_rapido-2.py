import warnings

warnings.filterwarnings('ignore')

# To  collect garbage (delete files)

import gc

# To save dataset as pcikle file for future use

import pickle



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# for basic math operations like sqrt

import math

from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor 

from math import sin, cos, sqrt, atan2, radians

from sklearn.cluster import KMeans





from sklearn.svm import SVR

import matplotlib.pyplot as plt

import seaborn as sns

from xgboost import plot_importance

from sklearn.linear_model import LinearRegression

def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)



import os

print(os.listdir("../input/rapido"))
data = pickle.load(open("../input/rapido/rapido_v_save","rb"))

print(data.shape)

data.head(3)
def reduce_size(df):

    float_cols = [c for c in df if df[c].dtype == "float64"]

    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]

    df[float_cols] = df[float_cols].astype(np.float32)

    df[int_cols] = df[int_cols].astype(np.int16)

    return df



data = reduce_size(data)
K_clusters = range(1,6) 

kmeans = [KMeans(n_clusters=i) for i in K_clusters] 

Y_axis = data[['pick_lat']] 

X_axis = data[['pick_lng']] 

score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
plt.plot(K_clusters, score)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve')

plt.show()
kmeans = KMeans(n_clusters = 3, init ='k-means++')

kmeans.fit(data[data.columns[1:2]]) # Compute k-means clustering.



centers = kmeans.cluster_centers_ # Coordinates of cluster centers.

labels = kmeans.predict(data[data.columns[1:2]]) # Labels of each point

data['cluster_label'] = kmeans.fit_predict(data[data.columns[1:2]])

data.head(3)
day_time = {0:'Night', 1 : 'Night', 2 : 'Night', 3 : 'Night', 4 : 'Night', 5 : 'Night', 6 : 'Morning', 	7 : 'Morning', 	

                  8 : 'Morning', 	9 : 'Morning', 10 : 'Morning', 11 : 'Morning',12 : 'Afternoon', 13 : 'Afternoon', 	

                  14 : 'Afternoon', 15 : 'Afternoon', 16 : 'Afternoon', 17 :'Evening', 	18 : 'Evening', 19 : 'Evening', 

                  20 : 'Evening', 	21 : 'Evening',22 : 'Night', 23 : 'Night'}



data['time_type'] = data['hour'].map(day_time)
data.plot.scatter(x = 'pick_lat', y = 'pick_lng', c=labels, s=50, cmap='viridis')
#creating one hot encoding vectors for clusters, this will be used later to create features

data = pd.get_dummies(data,columns=['cluster_label'])

data = pd.get_dummies(data,columns=['time_type'])

print(data.shape)

data.head(2)
data['date'] = pd.to_numeric(data.date)  #converting to numeric

df = data[['date', 'hour','new_month']].copy()



#Dropping duplicates and keeping first occurence only

df.drop_duplicates(keep = 'first', inplace = True)   # these are the main time based features on which forecasting wil be done
print("Number of hours and dates data points available for April 2019: ",len(df[df.new_month==16]))
df = df[df.new_month!=16]  #Removing April data

df.shape
#Adding all combinations of hour and date  on neew month

for i in range(24):

    

    hour = i  #minimum hour is 0 in data

    

    new_month = 16

    

    for  j in range(30):

        date = (j+1)%30

        if date == 0:

            date = 30

        df = df.append({'date' : date , 'hour' : hour, 'new_month': new_month} , ignore_index=True)





df = df.reset_index()

df.drop(['index'],axis=1,inplace=True)

df = df.sort_values(['new_month', 'date', 'hour'], ascending=[True, True,True])

print(df.shape)

df.tail(3)
ab = data.groupby(['new_month']).distance.count()

ab = ab.reset_index()

ab.head()
from scipy.optimize import curve_fit



def test(x, a, b): 

    return (a * x*x*x*x + b)   #fitting curve

  

 

x = ab.new_month

y = ab.distance



param, param_cov = curve_fit(test, x, y) 



ans = (param[0]*x*x*x*x + param[1])

  

plt.plot(x, y, 'o', color ='red', label ="actual number of rides") 

plt.plot(x, ans, '--', color ='blue', label ="curve fit") 

plt.legend() 

plt.show() 
#group 1: mean distance, mean price, mean ride_minutes at date x hour x new_month level

group1 = data.groupby(['date', 'hour','new_month']).agg({'distance':['mean'], 'ride_price':['mean'], 'ride_minutes':['mean']})

group1.columns = ['mean_dhm_dist','mean_dhm_price','mean_dhm_minutes']

group1.reset_index(inplace=True)



#group 2: mean distance, mean price, mean ride_minutes at date  x new_month level

group2 = data.groupby(['date', 'new_month']).agg({'distance':['mean'], 'ride_price':['mean'], 'ride_minutes':['mean']})

group2.columns = ['mean_dm_dist','mean_dm_price','mean_dm_minutes']

group2.reset_index(inplace=True)





#group 3: mean distance, mean price, mean ride_minutes at new_month level

group3 = data.groupby(['new_month']).agg({'distance':['mean'], 'ride_price':['mean'], 'ride_minutes':['mean']})

group3.columns = ['mean_m_dist','mean_m_price','mean_m_minutes']

group3.reset_index(inplace=True)





#group 4: min distance, min price, min ride_minutes at date x hour x new_month level

group4 = data.groupby(['date', 'hour','new_month']).agg({'distance':['min'], 'ride_price':['min'], 'ride_minutes':['min']})

group4.columns = ['min_dhm_dist','min_dhm_price','min_dhm_minutes']

group4.reset_index(inplace=True)



#group 5: min distance, min price, min ride_minutes at date  x new_month level

group5 = data.groupby(['date', 'new_month']).agg({'distance':['min'], 'ride_price':['min'], 'ride_minutes':['min']})

group5.columns = ['min_dm_dist','min_dm_price','min_dm_minutes']

group5.reset_index(inplace=True)





#group 6: min distance, min price, min ride_minutes at new_month level

group6 = data.groupby(['new_month']).agg({'distance':['min'], 'ride_price':['min'], 'ride_minutes':['min']})

group6.columns = ['min_m_dist','min_m_price','min_m_minutes']

group6.reset_index(inplace=True)



#group 7: max distance, max price, max ride_minutes at date x hour x new_month level

group7 = data.groupby(['date', 'hour','new_month']).agg({'distance':['max'], 'ride_price':['max'], 'ride_minutes':['max']})

group7.columns = ['max_dhm_dist','max_dhm_price','max_dhm_ride_minutes']

group7.reset_index(inplace=True)



#group 8: max distance, max price, max ride_minutes at date  x new_month level

group8 = data.groupby(['date', 'new_month']).agg({'distance':['max'], 'ride_price':['max'], 'ride_minutes':['max']})

group8.columns = ['max_dm_dist','max_dm_price','max_dm_ride_minutes']

group8.reset_index(inplace=True)





#group 9: max distance, max price, max ride_minutes at new_month level

group9 = data.groupby(['new_month']).agg({'distance':['max'], 'ride_price':['max'], 'ride_minutes':['max']})

group9.columns = ['max_m_dist','max_m_price','max_m_ride_minutes']

group9.reset_index(inplace=True)



#group 10: cluster countat new_month level

group10 = data.groupby(['new_month']).agg({'cluster_label_0':['sum'], 'cluster_label_1':['sum'],'cluster_label_2':['sum']})

group10.columns = ['count_m_cluster_0','count_m_cluster_1','count_m_cluster_2']

group10.reset_index(inplace=True)



#group 11: cluster count at date x hour x new_month level

group11 = data.groupby(['date', 'hour','new_month']).agg({'cluster_label_0':['sum'], 'cluster_label_1':['sum'],'cluster_label_2':['sum']})

group11.columns = ['count_dhm_cluster_0','count_dhm_cluster_1','count_dhm_cluster_2']

group11.reset_index(inplace=True)



#group 12: cluster count at date  x new_month level

group12 = data.groupby(['date', 'new_month']).agg({'cluster_label_0':['sum'], 'cluster_label_1':['sum'],'cluster_label_2':['sum']})

group12.columns = ['count_dm_cluster_0','count_dm_cluster_1','count_dm_cluster_2']

group12.reset_index(inplace=True)







#group 13: ride count at new_month level

group13 = data.groupby(['new_month']).agg({'cluster_label_0':['count']})

group13.columns = ['count_m_ride']

group13.reset_index(inplace=True)



#group 14: ride count at date x hour x new_month level

group14 = data.groupby(['date', 'hour','new_month']).agg({'cluster_label_0':['count']})

group14.columns = ['count_dhm_ride']

group14.reset_index(inplace=True)



#group 15: ride count at date  x new_month level

group15 = data.groupby(['date', 'new_month']).agg({'cluster_label_0':['count']})

group15.columns = ['count_dm_ride']

group15.reset_index(inplace=True)





#group 16: cluster countat new_month level

group16 = data.groupby(['new_month']).agg({'time_type_Afternoon':['sum'], 

                                           'time_type_Evening':['sum'],

                                           'time_type_Morning':['sum'],

                                           'time_type_Night':['sum']})



group16.columns = ['count_m_time_type_0','count_m_time_type_1','count_m_time_type_2','count_m_time_type_3']

group16.reset_index(inplace=True)



#group 17: cluster count at date x hour x new_month level

group17 = data.groupby(['date', 'hour','new_month']).agg({'time_type_Afternoon':['sum'], 

                                                          'time_type_Evening':['sum'],

                                                          'time_type_Morning':['sum'],

                                                          'time_type_Night':['sum']})



group17.columns = ['count_dhm_time_type_0','count_dhm_time_type_1','count_dhm_time_type_2','count_dhm_time_type_3']

group17.reset_index(inplace=True)



#group 18: cluster count at date  x new_month level

group18 = data.groupby(['date', 'new_month']).agg({'time_type_Afternoon':['sum'],

                                                   'time_type_Evening':['sum'],

                                                   'time_type_Morning':['sum'],

                                                   'time_type_Night':['sum']})



group18.columns = ['count_dm_time_type_0','count_dm_time_type_1','count_dm_time_type_2','count_dm_time_type_3']

group18.reset_index(inplace=True)
#merge all



df = pd.merge(df, group1, on = ['date', 'hour','new_month'], how = 'left')

df = pd.merge(df, group2, on = ['date', 'new_month'], how = 'left')

df = pd.merge(df, group3, on = ['new_month'], how = 'left')



df = pd.merge(df, group4, on = ['date', 'hour','new_month'], how = 'left')

df = pd.merge(df, group5, on = ['date', 'new_month'], how = 'left')

df = pd.merge(df, group6, on = ['new_month'], how = 'left')



df = pd.merge(df, group7, on = ['date', 'hour','new_month'], how = 'left')

df = pd.merge(df, group8, on = ['date', 'new_month'], how = 'left')

df = pd.merge(df, group9, on = ['new_month'], how = 'left')



df = pd.merge(df, group10, on = ['new_month'], how = 'left')

df = pd.merge(df, group11, on = ['date', 'hour','new_month'], how = 'left')

df = pd.merge(df, group12, on = ['date', 'new_month'], how = 'left')



df = pd.merge(df, group13, on = ['new_month'], how = 'left')

df = pd.merge(df, group14, on = ['date', 'hour','new_month'], how = 'left')

df = pd.merge(df, group15, on = ['date', 'new_month'], how = 'left')



df = pd.merge(df, group16, on = ['new_month'], how = 'left')

df = pd.merge(df, group17, on = ['date', 'hour','new_month'], how = 'left')

df = pd.merge(df, group18, on = ['date', 'new_month'], how = 'left')



print(df.shape)

df.head(3)
#defining lag function. Lag will be taken at month level.



def lag_feature(df, lags, col):

    tmp = df[['date', 'hour', 'new_month', col]]



    for i in lags:

        shifted = tmp.copy()

        shifted.columns = ['date', 'hour', 'new_month',col+'_lag_'+str(i)]

        shifted['new_month'] += i

        #df.loc[i*, col+'_lag_'+str(i)] = shifted[col+'_lag_'+str(i)]

        df = pd.merge(df, shifted, on=['date', 'hour', 'new_month'], how='left')

        del shifted

    del tmp

    gc.collect();

    return df
for i in df.columns[3:]:

    df = lag_feature(df, [1], i)

    df.drop(i,axis=1,inplace=True)   #dropping original rolled up features asthey are not required.
#removing null values which came due to 1 month lag, this will come because for the starting month we dont have any lag

df = df[(df.new_month>=5)]

print(df.shape)

print("Missing values:")

print(df.isnull().sum().sum()>0)


group4 = df[~(df.mean_dhm_dist_lag_1.isnull())].groupby(['new_month']).mean()

group4.drop(['hour','date'],axis=1, inplace=True)

group4 = group4.reset_index()

group4.head()
#now after finding mean, replace it with null values in the data. Creating two datasets, one for non null and other for null



d1 = df[~(df.mean_dhm_dist_lag_1.isnull())]

d2 = df[(df.mean_dhm_dist_lag_1.isnull())]



d2 = pd.merge(d2[['date','hour','new_month']],group4, how = 'left', on = ['new_month'])



df = d1.append(d2)



del d1

del d2

gc.collect();
#now to create separate train and test datasets

train = df[df.new_month<16]

test = df[df.new_month==16]
group1 = data[data.new_month!=4].groupby(['date', 'new_month','hour']).agg({'cluster_label_0':['count']}) #this count wil act as #obs

group1.columns = ['num_rides']

group1.reset_index(inplace=True)



train = pd.merge(train, group1, on = ['date', 'new_month','hour'], how = 'left')
X_train = train[train.new_month < 14].drop(['num_rides'], axis=1)

Y_train = train[train.new_month < 14]['num_rides']

X_valid = train[train.new_month == 15].drop(['num_rides'], axis=1)  # will be used to run validation

Y_valid = train[train.new_month == 15]['num_rides']

X_test = test[test.new_month == 16]
#removing features based on feature imporrtance

feat_to_remove = [ 'count_m_cluster_0_lag_1','min_dhm_minutes_lag_1','mean_m_price_lag_1','count_m_cluster_0_lag_1',

                  'count_m_cluster_2_lag_1','min_dhm_dist_lag_1','max_dhm_dist_lag_1','mean_m_dist_lag_1',

                 'count_m_cluster_1_lag_1','min_dhm_minutes_lag_1','mean_m_price_lag_1','count_m_cluster_0_lag_1',

                'count_m_cluster_2_lag_1','min_dhm_dist_lag_1','max_dhm_dist_lag_1',

                  'max_dhm_ride_minutes_lag_1', 'count_m_time_type_0_lag_1',

                  'max_dm_price_lag_1']



X_train.drop(feat_to_remove,axis=1, inplace=True)

X_valid.drop(feat_to_remove,axis=1, inplace=True)

X_test.drop(feat_to_remove,axis=1, inplace=True)
print('Shape of X train:  ', X_train.shape)

print('Shape of X_valid:  ', X_valid.shape)

print('Shape of X_test:  ', X_test.shape)
#Defining function to calculate RMSE, for manual evaluation

from sklearn.metrics import mean_squared_error

from math import sqrt



def rmse(y_pred, y_test):

    rmse = sqrt(mean_squared_error(y_test,y_pred))

    return print(rmse)
# A parameter grid for XGBoost

params = {'min_child_weight':[10,12],  'subsample':[0.8],

'colsample_bytree':[0.9], 'max_depth': [12,14],'objective':['reg:squarederror'],'eta':[0.1]}





xgb = XGBRegressor(nthread=-1) 



grid = GridSearchCV(xgb, params,cv = 2,

                        n_jobs = 5,

                        verbose=False)

grid.fit(X_train, Y_train)



print(grid.best_params_)
#After finalizing params, performing model fit



xgb = XGBRegressor(colsample_bytree = 0.9,max_depth=12,

    n_estimators=150,

    min_child_weight=10,  

    subsample=0.8, 

    eta=0.1)



xgb.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_valid, Y_valid)],

        eval_metric = 'rmse', early_stopping_rounds =5)



plot_features(xgb, (10,30))


model1_train = xgb.predict(X_train).clip(0.0,8000.0)

model1_valid = xgb.predict(X_valid).clip(0.0,8000.0)

model1_test = xgb.predict(X_test).clip(0.0,8000.0)

print("RMSE for training dataset: ")

rmse(model1_train,Y_train)

print("RMSE for validation dataset: ")

rmse(model1_valid,Y_valid)
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1, max_iter=100)

lasso.fit(X_train, Y_train)

model3_train = lasso.predict(X_train).clip(0.0,8000.0)

model3_valid = lasso.predict(X_valid).clip(0.0,8000.0)

model3_test = lasso.predict(X_test).clip(0.0,8000.0)

print("RMSE for training dataset: ")

rmse(model3_train,Y_train)

print("RMSE for validation dataset: ")

rmse(model3_valid,Y_valid)
from sklearn.neural_network import MLPRegressor





mlp = MLPRegressor(hidden_layer_sizes=(32,32,32),activation='relu',alpha = 0.0001,max_iter = 1000,solver='lbfgs',random_state=6)



mlp.fit(X_train,Y_train)



model2_train = mlp.predict(X_train).clip(0.0,8000.0)

model2_valid = mlp.predict(X_valid).clip(0.0,8000.0)

model2_test = mlp.predict(X_test).clip(0.0,8000.0)

print("RMSE for training dataset: ")

rmse(model2_train,Y_train)

print("RMSE for validation dataset: ")

rmse(model2_valid,Y_valid)
p1 = 0

p2 = 1

p3 = 0

final_pred_train = (p1*np.array(model1_train) + p2*np.array(model2_train) + p3*np.array(model3_train))

final_pred_valid = (p1*np.array(model1_valid) + p2*np.array(model2_valid) + p3*np.array(model3_valid))

final_pred_test = (p1*np.array(model1_test) + p2*np.array(model2_test)+ p3*np.array(model3_test))
final_pred_train = pd.DataFrame({'pred':final_pred_train})

final_pred_train = final_pred_train.apply(lambda x : round(x,0))



final_pred_valid = pd.DataFrame({'pred':final_pred_valid})

final_pred_valid = final_pred_valid.apply(lambda x : round(x,0))



final_pred_test = pd.DataFrame({'pred':final_pred_test})

final_pred_test = final_pred_test.apply(lambda x : round(x,0))

final_pred_test = final_pred_test.reset_index()



print("RMSE for training dataset: ")

rmse(final_pred_train,Y_train)

print("RMSE for validation dataset: ")

rmse(final_pred_valid,Y_valid)
test = df[df.new_month==16]

final = test[['date','new_month','hour']]

final = final.reset_index()

final.drop(['index'],axis=1,inplace=True)

final['num_rides'] = final_pred_test['pred']

print(final.shape)

final.tail(2)
print("Total number of rides in April 2019: ")

num_rides_last = final.num_rides.sum()

print(num_rides_last)
final.to_csv('final_submission.csv', index=False)