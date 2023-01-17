#Data handling

import pandas as pd

import numpy as np



#modelling

from catboost import CatBoostRegressor, Pool, cv

from sklearn.feature_selection import SelectKBest, chi2 , f_regression



#visualization

import folium

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="darkgrid")

%matplotlib inline



#display settings

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
day_list= [26,27,28,29,30,31]
day_samples = []

for day in day_list:

    sample_df = pd.read_csv('../input/dubai-metro-ridership-data/metro_ridership_2018-03-'+str(day)+'_00-00-00.csv')

    day_samples.append(sample_df)
final_df = pd.concat(day_samples)
final_df.head()
stations=pd.read_csv('../input/dubai-metro-ridership-data/Metro_Stations.csv')
stations.head()
final_df = pd.merge(final_df, stations, how='inner',left_on='end_location',right_on='station_name')
final_df.shape
final_df.head()
final_df = final_df.drop(['start_location', 'start_location','line_name_x','start_zone','end_zone'], axis=1)
final_df.head()
final_df.dtypes
final_df["txn_date"] = pd.to_datetime(final_df["txn_date"], format="%Y-%m-%d")
final_df["txn_time"] = pd.to_datetime(final_df["txn_time"], format="%H:%M:%S")
final_df['Weekday'] = final_df['txn_date'].dt.day_name()
final_df['Weekday_numeric'] = final_df['txn_date'].dt.dayofweek
final_df['Hour'] = final_df['txn_time'].dt.hour
final_df.head()
final_df_grouped = final_df.groupby(['station_name','line_name_y','station_location_longitiude','station_location_latitiude','Weekday','Weekday_numeric','Hour'])['txn_type'].count()

final_df_grouped = final_df_grouped.reset_index()
final_df_grouped = final_df_grouped.rename(columns={"txn_type": "transaction_count"})
final_df_grouped.head()
from folium.plugins import HeatMap

base_map = folium.Map([final_df_grouped['station_location_latitiude'].mean(),final_df_grouped['station_location_longitiude'].mean()], zoom_start=11)

hm = HeatMap(data=final_df_grouped[['station_location_latitiude','station_location_longitiude','transaction_count']].groupby(['station_location_latitiude','station_location_longitiude']).sum().reset_index().values.tolist(), radius=12, max_zoom=13).add_to(base_map)
base_map
df_hour_list = []

for hour in final_df_grouped.Hour.sort_values().unique():

    df_hour_list.append(final_df_grouped.loc[final_df_grouped.Hour == hour, ['station_location_latitiude','station_location_longitiude', 'transaction_count']].groupby(['station_location_latitiude','station_location_longitiude']).sum().reset_index().values.tolist())
from folium.plugins import HeatMapWithTime

base_map = folium.Map([final_df_grouped['station_location_latitiude'].mean(),final_df_grouped['station_location_longitiude'].mean()], zoom_start=12)

HeatMapWithTime(df_hour_list, radius=25, gradient={0: 'blue', 0.2: 'lime', 0.3: 'orange', 0.5: 'red'}, min_opacity=0.5, max_opacity=0.8, use_local_extrema=True).add_to(base_map)

base_map
final_df_grouped.head()
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))

ax = sns.lineplot(x="Hour", y="transaction_count", data=final_df_grouped,style ='Weekday',hue="Weekday",markers=True, dashes=False)
import numpy as np 

corrMatt = final_df_grouped[['station_location_longitiude','station_location_latitiude','Weekday_numeric','Hour','transaction_count']].corr()

mask = np.array(corrMatt)

mask[np.tril_indices_from(mask)] = False

fig,ax= plt.subplots()

fig.set_size_inches(20,10)

sns.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=False,cmap='RdYlGn')
train_test_mask = np.random.rand(len(final_df_grouped)) < 0.8
ads_train_df = final_df_grouped[train_test_mask]

ads_test_df = final_df_grouped[~train_test_mask]
ads_train_df.shape
ads_test_df.shape
ads_train_df.dtypes
X = ads_train_df.drop(['transaction_count'], axis=1)

y = ads_train_df.transaction_count
from sklearn.model_selection import train_test_split



X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8, random_state=2019)
from sklearn.model_selection import GridSearchCV
cat_features = ['station_name','line_name_y','Weekday']
model = CatBoostRegressor()
parameters = {'depth'         : [6,10],

              'learning_rate' : [0.01,0.5,1],

              'iterations'    : [30, 100],

              'loss_function' : ['RMSE']

             }
#grid = GridSearchCV(estimator=model, param_grid = parameters, cv = 3, n_jobs=-1)

#grid.fit(X_train, y_train,cat_features=cat_features)
#print("\n========================================================")

#print(" Results from Grid Search " )

#print("========================================================")    



#print("\n The best estimator across ALL searched params:\n",

#      grid.best_estimator_)



#print("\n The best score across ALL searched params:\n",

#      grid.best_score_)



#print("\n The best parameters across ALL searched params:\n",

#      grid.best_params_)



#print("\n")
from catboost import CatBoostRegressor, Pool, cv

from sklearn.metrics import accuracy_score
model=CatBoostRegressor(iterations=100, depth=10, learning_rate=0.5, loss_function='RMSE')
model.fit(

    X_train, y_train,

    cat_features=cat_features,

    eval_set=(X_validation, y_validation),

    logging_level='Silent',  # you can uncomment this for text output

    plot=False

);
Y_test = ads_test_df.transaction_count

X_test = ads_test_df.drop(['transaction_count'], axis=1)
predictions = model.predict(X_test)
prediction_df = pd.DataFrame() 
prediction_df['Actual'] = Y_test
prediction_df['Predicted'] = predictions
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error

import numpy as np

def calculate_error( g ):

    r2 = r2_score( g['Actual'], g['Predicted'] )

    rmse = np.sqrt( mean_squared_error( g['Actual'], g['Predicted'] ) )

    mea = mean_absolute_error(g['Actual'],g['Predicted'])

    return pd.Series( dict(  r2 = r2, rmse = rmse , mea=mea) )
calculate_error(prediction_df)
prediction_df.to_csv("final_submission.csv", index=False)