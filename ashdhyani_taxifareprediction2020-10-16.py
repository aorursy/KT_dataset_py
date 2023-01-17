import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import StackingRegressor
## Load 100k rows only
data = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=100_000, parse_dates=['pickup_datetime'])

print(data.shape)
print(data.info())
data.head()
data.describe()
data[data.fare_amount<100].fare_amount.hist(bins=100, figsize=(14,3))
plt.xlabel('fare $USD')
plt.title('Histogram');
from math import sin, cos, sqrt, atan2, radians

def calculateDistance(lt1, ln1, lt2, ln2):

    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lt1)
    lon1 = radians(ln1)
    lat2 = radians(lt2)
    lon2 = radians(ln2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c * 1000
    
    return distance
def featureCleanup(dfOrig, train = True):
    if(train):
        df = dfOrig[dfOrig['fare_amount'] >= 0]
    else:
        df = dfOrig.copy()
        
    df['weekday'] = df['pickup_datetime'].dt.day_name()
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_time'] = df['pickup_datetime'].dt.hour + df['pickup_datetime'].dt.minute/60
    
    df['distance'] = df.apply(lambda x: 
                              calculateDistance(x['pickup_latitude'], 
                                                x['pickup_longitude'],
                                                x['dropoff_latitude'],
                                                x['dropoff_longitude']), 
                              axis=1)
    
    df.drop(columns = ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','pickup_datetime','key'], 
          inplace = True)
    
    if(train):
        df.dropna(
            axis=0,
            how='any',
            thresh=None,
            subset=None,
            inplace=True
        )

        df = df[df['distance'] > 0]
    
    return df
trainData = featureCleanup(data)
trainData.head()
def plotChart(df, x, y, title, num):
    plt.subplot(5, 2, num)
    sns.lineplot(data = df, x= x, y = y)
    plt.title(title)
    #plt.xticks(rotation = 90)
    plt.legend(loc='upper right')
plt.figure(figsize  = (15,30))
plotChart(trainData.groupby(by="weekday").mean().reset_index(), 'weekday', 'fare_amount', 'weekday vs fare', 1)
plotChart(trainData.groupby(by="weekday").mean().reset_index(), 'weekday', 'distance', 'weekday vs distance', 2)
plotChart(trainData.groupby(by="pickup_hour").mean().reset_index(), 'pickup_hour', 'fare_amount', 'hour vs fare', 3)
plotChart(trainData.groupby(by="distance").mean().reset_index(), 'distance', 'fare_amount', 'distance vs fare', 4)
plt.figure(figsize  = (20,40))
for i in enumerate(trainData.columns.drop(['fare_amount', 'distance', 'pickup_time'])):
    plt.subplot(10, 2, i[0]+1)
    sns.countplot(trainData[i[1]])

trainData.drop(columns=['pickup_hour'], inplace=True)
trainData['weekday'] = trainData['weekday'].map({"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7})
y_train = trainData.pop('fare_amount')
X_train = trainData
X_train.head()
params = {
    'max_depth': [4,5,6,7,8,9,10]
}
# Instantiate the grid search model

dt = DecisionTreeRegressor(random_state=100)

grid_search = GridSearchCV(estimator=dt, param_grid = params, 
                          cv=4, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
grid_search.best_estimator_
y_train_predict = grid_search.predict(X_train)
print("Decision Tree Accuracy:", round(r2_score(y_train, y_train_predict)*100, 2), "%")
rfEstimator = RandomForestRegressor(random_state=42)
para_grids = {
            "n_estimators" : [50],
            "max_depth": [6,7,8],
            'max_features': [2,3,4]
        }
grid_rf = GridSearchCV(rfEstimator, para_grids, verbose=1, n_jobs=-1, cv=5)
grid_rf.fit(X_train, y_train)
grid_rf.best_estimator_
y_train_pred_rf = grid_rf.predict(X_train)
print("Random Forest Accuracy:", round(r2_score(y_train, y_train_pred_rf)*100, 2), "%")
xg_reg = xgb.XGBRegressor(n_jobs=-1)
xg_reg.fit(X_train,y_train)
from sklearn import metrics

y_train_pred_xg = xg_reg.predict(X_train)
y_train_pred_xg
print("XGBoost Accuracy:", round(r2_score(y_train, y_train_pred_xg)*100, 2), "%")
para_grids = {
            "n_estimators": [100,200],
            "learning_rate": [0.3,0.4,0.5],
            "max_depth": [6,7,8]
        }

grid_xg = GridSearchCV(xg_reg, para_grids, verbose=1, n_jobs=-1, cv=4)
grid_xg.fit(X_train, y_train)
grid_xg.best_estimator_
y_train_pred_xg_cv = grid_xg.predict(X_train)
y_train_pred_xg_cv
print("XGBoost Accuracy after Hyperparameter tuning:", round(r2_score(y_train, y_train_pred_xg_cv)*100, 2), "%")
xgb.plot_tree(grid_xg.best_estimator_,num_trees=0)
plt.show()
base_learners = [
                 ('es1', xg_reg),
                 ('es2', grid_rf.best_estimator_)     
                ]
stregr = StackingRegressor(estimators=base_learners, cv=4,n_jobs=1,verbose=1)
stregr.fit(X_train, y_train)
y_predict_stack_reg = stregr.predict(X_train)
print("Accuracy:", round(r2_score(y_train, y_predict_stack_reg)*100, 2), "%")

test = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv", parse_dates=['pickup_datetime'])

testData = featureCleanup(test, False)

testData.head()
testData.drop(columns=['pickup_hour'], inplace=True)
testData['weekday'] = testData['weekday'].map({"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7})
y_test_pred_xg_cv = grid_xg.predict(testData)
y_test_pred_xg_cv
test['fare_amount_predicted'] = y_test_pred_xg_cv
test.head()
