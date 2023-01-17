# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
pd.options.display.max_columns=200

import math
#import pyproj

import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline

from geopy.distance import great_circle

import lightgbm as lgb
from lightgbm import LGBMRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.inspection import permutation_importance

from tqdm import tqdm_notebook as tqdm
df_train=pd.read_csv('../input/machine-learning-homework/train.csv', index_col=0)
df_test=pd.read_csv('../input/machine-learning-homework/test.csv', index_col=0)
df_city=pd.read_csv('../input/machine-learning-homework/city_info.csv', index_col=0)
df_station=pd.read_csv('../input/machine-learning-homework/station_info.csv')
df_train.insert(9,"AverageTimeToNearestStation",0)
df_train['AverageTimeToNearestStation']=df_train['MinTimeToNearestStation']*0.5+df_train['MaxTimeToNearestStation']*0.5
df_test.insert(9,"AverageTimeToNearestStation",0)
df_test['AverageTimeToNearestStation']=df_test['MinTimeToNearestStation']*0.5+df_test['MaxTimeToNearestStation']*0.5
df_train=df_train.merge(df_station,left_on="NearestStation", right_on="Station", how='left')
df_test=df_test.merge(df_station,left_on="NearestStation", right_on="Station", how='left')
def calc_distance(df, dist):
    return great_circle((df.Latitude, df.Longitude), dist).meters
station = 'Tokyo'
lat_dist = df_station[df_station.Station==station].Latitude.values[0]
lon_dist = df_station[df_station.Station==station].Longitude.values[0]

df_train.loc[df_train.Latitude.isnull()==False, 'distance_%s'%station] = \
df_train[df_train.Latitude.isnull()==False].apply(calc_distance, dist=(lat_dist, lon_dist), axis=1)
station = 'Tokyo'
lat_dist = df_station[df_station.Station==station].Latitude.values[0]
lon_dist = df_station[df_station.Station==station].Longitude.values[0]

df_test.loc[df_test.Latitude.isnull()==False, 'distance_%s'%station] = \
df_test[df_test.Latitude.isnull()==False].apply(calc_distance, dist=(lat_dist, lon_dist), axis=1)
y_train=df_train.TradePrice
X_train=df_train.drop(['TradePrice'],axis=1)
X_test=df_test.copy()
X_concat=pd.concat([X_train, X_test])
print(len(X_concat))
station = 'Tokyo'
lat_dist = df_station[df_station.Station==station].Latitude.values[0]
lon_dist = df_station[df_station.Station==station].Longitude.values[0]

df_city.loc[df_city.Latitude.isnull()==False, 'distance2_%s'%station] = \
df_city[df_city.Latitude.isnull()==False].apply(calc_distance, dist=(lat_dist, lon_dist), axis=1)
X_concat=X_concat.merge(df_city,on="Municipality", how='left')
X_concat2=X_concat.copy()
X_concat2.shape
print(X_concat2.isnull().sum())
X_concat2.insert(11,"FloorPlanNumber",0)
X_concat2.insert(25,'FrontageDirectionFlag',0)
X_concat2.insert(27,"PrivateRoadFlag",0)
X_concat2.FloorPlanNumber=X_concat2.FloorPlanNumber.astype(object)
X_concat2.FloorPlanNumber=X_concat2.FloorPlan
X_concat2.FrontageDirectionFlag=X_concat2.FrontageDirectionFlag.astype(object)
X_concat2.FrontageDirectionFlag=X_concat2.Direction
X_concat2.PrivateRoadFlag=X_concat2.PrivateRoadFlag.astype(object)
X_concat2.PrivateRoadFlag=X_concat2.Classification
X_concat2.insert(1,'TypeFlag',0)
X_concat2.TypeFlag=X_concat2.Type
X_concat2.insert(3,"RegionFlag",0)
X_concat2.RegionFlag=X_concat2.Region
X_concat2.insert(35,'Year2',0)
X_concat2.insert(43,'kencho',0)
X_concat2.insert(15,'EconomicArea',0)
replace_dict_rooms={
    '3LDK':4, '4DK':5, '2LDK': 3, '4LDK':5, '2DK':3, '1K':2, '3LDK+S':5, '5LDK':6, '3DK':4, '1LDK':2,
    '2DK+S':4, 'Open Floor':2, '1DK':2, '1R':2, '4LDK+S':5, '2K':3, '2LDK+S':4, '6DK':7, '1LDK+S':2, '5DK':6,
    '1R+S':2, '1LK':2, '1K+S':2, '3K':4, '7LDK':8, '4K':5, '3DK+S':4, '3D':4, '1DK+S':3, '6LDK':7,
    'Studio Apartment':1,'6LDK+S':8, '4L+K':6, '5LDK+S':7, '7DK':9, '3LK':5, '5K':6, '2K+S':4, '8LDK':10, '3LDK+K':5, '3LD':4,
    '1L':1, '4DK+S':7, '2LK':4, 'Duplex':6,'7LDK+S':9, '4LDK+K':6, '3LD+S':5, '2LD+S':4, '8LDK+S':10, '4L':4, '2L':2,
    '2LDK+K':4, '2LK+S':4, '5LDK+K':7, '1LD+S':3, '2L+S':4, '3K+S':5, '1DK+K':3, '2LD':3, '1L+S':2, '2D':3, '4D':5}
X_concat2.FloorPlanNumber.replace(replace_dict_rooms, inplace=True)
replace_dict_facing_road={'No facing road':1, 'Southwest':0, 'Northwest':0, 'East':0, 'Northeast':0, 'Southeast':0,
                          'South':0, 'West':0, 'North':0}
X_concat2.FrontageDirectionFlag.replace(replace_dict_facing_road, inplace=True)
replace_dict_private_road={'Private Road':1, 'Road':0, 'City Road':0, 'Prefectural Road':0, 'Village Road':0, 
                           'National Highway':0,
                           'Access Road':0, 'Agricultural Road':0, 'Ward Road':0, 'Town Road':0, 'Kyoto/ Osaka Prefectural Road':0,
                           'Forest Road':0, 'Hokkaido Prefectural Road':0, 'Tokyo Metropolitan Road':0}
X_concat2.PrivateRoadFlag.replace(replace_dict_private_road, inplace=True)
replace_dict_type_flag={'Residential Land(Land Only)':1, 'Agricultural Land':0,
                           'Residential Land(Land and Building)':1, 'Pre-owned Condominiums, etc.':1,
                           'Forest Land':0}
X_concat2.TypeFlag.replace(replace_dict_type_flag, inplace=True)
replace_dict_region_flag={'Residential Area':0, 'Potential Residential Area':0, 'Commercial Area':1,
                        'Industrial Area':0}
X_concat2.RegionFlag.replace(replace_dict_region_flag, inplace=True)
X_concat2['Year2']=X_concat2['Year']+X_concat2['Quarter']*0.25
print(X_concat2.isnull().sum())
#X_concat2['FloorAreaRatio'].fillna(200, inplace=True)
X_concat2['RegionFlag'].fillna(0, inplace=True)
X_concat2['EconomicArea']=X_concat2['Area']*(X_concat2['TypeFlag']+X_concat2['RegionFlag']+0.001)*X_concat2['FloorAreaRatio']/100
X_concat2.head()
X_concat2.drop(['Latitude_x'], axis=1, inplace=True)
X_concat2.drop(['Longitude_x'],axis=1, inplace=True)
X_concat2.drop(['Latitude_y'], axis=1, inplace=True)
X_concat2.drop(['Longitude_y'],axis=1, inplace=True)
#X_concat2.drop(['Latitude'], axis=1, inplace=True)
#X_concat2.drop(['Longitude'], axis=1, inplace=True)
X_concat2.drop(['TimeToNearestStation'], axis=1, inplace=True)
X_concat2.drop(['Station'], axis=1, inplace=True)
X_concat2.drop(['Year','Quarter'], axis=1, inplace=True)
X_concat2.drop(['kencho'], axis=1, inplace=True)
X_concat2.drop(['distance_Tokyo'], axis=1, inplace=True)
X_concat2.drop(['EconomicArea'], axis=1, inplace=True)
X_concat2.drop(['MaxTimeToNearestStation'], axis=1, inplace=True)
X_concat2.drop(['AverageTimeToNearestStation'], axis=1, inplace=True)
#X_concat2.drop(['Municipality'])
#print(df_station[df_station.Station=='Yokohama'].Latitude.values[0])
#print(df_station[df_station.Station=='Yokohama'].Ltitude.values[0])
#緯度、経度情報(station_infoから）→Yokohama(35.46579,139.6223),Omiya(35.90645,139.6239), 
#Takasaki(36.32283,139.0127), Mito(36.37076,140.4763),Utsunomiya(36.55902,139.8985),
#Chiba(35.61313,140.1134)
#X_concat2.head()
X_concat2[X_concat2.MinTimeToNearestStation.isnull()==True]
#MinTimeToNearestStationの欠損を補完

null_ix=X_concat2[X_concat2.MinTimeToNearestStation.isnull()==True].index
summary=X_concat2.groupby(['DistrictName'])['MinTimeToNearestStation'].median()
#summary
X_concat2.loc[null_ix, 'MinTimeToNearestStation']=X_concat2.DistrictName.map(summary)

null_ix=X_concat2[X_concat2.MinTimeToNearestStation.isnull()==True].index
summary=X_concat2.groupby(['Municipality'])['MinTimeToNearestStation'].median()
X_concat2.loc[null_ix, 'MinTimeToNearestStation']=X_concat2.Municipality.map(summary)
#X_concat2.loc[null_ix]


#Municipalityにも最短時間が無いのは離島なので、12時間を代入
X_concat2['MinTimeToNearestStation'].fillna(1200,inplace=True)
#FloorAreaRatioの欠損を補完

null_ix=X_concat2[X_concat2.FloorAreaRatio.isnull()==True].index
summary=X_concat2.groupby(['DistrictName'])['FloorAreaRatio'].median()
#summary
X_concat2.loc[null_ix, 'FloorAreaRatio']=X_concat2.DistrictName.map(summary)

null_ix=X_concat2[X_concat2.FloorAreaRatio.isnull()==True].index
summary=X_concat2.groupby(['Municipality'])['FloorAreaRatio'].median()
X_concat2.loc[null_ix, 'FloorAreaRatio']=X_concat2.Municipality.map(summary)
#X_concat2.loc[null_ix]

X_concat2['FloorAreaRatio'].fillna(200, inplace=True) #中央値を代入


X_concat2.loc[null_ix]
print(X_concat2.isnull().sum())
print(X_concat2.median())
X_concat2.head()
#フロア数の中央値が3、面積の中央値が120なので、フロア数は面積/40でNaNを穴埋め計算
#X_concat2['FloorPlanNumber'].fillna(X_concat['Area']/40,inplace=True)
#面積中央値が120、容積率中央値が200、総面積中央値が200なので、
X_concat2['TotalFloorArea'].fillna(X_concat2['Area']*X_concat2['FloorAreaRatio']/100*X_concat2['TypeFlag']*0.416, inplace=True)
X_concat2.head()
#X_concat2.to_csv('hasebe_analysis_2')
#scaler=StandardScaler()
#X_concat2['Area']=scaler.fit_transform(X_concat2['Area'])
for col in X_concat2.columns:
    if (X_concat2[col].dtype == 'float64'):
        scaler=StandardScaler()
        X_concat2[col]=scaler.fit_transform(X_concat2[col].values.reshape(-1,1))
for col in X_concat2.columns:
    if (X_concat2[col].dtype == "object"):
        le = LabelEncoder()
        X_concat2[col]=le.fit_transform(X_concat2[col].fillna('NaN'))

X_train=X_concat2[X_concat2.index.isin(X_train.index)].fillna(-99999)
X_test=X_concat2[~X_concat2.index.isin(X_train.index)].fillna(-99999)
X_concat2.head()
groups=X_train.Prefecture.values
X_train.drop(['Prefecture'], axis=1, inplace=True)
X_test.drop(['Prefecture'],axis=1, inplace=True)
X_train.drop(['Municipality'], axis=1, inplace=True)
X_test.drop(['Municipality'], axis=1, inplace=True)
X_train.drop(['DistrictName'], axis=1, inplace=True)
X_test.drop(['DistrictName'], axis=1, inplace=True)
X_train.head()
X_test.head()
n_fold = 5
cv = GroupKFold(n_splits=n_fold)

y_pred_train = np.zeros(len(X_train))
y_pred_test = np.zeros(len(X_test))
scores = []

for i, (train_index, val_index) in enumerate(cv.split(X_train, y_train, groups)):
    X_train_, y_train_ = X_train.iloc[train_index], y_train.iloc[train_index]
    X_val, y_val = X_train.iloc[val_index], y_train.iloc[val_index]
    
    #model = HistGradientBoostingRegressor(learning_rate=0.05, random_state=71, max_iter=500)
    model = LGBMRegressor()
    model.fit(X_train_, np.log1p(y_train_))
    y_pred_val = np.expm1(model.predict(X_val))
    y_pred_test += np.expm1(model.predict(X_test))/n_fold
    
    y_pred_train[val_index] = y_pred_val
    score = mean_squared_log_error(y_val, y_pred_val)**0.5
    scores.append(score)
    
    print("Fold%d RMSLE: %f"%(i, score))
    
print("Overall RMSLE: %f±%f"%(np.mean(scores), np.std(scores)))
val_area=['Nerima Ward', 'Kita Ward', 'Itabashi Ward']
X_val2= X_train[df_train.Municipality.isin(val_area)]
y_val2= y_train[df_train.Municipality.isin(val_area)]

X_train2=X_train[~df_train.Municipality.isin(val_area)]
y_train2=y_train[~df_train.Municipality.isin(val_area)]
X_val2.head()
X_train2.head()
model=HistGradientBoostingRegressor(learning_rate=0.05, random_state=71, max_iter=500)
model.fit(X_train2, np.log1p(y_train2))
result = permutation_importance(model, X_val2, np.log1p(y_val2), n_repeats=10, random_state=71)

perm_sorted_idx=result.importances_mean.argsort()
num_features=len(X_val2.columns)

plt.figure(figsize=[8,15])
plt.title('Permutation Importance')
plt.barh(range(num_features), result['importances_mean'][perm_sorted_idx], xerr=result['importances_std'][perm_sorted_idx])
plt.yticks(range(num_features), X_val.columns[perm_sorted_idx])
plt.show()
df_sub=pd.read_csv('../input/machine-learning-homework/sample_submission.csv', index_col=0)
df_sub.TradePrice=y_pred_test
df_sub.to_csv('submission.csv')
