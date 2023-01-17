# Importing Libraries
import numpy as np # linear algebra
import pandas as pd 
train = pd.read_csv('/kaggle/input/machine-hack-house-dataset/Participants_Data_HPP/Train.csv')
test = pd.read_csv('/kaggle/input/machine-hack-house-dataset/Participants_Data_HPP/Test.csv')
sub = pd.read_csv('/kaggle/input/machine-hack-house-dataset/Participants_Data_HPP/sample_submission.csv')
train.head()
train['TARGET(PRICE_IN_LACS)'] = np.log(train['TARGET(PRICE_IN_LACS)'])
data = pd.concat([train,test],axis=0)
data
for i in data.describe(include='O'):
    print(i,'--',data[i].nunique())
data.info()
data.LATITUDE.nunique()
# lets combine latitude and longitude
data['lat_long'] = (data['LATITUDE'])+(data['LATITUDE'])
data
#lets drop address and lat and long columns
data.drop(columns=['LONGITUDE','LATITUDE'], inplace = True)
data
data['BHK_OR_RK'].unique()
# lets convert categorical data
posted = {'Owner':0,
         'Dealer':1,
            'Builder':2}
data['POSTED_BY'] = data['POSTED_BY'].map(posted)

bhk = {'BHK':0,
        'RK':1}
data['BHK_OR_RK'] = data['BHK_OR_RK'].map(bhk)
from scipy.stats import skew
for i in data.describe():
    print(i,'--',skew(data[i]))
# Lets normalize Square feet, Resale, lat_long
data ['SQUARE_FT'] = (np.log(data['SQUARE_FT']))
# data['lat_long'] = np.log(data['lat_long'])
data.head()
data['lat_long'] = abs(data['lat_long'])
data['lat_long_sq_ft_mean'] = data.groupby(['lat_long'])['SQUARE_FT'].transform('mean')
# data['lat_long_sq_ft_std'] = data.groupby(['lat_long'])['SQUARE_FT'].transform('std')
data['lat_long_sq_ft_count'] = data.groupby(['lat_long'])['SQUARE_FT'].transform('count')
data['POSTED_BY_sq_ft_mean'] = data.groupby(['POSTED_BY'])['SQUARE_FT'].transform('mean')
# data['POSTED_BY_sq_ft_std'] = data.groupby(['POSTED_BY'])['SQUARE_FT'].transform('std')
data['BHK_NO_sq_ft_mean'] = data.groupby(['BHK_NO.'])['SQUARE_FT'].transform('mean')
# data['BHK_NO_sq_ft_std'] = data.groupby(['BHK_NO.'])['SQUARE_FT'].transform('std')
data['lat/sq_ft'] = data['lat_long'] / data['SQUARE_FT']
data['lat/sq_ft_mean'] = data.groupby(['lat/sq_ft'])['SQUARE_FT'].transform('mean')
data['sq_per_room']=data['SQUARE_FT']/data['BHK_NO.']
data['BHK_NO_price_mean'] = data.groupby(['BHK_NO.'])['TARGET(PRICE_IN_LACS)'].transform('mean')
# Extracting name of city and locality of house
import re
def city(address):
 city_name=address.split(',')[-1]
 return city_name
def locality(address):
 locality=address.split(',')[-2]
 return locality
data['loc']=data['ADDRESS'].apply(lambda x : locality(x))
data['City']=data['ADDRESS'].apply(lambda x : city(x))
Encoding = data.groupby('City')['TARGET(PRICE_IN_LACS)'].mean()
data['City_mean']= data.City.map(Encoding )
# for i in data['ADDRESS']:
#     city_name=i.split(',')[1]
#     print(city_name)
data.iloc[:,4:]
data.drop(columns=['ADDRESS','loc','City'], inplace=True)
Train  = data.iloc[:len(train)]
X = Train.drop(columns='TARGET(PRICE_IN_LACS)')
Y = Train['TARGET(PRICE_IN_LACS)']
X.isnull().sum()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
from  sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
# from gbm import LGBMRegressor
model = RandomForestRegressor()
model.fit(x_train, y_train)
# let see most important Features
imp = model.feature_importances_
top_imp = [(a*100)/sum(imp) for a in imp]
feature_importance = pd.DataFrame()
feature_importance['features'] = x_train.columns
feature_importance['% imp'] = top_imp
feature_importance.sort_values(by ='% imp',ascending=False).reset_index(drop=True)
pred = (model.predict(x_test))
y_test = np.exp(y_test)   
y_test
pred=  np.abs((np.exp(pred)-1))
pred

from sklearn.metrics import mean_squared_log_error
print(np.sqrt(mean_squared_log_error(pred, abs(y_test))))
! pip install pycaret
from pycaret.regression import *
session_1 = setup(data=Train, target='TARGET(PRICE_IN_LACS)')
allm = models()
allm
lgbm_model = create_model('catboost',fold = 5)
tune_gbm = tune_model(lgbm_model)
final_model = finalize_model(tune_gbm)
Test = data.iloc[len(Train):]
Test = Test.drop(columns='TARGET(PRICE_IN_LACS)')
final_prediction = predict_model(final_model, data = Test)
final_prediction
a = final_prediction['Label']
final_prediction = np.abs((np.exp(a)-1))
sub['TARGET(PRICE_IN_LACS)'] = final_prediction
sub.to_csv('sub_new.csv')