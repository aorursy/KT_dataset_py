import sklearn.model_selection as model_selection

from sklearn import metrics

import xgboost as xgb

from sklearn.model_selection import GridSearchCV, cross_val_score

import matplotlib as mpl

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from statsmodels import api as sm

from scipy import stats

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge, Lasso,ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsRegressor

from pandas_profiling import ProfileReport

from sklearn.metrics import mean_squared_error, r2_score

import itertools

from sklearn.model_selection import cross_validate

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

import lightgbm as lgbm

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from datetime import datetime

from mlxtend.plotting import plot_learning_curves

from scipy.stats import kurtosis

from scipy.stats import skew

import warnings

warnings.simplefilter('ignore')



mpl.rcParams['agg.path.chunksize'] = 100000
farm_data = pd.read_csv("/content/drive/My Drive/PHD/farm_data.csv")
train_data = pd.read_csv("/content/drive/My Drive/PHD/train_data.csv")
test_data = pd.read_csv("/content/drive/My Drive/PHD/test_data.csv")
train_weather = pd.read_csv("/content/drive/My Drive/PHD/train_weather.csv")
test_weather = pd.read_csv("/content/drive/My Drive/PHD/test_weather.csv")
submission_file = pd.read_csv("/content/drive/My Drive/PHD/sample_submission1-1578562773139.csv")
submission_file_large = pd.read_csv("/content/drive/My Drive/PHD/sample_submission.csv")
ingw_demand = pd.read_excel("/content/drive/My Drive/PHD/Ing_w_demand.xlsx")
ProfileReport(farm_data)
farm_data = farm_data.drop_duplicates(subset=['farm_id'],keep='first')
#farm_data = farm_data.drop(columns=['operations_commencing_year','num_processing_plants'])
farm_data.shape
farm_data.isna().sum()
farm_data['deidentified_location'].unique()
sns.relplot(x='deidentified_location',y='farm_area',hue='farming_company',data=farm_data)

plt.xticks(rotation='90')
sns.boxplot(x=farm_data['farm_area'])
def remove_outlier(df_in, col_name):

    q1 = df_in[col_name].quantile(0.25)

    q3 = df_in[col_name].quantile(0.75)

    iqr = q3-q1 #Interquartile range

    low  = q1-1.5*iqr

    high = q3+1.5*iqr

    df_out = df_in.loc[(df_in[col_name] > low) & (df_in[col_name] < high)]

    return df_out
farm_data = remove_outlier(farm_data, 'farm_area')
farm_data.shape
train_data.head()
train_data.tail()
train_data.shape
train_data.isna().sum()
train_data  = remove_outlier(train_data, 'yield')
train_data.shape
ProfileReport(train_weather)
train_weather = train_weather.rename(columns= {'timestamp' : 'date'})
test_weather = test_weather.rename(columns= {'timestamp' : 'date'})
test_weather.columns
train_weather.columns
train_weather.shape
train_weather.isna().sum()
test_weather.isna().sum()
train_weather.describe()
total = train_weather.isnull().sum().sort_values(ascending=False)

percent = (train_weather.isnull().sum()/train_weather.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total,percent],axis=1, keys=['Total','Percent'])

f, ax = plt.subplots(figsize=(15,6))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index,y=missing_data['Percent'])

plt.xlabel('Features',fontsize=15)

plt.ylabel("Percent of missing values", fontsize=15)

plt.title('Percentage of missing values - Featurewise', fontsize = 15)

missing_data.head()
train_weather['cloudiness'] = train_weather['cloudiness'].fillna(train_weather['cloudiness'].mean())

train_weather['temp_obs'] = train_weather['temp_obs'].fillna(train_weather['temp_obs'].mean())

train_weather['wind_direction'] = train_weather['wind_direction'].fillna(train_weather['wind_direction'].mean())

train_weather['dew_temp'] = train_weather['dew_temp'].fillna(train_weather['dew_temp'].mean())

train_weather['pressure_sea_level'] = train_weather['pressure_sea_level'].fillna(train_weather['pressure_sea_level'].mean())

train_weather['precipitation'] = train_weather['precipitation'].fillna(train_weather['precipitation'].mean())

train_weather['wind_speed'] = train_weather['wind_speed'].fillna(train_weather['wind_speed'].mean())
test_weather['cloudiness'] = test_weather['cloudiness'].fillna(test_weather['cloudiness'].mean())

test_weather['temp_obs'] = test_weather['temp_obs'].fillna(test_weather['temp_obs'].mean())

test_weather['wind_direction'] = test_weather['wind_direction'].fillna(test_weather['wind_direction'].mean())

test_weather['dew_temp'] = test_weather['dew_temp'].fillna(test_weather['dew_temp'].mean())

test_weather['pressure_sea_level'] = test_weather['pressure_sea_level'].fillna(test_weather['pressure_sea_level'].mean())

test_weather['precipitation'] = test_weather['precipitation'].fillna(test_weather['precipitation'].mean())

test_weather['wind_speed'] = test_weather['wind_speed'].fillna(test_weather['wind_speed'].mean())
test_weather.isna().sum()
train_weather.isna().sum()
train_weather.shape
train_weather = remove_outlier(train_weather, 'wind_speed')
train_weather = remove_outlier(train_weather,'precipitation')
train_weather = remove_outlier(train_weather,'pressure_sea_level')
train_weather = remove_outlier(train_weather,'dew_temp')
train_weather = remove_outlier(train_weather,'wind_direction')
train_weather = remove_outlier(train_weather,'cloudiness')
train_weather = remove_outlier(train_weather,'temp_obs')
New_train = pd.merge(train_data,farm_data,on = 'farm_id',how = 'inner')
New_test = pd.merge(test_data,farm_data,on='farm_id',how='inner')
cmbd_train = pd.merge(New_train,train_weather,on = ['date','deidentified_location'],how ='inner')
cmbd_train = cmbd_train.drop(['operations_commencing_year','num_processing_plants'],axis=1)
cmbd_test = pd.merge(New_test,test_weather,on=['date','deidentified_location'],how='inner')
cmbd_test = cmbd_test.drop(['operations_commencing_year','num_processing_plants'],axis=1)
cmbd_test.shape
cmbd_train.shape
cmbd_train.columns
cmbd_train.head()
cmbd_train.describe(include='all')
N_ts = pd.merge(train_data,test_data,on = 'date',how = 'outer')
train_data['month'] = train_data['date'].dt.month 

train_data['day'] = train_data['date'].dt.day
train_data.head()
train_data.tail()
train_data.set_index(['month'], inplace=True)
ts = train_data['yield']
ts.head()
ts.plot()

plt.ylabel('Yield')

plt.xlabel('Month')

plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(ts, model = 'additive', freq=1)

result.plot()

plt.show()
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):

    #Determing rolling statistics

    rolmean = timeseries.rolling(12).mean()

    rolstd = timeseries.rolling(12).std()

    #Plot rolling statistics:

    plt.plot(timeseries, color='blue',label='Original')

    plt.plot(rolmean, color='red', label='Rolling Mean')

    plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean and Standard Deviation')

    plt.show(block=False)

    

test_stationarity(ts)
# Define the d and q parameters to take any value between 0 and 1

q = d = range(0, 2)

# Define the p parameters to take any value between 0 and 3

p = range(0, 4)



# Generate all different combinations of p, q and q triplets

pdq = list(itertools.product(p, d, q))



# Generate all different combinations of seasonal p, q and q triplets

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]



print('Examples of parameter combinations for Seasonal ARIMA...')

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
df_log = np.log(ts)

moving_avg = df_log.rolling(12).mean()

std_dev = df_log.rolling(12).std()

plt.plot(df_log)

plt.plot(moving_avg, color="red")

plt.plot(std_dev, color ="black")

plt.show()
df_log_moving_avg_diff = df_log-moving_avg

df_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(df_log_moving_avg_diff)
weighted_average = df_log.ewm(halflife=12, min_periods=0,adjust=True).mean()
logScale_weightedMean = df_log-weighted_average

from pylab import rcParams

rcParams['figure.figsize'] = 10,6

test_stationarity(logScale_weightedMean)
sns.jointplot(x="temp_obs", y="pressure_sea_level", data=train_weather, height=5)
plt.figure(figsize = (15,6))

plt.xticks(rotation='90')

sns.boxplot(x="deidentified_location", y="wind_speed", data=train_weather)

plt.show()
plt.figure(figsize = (15,6))

sns.scatterplot(cmbd_train['temp_obs'].dropna(), cmbd_train['yield'])

plt.show()
plt.figure(figsize = (15,6))

sns.scatterplot(cmbd_train['farm_area'], cmbd_train['yield'])

plt.show()
cmbd_train.dtypes
plt.figure(figsize=(12,8))

cmbd_train.plot()

plt.title('ingredient_type VS yield')

plt.xlabel('Ingredient_type')

plt.ylabel('yield')

plt.legend(['Yield'])
plt.figure(figsize=(15,10))

sns.heatmap(cmbd_train.corr(), annot=True)
cmbd_train.hist(column='yield', bins=3, grid=False);

plt.suptitle('Yield Year wise')

plt.xlabel('date')

plt.ylabel('yield')
plt.figure(figsize=(15, 7))

plt.plot(cmbd_train.temp_obs)

plt.title('Temp Obsevations')

plt.grid(True)

plt.show()
plt.hist(cmbd_train['ingredient_type']);
plt.figure(figsize=(15, 7))

plt.plot(cmbd_train.dew_temp)

plt.title('Dew Temp Obsevations')

plt.grid(True)

plt.show()
sns.relplot(x='dew_temp',y='yield',data=cmbd_train)  
plt.figure(figsize=(15, 7))

sns.relplot(x='wind_speed',y='yield',hue='ingredient_type',data=cmbd_train)
sns.relplot(x='month',y='yield',hue='ingredient_type',data=cmbd_train)
all_Data = []
cmbd_train = cmbd_train.drop(['farm_id','farming_company','deidentified_location','date'],axis =1)
cmbd_train.dtypes
cmbd_train.shape
#num_cols = ['farm_area','temp_obs','cloudiness','wind_direction','dew_temp','pressure_sea_level','precipitation','wind_speed']
#scaler = MinMaxScaler()
#scaler.fit(cmbd_train[num_cols])
#cmbd_train[num_cols] =scaler.transform(cmbd_train[num_cols])
train_dummies = pd.get_dummies(cmbd_train[["ingredient_type"]])
all_Data = cmbd_train.drop(["ingredient_type","yield",'farm_area'],axis=1).join(train_dummies)
all_Data.head()
cmbd_test = cmbd_test.drop(['farm_id','farming_company','deidentified_location','date'],axis=1)
cmbd_test.columns
#num_cols_test = ['farm_area','temp_obs','cloudiness','wind_direction','dew_temp','pressure_sea_level','precipitation','wind_speed']
#scaler.fit(cmbd_test[num_cols_test])
#cmbd_test[num_cols_test] =scaler.transform(cmbd_test[num_cols_test])
test_dummies = pd.get_dummies(cmbd_test[["ingredient_type"]])
all_Data_test = cmbd_test.drop(["ingredient_type",'farm_area','id'],axis=1).join(test_dummies)
all_Data_test.columns
all_Data_test.head()
all_Data_test.shape
X = all_Data
y = cmbd_train['yield']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
regressor = DecisionTreeRegressor(min_samples_split=5,random_state = 0)
regressor.fit(X, y)
y_pred_train = regressor.predict(X_train)

y_pred_test = regressor.predict(X_val)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

rmse_test = np.sqrt(mean_squared_error(y_val, y_pred_test))
print('Decision Tree:')

print('root mean squared error for train: ', rmse_train)

print('root mean squared error validation: ', rmse_test)
plot_learning_curves(X_train,y_train,X_val,y_val,regressor,scoring='mean_squared_error')
all_Data_test.columns
test_predict_dt = regressor.predict(all_Data_test)
test_predict_dt = pd.DataFrame(test_predict_dt,index = None,columns=["yield"])
test_submission_id = pd.DataFrame(submission_file,columns=["id"])
test_submission_id_large = pd.DataFrame(submission_file_large,columns=["id"])
full_test_dt = pd.concat([test_submission_id,test_predict_dt],axis=1)
full_test_dt_large = pd.concat([test_submission_id_large,test_predict_dt],axis=1)
full_test_dt_large.shape
full_test_dt.shape
full_test_dt = full_test_dt.iloc[0:999999,:]
full_test_dt.shape
full_test_dt_large.to_csv("Predictions_whole.csv",index=False,header=True)
full_test_dt.to_csv("test_predictions_dt_99.csv",index=False,header=True )
plt.scatter(y_val, y_pred_test)

plt.ylim()

plt.xlim()
print("MAE:", metrics.mean_absolute_error(y_val, y_pred_test))

print('MSE:', metrics.mean_squared_error(y_val, y_pred_test))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_val, y_pred_test)))
from sklearn.linear_model import LinearRegression
lm = LinearRegression(normalize=True,fit_intercept=False,n_jobs=-1)
lm.fit(X,y)
y_pred_train = regressor.predict(X_train)

y_pred_test = regressor.predict(X_val)
plt.scatter(y_val, predictions)

plt.ylim()

plt.xlim()
print("MAE:", metrics.mean_absolute_error(y_val, predictions))

print('MSE:', metrics.mean_squared_error(y_val, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_val, predictions)))
plot_learning_curves(X_train,y_train,X_val,y_val,lm,scoring='mean_squared_error')
full_test_dt_large.shape
test_data.columns
test_data['date']= pd.to_datetime(test_data['date'])
test_data['Month'] = test_data['date'].dt.month
test_data_ing = pd.DataFrame(test_data,columns=["ingredient_type","Month"])
test_data_ing.head()
test_data_ing.shape
ing_type = pd.concat([full_test_dt_large,test_data_ing],axis=1)
ing_type.head()
ing_type = ing_type.drop(['id'],axis=1)
ing_type.head()
ing_type['ingredient_type'].unique()
ing_type['ingredient_type'].value_counts()
ing_w = ing_type[ing_type.ingredient_type == 'ing_w']
ing_w.shape
new_ing = ing_w.iloc[0:12377880]
new_ing_w = new_ing.dropna()
new_ing_w.shape
new_ing_w.head()
new_ing_w.tail()
new_ing_w['Month'].value_counts()
Overall_stock =new_ing_w['yield'].sum()
Overall_demand = ingw_demand['demand'].sum()
Overall_stock - Overall_demand
Monthly = new_ing_w.groupby(['Month']).sum()
Monthly = Monthly.reset_index()
Monthly = Monthly.rename(columns={'Month':'month'})
ingw_demand.head(12)
stock_demand = pd.merge(Monthly,ingw_demand,on='month',how='inner')
stock_demand.head()
stock_demand['Balance'] = stock_demand['yield'][0]-stock_demand['demand'][0] 
stock_demand = stock_demand.set_index('month')
stock_demand.head()
stock_demand['Balance'][0] = stock_demand['Balance'].iloc[0]

import pandas as pd

farm_data = pd.read_csv("../input/farm_data.csv")

sample_submission = pd.read_csv("../input/sample_submission.csv")

test_data = pd.read_csv("../input/test_data.csv")

test_weather = pd.read_csv("../input/test_weather.csv")

train_data = pd.read_csv("../input/train_data.csv")

train_weather = pd.read_csv("../input/train_weather.csv")