import pandas as pd 
import datetime
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from pylab import rcParams
import warnings
from pandas.core.nanops import nanmean as pd_nanmean
from tqdm.notebook import tqdm

from sklearn.metrics import mean_absolute_error

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from scipy.signal import periodogram
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from itertools import chain

warnings.filterwarnings('ignore')
%matplotlib inline
from sklearn.metrics import mean_squared_error
from math import sqrt
    
def rmse(y_actual, y_predicted):
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    return rmse
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
df = pd.read_csv('train.csv', sep =',')
df
df['error']  = np.array(np.linalg.norm(df[['x', 'y', 'z']].values - df[['x_sim', 'y_sim', 'z_sim']].values, axis=1))

temp2 = df.copy()
temp2.sort_values('epoch', axis=0, inplace=True)
temp2['epoch'] = pd.to_datetime(temp2.epoch,format='%Y-%m-%d %H:%M:%S') 
temp2.index = temp2.epoch
temp2 = temp2.drop(columns = 'epoch')

#err = temp2[temp2.type=='train']['error']
#err = err.iloc[-int(len(err)/2):]



sat_id_max =df.sat_id.max()
train_err = temp2[temp2.type == 'train']
test_err = temp2[temp2.type == 'test']
res = []
for sat in tqdm(range(sat_id_max + 1)):
    train = train_err[train_err.sat_id == sat]
    
    train = train.iloc[-int(train.count()[0]/2):]

    test = test_err[test_err.sat_id == sat]
    model = ExponentialSmoothing(np.asarray(train.error) ,seasonal_periods=24 , seasonal='additive').fit()
    forecast = pd.DataFrame(model.forecast(len(test)),index = test.index)
    res.append(forecast.values)
    
res = list(chain.from_iterable(res))
result = pd.DataFrame()
df1 = df[df.type == 'test']
result['id'] = np.array(test_err.id.sort_values().astype(int))
result['error']  = res
result.info()
with open('my_sub.csv', mode='w', encoding='utf-8') as f_csv:
    result.to_csv(f_csv, index=False)
temp = df.copy()
temp = temp[(temp.sat_id == 0)]
temp = temp.drop(columns = ['epoch','y','z','y_sim','z_sim', 'sat_id'])
temp.index = temp.id
temp = temp.drop(columns = 'id')
temp

model = LinearRegression()
train_temp = temp[temp.type == 'train'].drop(columns ='type')
test_temp = temp[temp.type == 'test'].drop(columns ='type')

model.fit(pd.DataFrame(train_temp.x_sim) ,pd.DataFrame(train_temp.x))

forecast = model.predict(test_temp.drop(columns ='x')).reshape(1,-1)
forecast = list(chain.from_iterable(forecast))
forecast = pd.Series(forecast)

#test_temp.x = forecast
test_temp = pd.DataFrame({'x':forecast.values, 'x_sim': test_temp.x_sim}, index = test_temp.index)
test_temp
asd = pd.concat([train_temp,test_temp])
print(temp)
temp[['x','x_sim']] = asd
temp
temp2 = df.copy()
temp2.sort_values('epoch', axis=0, inplace=True)
temp2 = temp2[(temp2.sat_id == 0)]
temp2 = temp2.drop(columns = ['id','y','z','y_sim','z_sim', 'sat_id'])
temp2.index = temp2.epoch
temp2 = temp2.drop(columns = 'epoch')
temp2
from sklearn.model_selection import TimeSeriesSplit 
errors = []
    

#temp1 = temp[temp.type == 'train']
#temp1 = temp1.drop(columns='type')
#temp2 = temp[temp.type == 'train']
#temp2 = temp2.drop(columns='type')
tscv = TimeSeriesSplit(n_splits=4)
for train_idx, test_idx in tscv.split(err):
    #print(train_idx)
    #print(test_idx)
    print('len_train - ', len(train_idx))
    print('len_test - ', len(test_idx))
    model = ExponentialSmoothing(np.asarray(err.iloc[train_idx]) ,seasonal_periods=24 , seasonal='additive').fit()
    forecast = pd.Series(model.forecast(len(test_idx)))
    #model = Holt(np.asarray(temp2.x.iloc[train_idx])).fit(smoothing_level = 0.5,smoothing_slope = 0.5)
    #model = SimpleExpSmoothing(np.asarray(temp2.x.iloc[train_idx])).fit(smoothing_level = 0.5,optimized=False)
    #model = ExponentialSmoothing(np.asarray(temp2.x.iloc[train_idx]) ,seasonal_periods=24 , seasonal='additive').fit()
   
    #forecast = pd.Series(model.forecast(len(test_idx)))
    #model = LinearRegression()
    #train_temp = temp1.iloc[train_idx]
    #test_temp = temp1.iloc[test_idx]

    #model.fit(pd.DataFrame(train_temp.x_sim) ,pd.DataFrame(train_temp.x))
    #forecast = (model.predict(pd.DataFrame(test_temp.x_sim)))
    #forecast = list(chain.from_iterable(forecast))
    #forecast = pd.Series(forecast)
    actual = err.iloc[test_idx]
    error = smape(actual.values, forecast.values)
    errors.append(error)
plt.plot(actual.values)
plt.plot(forecast.values)
errors
temp2 = df.copy()
temp2.sort_values('epoch', axis=0, inplace=True)
temp2.index = temp2.epoch
temp2 = temp2.drop(columns = 'epoch')
temp2
#df1 = df.copy()
df1 = temp2.copy()
coords = ['x','y','z']
model = LinearRegression()
sat_id_max = df.sat_id.max()
train_df = df1[df1.type=='train']
test_df = df1[df1.type == 'test']
for sat in tqdm(range(sat_id_max + 1)):
    train = train_df[train_df.sat_id == sat]
    test = test_df[test_df.sat_id == sat]
    for coord in coords:
        model = ExponentialSmoothing(np.asarray(train[coord]) ,seasonal_periods=24 , seasonal='additive').fit()
        forecast = pd.Series(model.forecast(len(test)))
        #model.fit(pd.DataFrame(train[coord]),pd.DataFrame(train[coord + '_sim']))
        #forecast = model.predict(pd.DataFrame(test[coord + '_sim']))
        #forecast = list(chain.from_iterable(forecast))
        #forecast = pd.Series(forecast)
        test_temp = pd.DataFrame({coord:forecast.values, coord + '_sim': test[coord + '_sim']}, index = test.index)
        test[[coord,coord + '_sim']] = test_temp
    df1.update(test)

actual = test[coord + '_sim']
plt.plot(forecast.values)
plt.plot(actual.values)
df1
result = pd.DataFrame()
df1 = df1[df1.type == 'test']
result['id'] = np.array(df1.id.astype(int))
result['error']  = np.array(np.linalg.norm(df1[['x', 'y', 'z']].values - df1[['x_sim', 'y_sim', 'z_sim']].values, axis=1))
print(result)
with open('my_sub.csv', mode='w', encoding='utf-8') as f_csv:
    result.to_csv(f_csv, index=False)
df2 = pd.read_csv('sub.csv', sep =',')
df2
#print(pd.DataFrame({'x':pd.Series(forecast.values),'x_sim':pd.Series(actual.values)},indices = actual.indices))
plt.plot(temp1)
df.sort_values('epoch', axis=0, inplace=True)
df.epoch = pd.to_datetime(df.epoch, format='%Y-%m-%d %H:%M:%S')
df.index = df.epoch
df.drop('epoch', axis = 1, inplace = True)
df.head(2)
tmp = df.copy()
rcParams['figure.figsize'] = 20, 5
tmp['year'] = tmp.index.year
tmp['month'] = tmp.index.month
tmp['hour'] = tmp.index.hour
tmp['dow'] = tmp.index.dayofweek
tmp_pivot = pd.pivot_table(tmp, values = "x", columns = "year", index = "month")
tmp_pivot.head(3)
df_x = df
df_x = df_x.drop(columns = ['y', 'z', 'y_sim', 'z_sim'])
df_x = df_x[df_x.type == 'train']
df_x_1 = df_x[df_x.sat_id == 0]
df_x_1
df_x_1.drop(columns=['type','sat_id'])
tmp = df_x_1.drop(columns='id')
p, ind  = periodogram(tmp.x, axis = 0)
plt.semilogy(p, ind)
plt.show()
spec = pd.DataFrame(ind,index=p)
top = spec.nlargest(3,columns=0)
time = 1/top.index
time
df[(df.sat_id == 10)].iloc[100:].y.plot(figsize=(15,6),title= 'Координаты спутников', fontsize=14)
#df[df.sat_id == 10].y_sim.plot(figsize=(15,6),title= 'Координаты спутников', fontsize=14)
df[df.sat_id == 10].x.describe()
plt.figure(figsize=(16,2))
mean = df[df.sat_id == 10].x.rolling(window=2400*30*6).mean()
plt.plot(mean)
plt.show()
fit = SimpleExpSmoothing(np.asarray(df[(df.sat_id == 0) & (df.type == 'train')].x)).fit(smoothing_level=1,optimized=False)
fit.forecast(10)
df[(df.sat_id == 0) & (df.type == 'train')].x
df[df.sat_id == 0].nsmallest(4,'x')
seasonality_mins = 18 * 60 + 41
seasonality_id = 24