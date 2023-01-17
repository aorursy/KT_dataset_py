# Ссылка для просмотра коллаба: https://colab.research.google.com/drive/1pymv0u92sTOZlDYn2IIbPd66uJy_eH1i?usp=sharing

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import datetime

import statsmodels.api as sm          # Для предсказания временных рядов. (содержит ARIMA)

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns                 # Для визуализации в 3D.
%matplotlib inline

# Тест Дики-Фуллера:
from statsmodels.tsa.stattools import adfuller

import warnings
warnings.filterwarnings('ignore')     # Чтобы не было лишних предупреждений.
# #Доступ к диску:
# from google.colab import drive
# drive.mount('/content/drive')
# df = pd.read_csv('/content/drive/My Drive/baby_back/train.csv', index_col='id')
df = pd.read_csv('../input/sputnik/train.csv', index_col='id')
df.head()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(33,8))
ax1 = fig.add_subplot(141, projection='3d')
ax2 = fig.add_subplot(142, projection='3d')
ax3 = fig.add_subplot(143, projection='3d')
ax4 = fig.add_subplot(144, projection='3d')

# fig, axes = plt.subplots(1, 4, subplot_kw={'projection':'3d'}, figsize=(33,8)) # Поддерживается только в коллабе.

# for i, id_ in enumerate([27,421,236,0]):
#   sample = df.query('sat_id==@id_ & type=="train"')
#   coords = sample.loc[:,['x','y','z']].to_numpy()
#   axes[i].scatter(coords[:,0], coords[:,1], coords[:,2])


coords = df.query('sat_id==27 & type=="train"').loc[:,['x','y','z']].to_numpy()
ax1.scatter(coords[:,0], coords[:,1], coords[:,2])

coords = df.query('sat_id==421 & type=="train"').loc[:,['x','y','z']].to_numpy()
ax2.scatter(coords[:,0], coords[:,1], coords[:,2])

coords = df.query('sat_id==236 & type=="train"').loc[:,['x','y','z']].to_numpy()
ax3.scatter(coords[:,0], coords[:,1], coords[:,2])

coords = df.query('sat_id==0 & type=="train"').loc[:,['x','y','z']].to_numpy()
ax4.scatter(coords[:,0], coords[:,1], coords[:,2])

plt.show()
data = df.copy()
for id_ in tqdm(range(600)):

  tmp_df = data.query('sat_id==@id_ & type=="train"').loc[:,['x','y','z']]
  xyz_center = (tmp_df.max(axis=0)/2 + tmp_df.min(axis=0)/2).to_numpy()
  data.loc[tmp_df.index,['x','y','z']] -= xyz_center

  tmp_df = data.query('sat_id==@id_').loc[:,['x_sim','y_sim','z_sim']]
  data.loc[tmp_df.index,['x_sim','y_sim','z_sim']] -= xyz_center
for id_ in tqdm(range(600)):

  tmp_df = data.query('sat_id==@id_')
  
  data.loc[tmp_df.index,'r'] = (tmp_df.x**2 + tmp_df.y**2 + tmp_df.z**2)**0.5
  data.loc[tmp_df.index,'r_sim'] = (tmp_df.x_sim**2 + tmp_df.y_sim**2 + tmp_df.z_sim**2)**0.5

  data.loc[tmp_df.index,'e'] = np.arctan(((tmp_df.x**2+tmp_df.y**2)**0.5)/tmp_df.z)
  data.loc[tmp_df.index,'e_sim'] = np.arctan(((tmp_df.x_sim**2+tmp_df.y_sim**2)**0.5)/tmp_df.z_sim)

  data.loc[tmp_df.index,'f'] = np.arctan(tmp_df.y/tmp_df.x)
  data.loc[tmp_df.index,'f_sim'] = np.arctan(tmp_df.y_sim/tmp_df.x_sim)
fig, axes = plt.subplots(2,3, figsize=(20,8))
id_ = 12

tmp_df = data.query('sat_id==@id_')

axes[0,0].plot(tmp_df.x)
axes[0,0].set_title('x')
axes[0,1].plot(tmp_df.y)
axes[0,1].set_title('y')
axes[0,2].plot(tmp_df.z)
axes[0,2].set_title('z')

axes[1,0].plot(tmp_df.r)
axes[1,0].set_title('r')
axes[1,1].plot(tmp_df.f)
axes[1,1].set_title('f')
axes[1,2].plot(tmp_df.e)
axes[1,2].set_title('e')

plt.show()
plt.close()
# Функция расчета расстояние между точками в сферической с.координат.
def distance_sphere(r1,e1,f1, r2,e2,f2):

  r1, r2 = np.asarray(r1), np.asarray(r2)
  e1, e2 = np.asarray(e1), np.asarray(e2)
  f1, f2 = np.asarray(f1), np.asarray(f2)

  x1, x2 = r1*np.sin(e1)*np.cos(f1), r2*np.sin(e2)*np.cos(f2)
  y1, y2 = r1*np.sin(e1)*np.sin(f1), r2*np.sin(e2)*np.sin(f2)
  z1, z2 = r1*np.cos(e1), r2*np.cos(e2)

  return ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5
# Функция расчета расстояние между точками в декартовой с.координат.
def distance_decard(x1,y1,z1, x2,y2,z2):

  x1, x2 = np.asarray(x1), np.asarray(x2)
  y1, y2 = np.asarray(y1), np.asarray(y2)
  z1, z2 = np.asarray(z1), np.asarray(z2)

  return ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5
# Добавляем столбец с расстояниями.
for id_ in tqdm(range(600)):
  tmp_df = data.query('sat_id==@id_')
  data.loc[tmp_df.index,'error'] = distance_decard(tmp_df.x,
                                                   tmp_df.y, 
                                                   tmp_df.z, 
                                                   tmp_df.x_sim, 
                                                   tmp_df.y_sim, 
                                                   tmp_df.z_sim)
data.head()
# data.drop(['x','y','z','x_sim','y_sim','z_sim'], axis=1, inplace=True)
#  Тут отдается дань Ким Г.Динховне за знание линала!
data_copy = data.copy()
# data = data_copy.copy()
# Подключаем все библиотеки для работы с временными рядами:
from statsmodels.tsa.api import ExponentialSmoothing
import statsmodels.api as sm

# Постоянный размер графиков, дальше их будет много)
plt.rcParams['figure.figsize'] = 12, 7 
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.svm import SVR

# Подключим все нужные метрики:
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error # Для RMSE, см. ниже!
from math import sqrt                          # Для RMSE, см. ниже!

# Mean absolute percentage error:
def mape(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Symmetric mean absolute percentage error:
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def rmse(y_actual, y_predicted):
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    return rmse
# Для прогноза с лагом t-1, t-2, t-3,...

class BoostingSeriesForecaster():

  def __init__(self, model):
    self.regressor = model

  def fit(self, Series, begin=1, end=25, step=1):
    """
    Series - ряд, который нужно научиться предсказывать.
    type: pd.Series/pd.DataFrame
    """
    self.data = pd.DataFrame(Series.copy())
    for log_ in range(begin,end,step):
      self.data['log_'+str(log_)] = self.data.iloc[:,0].shift(log_)

    self.data = self.data.dropna().to_numpy()


  def predict(self, N): # Переписать на Numpy.
    """
    N - кол-во предсказанных вперед значений.
    """

    try:
      for iter_ in range(N):
        
          self.regressor.fit(self.data[:,1:], self.data[:,0])
          test_sample = self.data[-1][:-1].reshape(1,-1) # Возможно нужно убрать reshape!
          current_pred = self.regressor.predict(test_sample)[0] # Возможно нужно убрать [0]!
          new_line = np.hstack((current_pred,self.data[-1][:-1]))
          self.data = np.vstack((self.data,new_line))

    except:
      # print("EXCEPTION")
      return np.ones(N)*99999

    return self.data[-N:,0]
# Для прогноза с лагом t-S, t-2S, ... (По дефолту стоит t-S)

class BoostingSeriesForecasterS():

  def __init__(self, model, W=24):
    self.model = BoostingSeriesForecaster(model)
    self.W = W


  def fit(self, Series, N):
    """
    Series - ряд, который нужно научиться предсказывать.
    type: pd.Series/pd.DataFrame
    N - кол-во предсказанных вперед значений.
    """

    self.Series = pd.DataFrame(Series.copy()).reset_index(drop=True)
    self.N = N

    for first_nan in range(self.Series.shape[0]):
      tmp = self.Series.iloc[first_nan,-1]
      if tmp - tmp != 0:
        break

    for _ in range(N): # Переписать на Numpy.
      # print(self.Series.iloc[-self.W::-self.W].iloc[::-1].reset_index(drop=True))
      self.model.fit(self.Series.iloc[-self.W::-self.W].iloc[::-1].reset_index(drop=True), 1, 2)
      self.Series.loc[self.Series.shape[0],:] = self.model.predict(1)[0]
    

  def predict(self):    
    return self.Series.iloc[-self.N:].to_numpy().reshape(-1)

  def predict_with_train(self):    
    return self.Series.to_numpy().reshape(-1)
# Функция возвращает:
# True - если гипотеза подтверждается.
# False - если отвергается в пользу альтернативной.
# (В итоге не использовалась.)
def test_stationarity(timeseries, window = 24, cutoff = 0.05):
    return adfuller(timeseries.values, autolag='AIC')[1] < cutoff
lag_period = 24

mse_history, mape_history, smape_history = [], [], []

mse_history_1,   mse_history_2,   mse_history_3,   mse_history_4,   mse_history_5   = [], [], [], [], []
mape_history_1,  mape_history_2,  mape_history_3,  mape_history_4,  mape_history_5  = [], [], [], [], []
smape_history_1, smape_history_2, smape_history_3, smape_history_4, smape_history_5 = [], [], [], [], []

id_to_model_dict = dict()

for id_ in tqdm(range(0,600)):

  preds_1, preds_2, preds_3, preds_4,  preds_5 = dict(), dict(), dict(), dict(), dict()
  tmp_df = data.query('sat_id==@id_ & type=="train"')

  ratio = tmp_df.shape[0] / data.query('sat_id==@id_').shape[0]
  th = int(len(tmp_df)*ratio)

  train = tmp_df.iloc[:th]
  val = tmp_df.iloc[th:]

  for coor in ['x','y','z']:

    mdl_1,   mdl_3,   mdl_5   = LinearRegression(), LinearRegression(), KNN()
    model_1, model_3, model_5 = BoostingSeriesForecasterS(mdl_1, 24), BoostingSeriesForecaster(mdl_3),  BoostingSeriesForecaster(mdl_5)
    model_1.fit(train.loc[:,coor], val.shape[0])
    model_3.fit(train.loc[:,coor])
    model_5.fit(train.loc[:,coor])
    forecast_1, forecast_3, forecast_5 = model_1.predict(), model_3.predict(val.shape[0]), model_5.predict(val.shape[0])
    preds_1.update({coor: forecast_1})
    preds_3.update({coor: forecast_3})
    preds_5.update({coor: forecast_5})


  for coor in ['r','f','e']:

    mdl_2,   mdl_4   = KNN(), LinearRegression()
    model_2, model_4 = BoostingSeriesForecasterS(mdl_2, 24), BoostingSeriesForecaster(mdl_4)    
    model_2.fit(train.loc[:,coor], val.shape[0])
    model_4.fit(train.loc[:,coor])
    forecast_2, forecast_4 = model_2.predict(), model_4.predict(val.shape[0])
    preds_2.update({coor: forecast_2})
    preds_4.update({coor: forecast_4})

  x_sim, y_sim, z_sim = val.x_sim,  val.y_sim,  val.z_sim
  r_sim, f_sim, e_sim = val.r_sim,  val.f_sim,  val.e_sim

  pred_err_1 = distance_decard(preds_1['x'], preds_1['y'], preds_1['z'], x_sim, y_sim, z_sim)
  pred_err_2 = distance_sphere(preds_2['r'], preds_2['f'], preds_2['e'], r_sim, f_sim, e_sim)
  pred_err_3 = distance_decard(preds_3['x'], preds_3['y'], preds_3['z'], x_sim, y_sim, z_sim)  
  pred_err_4 = distance_sphere(preds_4['r'], preds_4['f'], preds_4['e'], r_sim, f_sim, e_sim)
  pred_err_5 = distance_decard(preds_5['x'], preds_5['y'], preds_5['z'], x_sim, y_sim, z_sim)  

  x, y, z = val.x, val.y, val.z 
  r, f, e = val.r, val.f, val.e

  val_err = distance_decard(x, y, z, x_sim, y_sim, z_sim)

  mse_history_1.append(rmse(pred_err_1,    val_err))
  mape_history_1.append(mape(pred_err_1,   val_err))
  smape_history_1.append(smape(pred_err_1, val_err))

  mse_history_2.append(rmse(pred_err_2,    val_err))
  mape_history_2.append(mape(pred_err_2,   val_err))
  smape_history_2.append(smape(pred_err_2, val_err))

  mse_history_3.append(rmse(pred_err_3,    val_err))
  mape_history_3.append(mape(pred_err_3,   val_err))
  smape_history_3.append(smape(pred_err_3, val_err))  

  mse_history_4.append(rmse(pred_err_4,    val_err))
  mape_history_4.append(mape(pred_err_4,   val_err))
  smape_history_4.append(smape(pred_err_4, val_err))

  mse_history_5.append(rmse(pred_err_5,    val_err))
  mape_history_5.append(mape(pred_err_5,   val_err))
  smape_history_5.append(smape(pred_err_5, val_err))

  all_smapes = np.array([smape_history_1[-1], smape_history_2[-1], smape_history_3[-1], smape_history_4[-1], smape_history_5[-1]])
  best_model_num = np.argmin(all_smapes)
  best_smape     = np.min(all_smapes)
  id_to_model_dict.update({id_: [best_model_num, best_smape]})
  smape_history.append(best_smape)

  print('id - ', id_, ' SMAPE: %.4f' % np.mean(smape_history), '    best model:', best_model_num, '   all smape: \t%.4f, \t%.4f, \t%.4f, \t%.4f, \t%.4f' % (all_smapes[0], all_smapes[1], all_smapes[2], all_smapes[3], all_smapes[4])) 
def change_name(float_num):

  if   float_num == 0.0:
    return 'Lin_SFS_dec'
  elif float_num == 1.0:
    return 'KNN_SFS_sph'
  elif float_num == 2.0:
    return 'Lin_SF_dec'
  elif float_num == 3.0:
    return 'Lin_SF_sph'
  elif float_num == 4.0:
    return 'KNN_SF_dec'
  else:
    raise ValueError
id_to_model = pd.DataFrame(id_to_model_dict).T
id_to_model.rename({0: 'model_type', 1: 'smape_score'}, axis=1, inplace=True)
id_to_model.index.name = 'sat_id'
id_to_model.iloc[:,0] = id_to_model.iloc[:,0].apply(change_name)
id_to_model.to_csv('best_model_per_id.csv')
id_to_model.head()
# Посмотрим на подробный обзор характеристик моделей:

id_to_model.groupby('model_type').describe()
# Удостоверяемся что все правильно сохранили:

pd.read_csv('best_model_per_id.csv', index_col='sat_id').groupby('model_type').describe()
np.median(id_to_model.smape_score), np.mean(id_to_model.smape_score)
plt.plot(sorted(id_to_model.smape_score.to_numpy()))
plt.plot([20]*600)
mask = id_to_model.smape_score.to_numpy() < 7  
mask = id_to_model.smape_score.to_numpy() < 40 

id_to_model.smape_score.to_numpy()[mask].mean()
id_to_model[mask].groupby('model_type').describe()
id_to_model[~mask].groupby('model_type').describe()
id_to_model_name_dict = id_to_model.to_dict()['model_type']
def get_best_smape(id_, data, forecast_size):

  global id_to_model_dict

  model_name = id_to_model_name_dict[id_]
  preds = dict()

  th = data.shape[0] - forecast_size

  train = tmp_df.iloc[:th]
  val = tmp_df.iloc[th:]

  if  model_name == 'Lin_SFS_dec':
    for coor in ['x','y','z']:
      mdl = LinearRegression()
      model = BoostingSeriesForecasterS(mdl, 24)
      model.fit(train.loc[:,coor], forecast_size)
      forecast = model.predict()
      preds.update({coor: forecast})
    x_sim, y_sim, z_sim = val.x_sim,  val.y_sim,  val.z_sim
    pred_err = distance_decard(preds['x'], preds['y'], preds['z'], x_sim, y_sim, z_sim)
    x, y, z = val.x, val.y, val.z 
    val_err = distance_decard(x, y, z, x_sim, y_sim, z_sim)

  elif model_name == 'Lin_SF_dec':
    for coor in ['x','y','z']:
      mdl = LinearRegression()
      model = BoostingSeriesForecaster(mdl)
      model.fit(train.loc[:,coor])
      forecast = model.predict(val.shape[0])
      preds.update({coor: forecast})
    x_sim, y_sim, z_sim = val.x_sim,  val.y_sim,  val.z_sim
    pred_err = distance_decard(preds['x'], preds['y'], preds['z'], x_sim, y_sim, z_sim)
    x, y, z = val.x, val.y, val.z 
    val_err = distance_decard(x, y, z, x_sim, y_sim, z_sim)

  elif model_name == 'Lin_SF_sph':
    for coor in ['r','f','e']:
      mdl = LinearRegression()
      model = BoostingSeriesForecaster(mdl)    
      model.fit(train.loc[:,coor])
      forecast = model.predict(val.shape[0])
      preds.update({coor: forecast})  
    r_sim, f_sim, e_sim = val.r_sim,  val.f_sim,  val.e_sim
    pred_err = distance_sphere(preds['r'], preds['f'], preds['e'], r_sim, f_sim, e_sim)
    r, f, e = val.r, val.f, val.e
    val_err = distance_sphere(r, f, e, r_sim, f_sim, e_sim)

  elif model_name == 'KNN_SFS_sph':
    for coor in ['r','f','e']:
      mdl = KNN()
      model = BoostingSeriesForecasterS(mdl, 24)
      model.fit(train.loc[:,coor], val.shape[0])
      forecast = model.predict()
      preds.update({coor: forecast})
    r_sim, f_sim, e_sim = val.r_sim,  val.f_sim,  val.e_sim
    pred_err = distance_sphere(preds['r'], preds['f'], preds['e'], r_sim, f_sim, e_sim)
    r, f, e = val.r, val.f, val.e
    val_err = distance_sphere(r, f, e, r_sim, f_sim, e_sim)

  elif model_name == 'KNN_SF_dec':
    for coor in ['x','y','z']:
      mdl = KNN()
      model = BoostingSeriesForecaster(mdl)
      model.fit(train.loc[:,coor])
      forecast = model.predict(val.shape[0])
      preds.update({coor: forecast})   
    x_sim, y_sim, z_sim = val.x_sim,  val.y_sim,  val.z_sim  
    pred_err = distance_decard(preds['x'], preds['y'], preds['z'], x_sim, y_sim, z_sim)      
    x, y, z = val.x, val.y, val.z 
    val_err = distance_decard(x, y, z, x_sim, y_sim, z_sim)

  else:
    raise ValueError

  return smape(pred_err, val_err), model_name
d = data.copy()
id_to_model_name_dict
def predict_error(id_, data):

  global id_to_model_dict

  model_name = id_to_model_name_dict[id_]
  preds = dict()

  train = data.query('type=="train"')
  val   = data.query('type=="test"')
  forecast_size = val.shape[0]

  if  model_name == 'Lin_SFS_dec':
    for coor in ['x','y','z']:
      mdl = LinearRegression()
      model = BoostingSeriesForecasterS(mdl, 24)
      model.fit(train.loc[:,coor], forecast_size)
      forecast = model.predict()
      preds.update({coor: forecast})
    x_sim, y_sim, z_sim = val.x_sim,  val.y_sim,  val.z_sim
    pred_err = distance_decard(preds['x'], preds['y'], preds['z'], x_sim, y_sim, z_sim)
    
  elif model_name == 'Lin_SF_dec':
    for coor in ['x','y','z']:
      mdl = LinearRegression()
      model = BoostingSeriesForecaster(mdl)
      model.fit(train.loc[:,coor])
      forecast = model.predict(val.shape[0])
      preds.update({coor: forecast})
    x_sim, y_sim, z_sim = val.x_sim,  val.y_sim,  val.z_sim
    pred_err = distance_decard(preds['x'], preds['y'], preds['z'], x_sim, y_sim, z_sim)
    
  elif model_name == 'Lin_SF_sph':
    for coor in ['r','f','e']:
      mdl = LinearRegression()
      model = BoostingSeriesForecaster(mdl)    
      model.fit(train.loc[:,coor])
      forecast = model.predict(val.shape[0])
      preds.update({coor: forecast})  
    r_sim, f_sim, e_sim = val.r_sim,  val.f_sim,  val.e_sim
    pred_err = distance_sphere(preds['r'], preds['f'], preds['e'], r_sim, f_sim, e_sim)
    
  elif model_name == 'KNN_SFS_sph':
    for coor in ['r','f','e']:
      mdl = KNN()
      model = BoostingSeriesForecasterS(mdl, 24)
      model.fit(train.loc[:,coor], val.shape[0])
      forecast = model.predict()
      preds.update({coor: forecast})
    r_sim, f_sim, e_sim = val.r_sim,  val.f_sim,  val.e_sim
    pred_err = distance_sphere(preds['r'], preds['f'], preds['e'], r_sim, f_sim, e_sim)

  elif model_name == 'KNN_SF_dec':
    for coor in ['x','y','z']:
      mdl = KNN()
      model = BoostingSeriesForecaster(mdl)
      model.fit(train.loc[:,coor])
      forecast = model.predict(val.shape[0])
      preds.update({coor: forecast})   
    x_sim, y_sim, z_sim = val.x_sim,  val.y_sim,  val.z_sim  
    pred_err = distance_decard(preds['x'], preds['y'], preds['z'], x_sim, y_sim, z_sim)   
  else:
    raise ValueError

  return pred_err
final_dict_model
for id_ in tqdm(range(0,600)):
  tmp_df = data.query('sat_id==@id_')

  if   final_dict_model[id_]:
    model = ExponentialSmoothing(np.asarray(tmp_df.query('type=="train"').error)[-24:], seasonal_periods=24, trend=None, seasonal='add').fit()
    stupid_predict = model.forecast(tmp_df.query('type=="test"').shape[0])
    data.loc[tmp_df.query('type=="test"').index,'error'] = stupid_predict
  
  else:
    data.loc[tmp_df.query('type=="test"').index,'error'] = predict_error(id_, tmp_df)
dd = data.copy()
# data = dd.copy()
data.isnull().sum()
ans_df = data.copy()
ans_df = ans_df.loc[:,['sat_id','type','error']]
ans_df
solution = ans_df.query('type=="test"').drop(['sat_id', 'type'], axis=1)
solution.to_csv('solution.csv')