import pandas as pd 
import datetime
import time
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from pylab import rcParams
import warnings
from pandas.core.nanops import nanmean as pd_nanmean
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.metrics import mean_absolute_error
import collections
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller
#from fbprophet import Prophet
#from fbprophet.plot import plot_plotly
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from datetime import timedelta
from itertools import product
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from pandas import read_csv, DataFrame
import statsmodels.api as sm
from statsmodels.iolib.table import SimpleTable
from sklearn.metrics import r2_score
import ml_metrics as metrics
from statsmodels.iolib.table import SimpleTable
warnings.filterwarnings('ignore')
%matplotlib inline
satelite_pos = pd.read_csv('/kaggle/input/sputnik/train.csv')
satelite_pos.info()
satelite_pos
#satelite_pos['epoch'].replace({'T':' '}, inplace=True)
satelite_pos.sort_values(by=['sat_id','epoch'],axis = 0,inplace =True)
satelite_pos['epoch'] = pd.to_datetime(satelite_pos.epoch,format='%Y-%m-%d %H:%M:%S') 
satelite_pos.index  = satelite_pos.epoch
satelite_pos.drop('epoch', axis = 1, inplace = True)
satelite_pos.head()
satelite_pos['error']  = np.linalg.norm(satelite_pos[['x', 'y', 'z']].values - satelite_pos[['x_sim', 'y_sim', 'z_sim']].values, axis=1)
satelite_pos.info()
sep_dat = []
for i in range(600):
    l = satelite_pos.loc[satelite_pos['sat_id'] == i]
    sep_dat.append(l)
y= []
for i in sep_dat:
    i['hour' ]=i.index.hour
    i['dow' ]=i.index.dayofweek
    i['month' ]=i.index.month
    i['year' ]=i.index.year
    i['minute'] = i.index.minute
    i['seconds'] = i.index.second
    i['dayofweek'] = i.index.dayofweek
    i['quarter'] = i.index.quarter
    i['weekofyear'] = i.index.weekofyear
    y.append(i['error'].dropna().values)
predi = []
j = 0
for i in sep_dat:
    linreg = LinearRegression()
    linreg.fit(i.dropna().drop(['sat_id', 'x','y','z','type','id','error'],axis = 1).values,y[j])
    predi.append(linreg.predict(i.loc[i['type'] == 'test'].drop(['sat_id', 'x','y','z','type','id','error'],axis = 1).values))
    j = j+1
out=open('submit_xgb.csv','w', encoding='utf-8')
for i in predi:
    for j in i:
        out.writelines(str(j)+'\n')
out.close()
