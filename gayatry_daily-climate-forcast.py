import numpy as np

import pandas as pd

from pandas import Series,DataFrame



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams['figure.figsize'] = (10,20)

plt.style.use('ggplot')
train_df = pd.read_csv('/kaggle/input/daily-climate-time-series-data/DailyDelhiClimateTrain.csv',parse_dates=['date'],index_col='date')

train_df.head()
train_df.plot(subplots=True)
from statsmodels.tsa.stattools import grangercausalitytests
variables = train_df.columns

maxlag = 12

test = 'ssr_chi2test'



cause = DataFrame(np.zeros((len(variables),len(variables))),columns=variables,index=variables)

for c in variables:

    for r in variables:

        x = grangercausalitytests(train_df[[r,c]],maxlag=maxlag,verbose=False)

        p_values = [round(x[i+1][0][test][1],5) for i in range(maxlag)]

        min_value = np.min(p_values)

        cause.loc[r,c] = min_value



cause
from statsmodels.tsa.vector_ar.vecm import coint_johansen
coint = coint_johansen(train_df,-1,12)
"""Trail Static"""

coint.lr1.astype(int)
"""Critical trail values"""

coint.cvt
"""Eigen Static"""

coint.lr2.astype(int)
"""Eigen Critical values"""

coint.cvm
from statsmodels.tsa.seasonal import seasonal_decompose
for variable in variables:

    decomposed = seasonal_decompose(train_df[variable])

    x = decomposed.plot(seasonal=False,resid=False)

    
from statsmodels.tsa.stattools import adfuller
print('significance level : 0.05')

for variable in variables:

    adf = adfuller(train_df[variable])

    print(f'For {variable}')

    print(f'Test static {adf[1]}',end='\n \n')
from statsmodels.tsa.vector_ar.var_model import VAR
"""Training AR Model"""

model = VAR(train_df)
for i in [1,2,3,4,5,6,7,8,9]:

    result = model.fit(i)

    print(f'lag_order {i}')

    print(f'AIC : {result.aic}')

    #print(f'BIC : {result.bic}')
""" Training the model with lag_order 6"""

model_fitted = model.fit(6)
"""Creating train-test dataset"""

qwer = train_df.dropna()

lag_order = model_fitted.k_ar

X = qwer[:-lag_order]

Y = qwer[-lag_order:]
from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)

for col,val in (zip(variables,out)):

    print(col, ':',  val)
y = X.values[-lag_order:]

forcast = model_fitted.forecast(y,steps=lag_order)
df_forcast = DataFrame(forcast,index=train_df.index[-lag_order:],columns=Y.columns)

df_forcast
import math

from math import sqrt
from sklearn.metrics import mean_squared_error
for i in train_df.columns:

    print(f'RMSE of {i} is {sqrt(mean_squared_error(Y[[i]],df_forcast[[i]]))}')
from sklearn.metrics import mean_absolute_error
for i in train_df.columns:

    print(f'MAE of {i} is {mean_absolute_error(Y[[i]],df_forcast[[i]])}')
test_df = pd.read_csv('/kaggle/input/daily-climate-time-series-data/DailyDelhiClimateTest.csv',parse_dates=['date'],index_col='date')

test_df.head()
""" Forcasting the next 6 periods"""

date_range = pd.date_range('2017-01-05',periods=6)



lag_order = model_fitted.k_ar

X1,Y1 = test_df[1:-lag_order],test_df[-lag_order:]

input_values = Y1.values[-lag_order:]

forcast1 = model_fitted.forecast(input_values,steps=lag_order)

forcast_df1 = DataFrame(forcast1,columns=X1.columns,index=date_range)

forcast_df1