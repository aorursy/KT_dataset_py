import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost

from xgboost import XGBRegressor

    

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Set work directory

os.chdir("../input")



# Load data set

data = pd.read_excel('Base de datos prueba tecnica.xlsx', header=3)

data.columns
bid_df = data[['Dates', 'Bid']].dropna().set_index('Dates')

ask_df = data[['Dates.1', 'Ask']].set_index('Dates.1')
dates = pd.DataFrame(pd.concat([data['Dates'], data['Dates.1']], sort=True).unique(), columns=['Dates']).dropna()
df_consol = pd.concat([dates.set_index('Dates'), bid_df, ask_df], axis=1)

df_consol.isna().sum()
df_consol.fillna(method='ffill', inplace=True)

df_consol.isna().sum()
df_consol.head(6)
def resample_data(frequency):

    data_High = df_consol.resample(frequency, closed='right', label='right').max().dropna()

    data_High.columns = ['High_Bid', 'High_Ask']

    data_Low = df_consol.resample(frequency, closed='right', label='right').min().dropna()

    data_Low.columns = ['Low_Bid', 'Low_Ask']

    data_Close = df_consol.resample(frequency, closed='right', label='right').last().dropna()

    data_Close.columns = ['Close_Bid', 'Close_Ask']

    dataT = pd.concat([data_High, data_Low, data_Close], axis=1)

    dataT['MidPrice'] = dataT[['Close_Bid', 'Close_Ask']].mean(axis=1)

    return dataT
data5 = resample_data('5T')

data20 = resample_data('20T')

data50 = resample_data('50T')
data5.tail()
data20.head()
data50.head()
data5['MA20'] = data5['MidPrice'].rolling(window=4).mean() # 20 Minutos

data5['MA50'] = data5['MidPrice'].rolling(window=10).mean() # 50 Minutos

data5['MA90'] = data5['MidPrice'].rolling(window=18).mean() # 90 Minutos
data5['EWMA90'] = data5['MidPrice'].ewm(ignore_na=False,span=90,min_periods=0,adjust=True).mean()

data5[['MidPrice','MA90', 'EWMA90']].plot()

plt.show()
data5[['MidPrice','MA90', 'EWMA90']][len(data5[data5.index.month <= 10]):].plot()

plt.show()
#First take the sign taking differences of MidPrice Close with EWMA90

midP_px = data5['MidPrice'] 

ewma_px = data5['EWMA90']

signal_px = midP_px - ewma_px

trade_pos = signal_px.apply(np.sign)
trade_pos = trade_pos.shift(1)
##Plot results

fig = plt.figure()

ax = fig.add_subplot(2,1,1)

ax.plot(midP_px.index, midP_px, label='Price')

ax.plot(ewma_px.index, ewma_px, label = 'EWMA90')



ax.set_ylabel('$')

ax.legend(loc='best')

ax.grid()



ax = fig.add_subplot(2,1,2)

ax.plot(trade_pos.index, trade_pos, label='Trading position')

ax.set_ylabel('Trading position')

plt.show()
##Calculate the log returns of asset

log_ret = np.log(midP_px).diff()

r_s = trade_pos * log_ret #El trade post es la señal que dice uno o menos uno

print (r_s.tail())
## Calculate the cumulative log returns

cum_ret = r_s.cumsum()

nuest_cum_ret = log_ret.cumsum()

# And relative returns

cum_rel_ret = np.exp(cum_ret) - 1

nuest_cum_rel_ret = np.exp(nuest_cum_ret) - 1
fig = plt.figure()

ax = fig.add_subplot(2,1,1)



ax.plot(cum_ret.index, cum_ret, label='Momentum')

ax.plot(nuest_cum_ret.index, nuest_cum_ret, label='buy and hold')



ax.set_ylabel('Cumulative log-returns')

ax.legend(loc='best')

ax.grid()



ax = fig.add_subplot(2,1,2)

ax.plot(cum_rel_ret.index, 100*cum_rel_ret, label='Momentum')

ax.plot(nuest_cum_rel_ret.index, 100*nuest_cum_rel_ret, label='buy and hold')



ax.set_ylabel('Total relative returns (%)')

ax.legend(loc='best')

ax.grid()

plt.show()
data_Total = df_consol.resample('5T', closed='right', label='right').ohlc().dropna()
data_Total.head()
Bid_Data = data_Total.Bid

Bid_Data.head()
Bid_Data['avg_price'] = (Bid_Data['low'] + Bid_Data['high'])/2

Bid_Data['range'] = Bid_Data['high'] - Bid_Data['low']

data_Mean = df_consol[['Bid']].resample('5T', closed='right', label='right').mean().dropna()

data_Mean.columns = ['Mean_Bid']



dataBid = pd.concat([Bid_Data, data_Mean], axis=1)
dataBid.isna().sum()
plt.plot(dataBid.close)
dataBid.shape
pd.DataFrame(dataBid.index.month).describe()
Train = dataBid[:len(dataBid[dataBid.index.month <= 9])]

Validate = dataBid[len(dataBid[dataBid.index.month <= 9]):]
#Features & Target

X_T = Train.drop('close', axis=1)

Y_T = Train[['close']]



X_V = Validate.drop('close', axis=1)

Y_V = Validate[['close']]
# Scale the data

min_max_scaler = MinMaxScaler(feature_range=(0, 1))

X_T2 = min_max_scaler.fit_transform(X_T)

X_V2 = min_max_scaler.fit_transform(X_V)



Y_T2 = min_max_scaler.fit_transform(Y_T)

Y_V2 = min_max_scaler.fit_transform(Y_V)
X_T.columns
Y_T.columns
xgb = XGBRegressor()



xgb_param_grid = {'learning_rate': [0.001, 0.01, 0.1, 1],

                  'n_estimators': [250, 300, 350],

                  'subsample': [0.3, 0.5, 0.7, 1]}



grid_search = GridSearchCV(estimator=xgb,    

                           param_grid=xgb_param_grid,

                           scoring='neg_mean_squared_error', 

                           cv=4, 

                           verbose=1,

                           n_jobs=-1)



grid_search.fit(X_T2, Y_T2) 



print("Best parameters: ",grid_search.best_params_)



xgb_best = grid_search.best_estimator_



xgb_best.fit(X_T2, Y_T2)

pred_xgb = xgb_best.predict(X_V2)



print('RMSE: {0:.3f}'.format(mean_squared_error(Y_V2, pred_xgb)**0.5))

print('MAE: {0:.3f}'.format(mean_absolute_error(Y_V2, pred_xgb)))

print('R^2: {0:.3f}'.format(r2_score(Y_V2, pred_xgb)))
dates = X_V.index.values

plt.figure(figsize = (18,9))

plot_truth, = plt.plot(Y_V)

plot_xgb, = plt.plot(dates, min_max_scaler.inverse_transform(pred_xgb.reshape(-1,1)))

plt.legend([plot_truth, plot_xgb], ['Truth', 'xgb'])

plt.title('Prediction vs Truth')

plt.show()
X_T.columns
from xgboost import plot_importance



# plot feature importance

plot_importance(xgb_best).set_yticklabels(X_T.columns)

plt.show()