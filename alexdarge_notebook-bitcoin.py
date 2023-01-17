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
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from scipy import stats
from itertools import product
import warnings

warnings.filterwarnings('ignore')
# chargement des données
df = pd.read_csv('../input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')
df2 = pd.read_csv('../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')
df.head()
# ON remplace les nan par la valeur 0
df['Volume_(BTC)'].fillna(value=0, inplace=True)
df['Volume_(Currency)'].fillna(value=0, inplace=True)
df['Weighted_Price'].fillna(value=0, inplace=True)
df2['Volume_(BTC)'].fillna(value=0, inplace=True)
df2['Volume_(Currency)'].fillna(value=0, inplace=True)
df2['Weighted_Price'].fillna(value=0, inplace=True)

# On remplace les valeur manquantes de open high low close qui sont continues par la valeur précendente
# lets fill forwards those values...
df['Open'].fillna(method='ffill', inplace=True)
df['High'].fillna(method='ffill', inplace=True)
df['Low'].fillna(method='ffill', inplace=True)
df['Close'].fillna(method='ffill', inplace=True)
df2['Open'].fillna(method='ffill', inplace=True)
df2['High'].fillna(method='ffill', inplace=True)
df2['Low'].fillna(method='ffill', inplace=True)
df2['Close'].fillna(method='ffill', inplace=True)

# taille des deux dataframes
print(df.shape)
print(df2.shape)
# conversion timestamp
df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')

# conversion jours
df.index = df.Timestamp
df = df.resample('D').mean()

#conversion mois
df_month = df.resample('M').mean()

#conversion année
df_year = df.resample('A-DEC').mean()
# conversion timestamp
df2.Timestamp = pd.to_datetime(df2.Timestamp, unit='s')

# conversion jours
df2.index = df2.Timestamp
df2 = df2.resample('D').mean()

#conversion mois
df_month2 = df2.resample('M').mean()

#conversion année
df_year2 = df2.resample('A-DEC').mean()
#affichage du df apres conversion de l'horodatage
df.head()
#variation du prix du bitcoin (1er df)
fig = plt.figure(figsize=(20,5))
#variation quotidienne
plt.subplot(131)
plt.plot(df.Weighted_Price, '-', label='Quotidien')
plt.legend()
#variation mensuelle
plt.subplot(132)
plt.plot(df_month.Weighted_Price, '-', label='Mensuel')
plt.legend()
#variation annuelle
plt.subplot(133)
plt.plot(df_year.Weighted_Price, '-', label='Annuel')
plt.legend()
plt.suptitle('Variation des prix du bitcoin (Datframe 1)')
# plt.tight_layout()
plt.show()
# variation quotidienne du prix du bitcoin pour le 1er df
fig = plt.figure(figsize=(20,5))
plt.plot(df.Weighted_Price, '-', label='Quotidien')
plt.legend()
plt.suptitle('Variation quotidienne des prix du bitcoin')
plt.grid(linestyle='dotted')
plt.show()
#variation du prix du bitcoin (2eme df)
fig = plt.figure(figsize=(20,5))
#variation quotidienne
plt.subplot(131)
plt.plot(df2.Weighted_Price, '-', label='Quotidien')
plt.legend()
#variation mensuelle
plt.subplot(132)
plt.plot(df_month2.Weighted_Price, '-', label='Mensuel')
plt.legend()
#variation annuelle
plt.subplot(133)
plt.plot(df_year2.Weighted_Price, '-', label='Annuel')
plt.legend()
plt.suptitle('Variation des prix du bitcoin (Datframe 2)')
# plt.tight_layout()
plt.show()
# variation quotidienne du prix du bitcoin pour le 2eme df
fig = plt.figure(figsize=(20,5))
plt.plot(df2.Weighted_Price, '-', label='Quotidien')
plt.legend()
plt.suptitle('Variation quotidienne des prix du bitcoin')
plt.grid(linestyle='dotted')
plt.show()
plt.figure(figsize=(20,15))
df2.High.plot(kind='line',color='g',label='high',linewidth=1,alpha=0.5,grid=True,linestyle=':')
df2.Low.plot(color='r',label='Low',linewidth=1,alpha=0.5,linestyle='-.',grid=True)
plt.legend('upper right')
plt.suptitle('Bitcoin')
plt.show()
# intervalle de temps concerné df1
df.index.min(), df.index.max()
#intervalle df2
df2.index.min(), df2.index.max()
def Dickey_Fuller_test(timeseries):
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = pd.rolling(timeseries, window=12).mean()
    rolstd = pd.rolling(timeseries, window=12).std()

    #Plot rolling statistics:
    plt.figure(figsize=(15,6))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    Dickey_Fuller_test(timeseries)
#test_stationarity(df.Weighted_Price)
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)
plt.show()
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(df2.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)
plt.show()
# meilleure visualisation de decomposition STL
plt.style.use('seaborn-poster')
#plt.figure(figsize=(20,15))
sm.tsa.seasonal_decompose(df_month.Weighted_Price).plot()
plt.title('decomposition saison du 1er df')
plt.show()
#plt.figure(figsize=(20,15))
sm.tsa.seasonal_decompose(df_month2.Weighted_Price).plot()
plt.title('decomposition saison du 2eme df')
plt.show()
# test de dickey fuller
print("Dickey–Fuller test df1: p=%f" % sm.tsa.stattools.adfuller(df_month.Weighted_Price)[1])
print("Dickey–Fuller test df2: p=%f" % sm.tsa.stattools.adfuller(df_month2.Weighted_Price)[1])
#décomposition STL df1:
print("Dickey–Fuller test df1: p=%f" % sm.tsa.stattools.adfuller(df_month.Weighted_Price)[1])
sm.tsa.seasonal_decompose(df_month.Weighted_Price).plot()
plt.show()
#décomposition STL df2:
print("Dickey–Fuller test df2: p=%f" % sm.tsa.stattools.adfuller(df_month2.Weighted_Price)[1])
sm.tsa.seasonal_decompose(df_month2.Weighted_Price).plot()
plt.show()
# Transformtion Box-Cox 
df_month['Weighted_Price_box'], lmbda = stats.boxcox(df_month.Weighted_Price)
df_month2['Weighted_Price_box'], lmbda = stats.boxcox(df_month2.Weighted_Price)
print("Dickey–Fuller test df1: p=%f" % sm.tsa.stattools.adfuller(df_month.Weighted_Price)[1])
print("Dickey–Fuller test df2: p=%f" % sm.tsa.stattools.adfuller(df_month2.Weighted_Price)[1])
# Differentiation saisonniere
df_month['prices_box_diff'] = df_month.Weighted_Price_box - df_month.Weighted_Price_box.shift(12)
df_month2['prices_box_diff'] = df_month2.Weighted_Price_box - df_month2.Weighted_Price_box.shift(12)

print("Dickey–Fuller test df1: p=%f" % sm.tsa.stattools.adfuller(df_month.prices_box_diff[12:])[1])
print("Dickey–Fuller test df2: p=%f" % sm.tsa.stattools.adfuller(df_month2.prices_box_diff[12:])[1])
# Differentiation df1
df_month['prices_box_diff2'] = df_month.prices_box_diff - df_month.prices_box_diff.shift(1)
# STL
sm.tsa.seasonal_decompose(df_month.prices_box_diff2[13:]).plot()   
print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.prices_box_diff2[13:])[1])
plt.show()
# Differentiation df2
df_month2['prices_box_diff2'] = df_month2.prices_box_diff - df_month2.prices_box_diff.shift(1)
# STL
sm.tsa.seasonal_decompose(df_month2.prices_box_diff2[13:]).plot()   
print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month2.prices_box_diff2[13:])[1])
plt.show()
df
#test modele 
from pandas.plotting import autocorrelation_plot

data = pd.read_csv('../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')
data = data.dropna()
data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')
df_index = data.set_index(['Timestamp'])
df_index = df_index.sort_index(axis=1, ascending=True)
weighted_price_data = df_index['Weighted_Price']
# autocorrelation graphe
mask = (data['Timestamp'] <= '2015-12-31 00:00:00')
sub_df = data.loc[mask]
sub_df = sub_df[['Timestamp', 'Weighted_Price']]
sub_df = sub_df.set_index('Timestamp')
autocorrelation_plot(sub_df)
plt.show()
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
arima_data = data[['Weighted_Price']]

residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
lag = 20
lag_pacf = pacf(df, nlags=lag, method='ols')
lag_acf = acf(df, nlags=lag)
# Approximation des parametres initiaux
Qs = range(0, 2)
qs = range(0, 3)
Ps = range(0, 3)
ps = range(0, 3)
D=1
d=1
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)
# Selection modele
results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')
for param in parameters_list:
    try:
        model=sm.tsa.statespace.SARIMAX(data.Weighted_Price, order=(param[0], d, param[1]), 
                                        seasonal_order=(param[2], D, param[3], 12),enforce_stationarity=False,
                                            enforce_invertibility=False).fit(disp=-1)
    except ValueError:
        #print('wrong parameters:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])