import pandas as pd

import pandas_datareader.data as web

#import quandl

from datetime import datetime

import warnings

from datetime import timedelta

import numpy as np
start = datetime(2018, 1, 1)

end = datetime.now()
data_input = pd.DataFrame()

data_input['head'] = ['spot','lib_yen','lib_eur','us_Tres','us_goods','for_goods','nasdaq','gold_etf','vix','vixsp','gold' ]

data_input['tag'] = ['DCOILWTICO','JPY3MTD156N','EUR3MTD156N','DLTIIT','DTWEXBGS','DTWEXAFEGS','NASDAQCOM','GVZCLS','VIXCLS','VXVCLS','GOLDPMGBD228NLBM']

print(data_input)
drip = pd.read_csv(r'../input/ext1data/DRIP.csv')

drip['DATE'] = pd.to_datetime(drip['DATE'],format = '%Y-%m-%d')

drip.set_index("DATE", inplace = True)

spdr = pd.read_csv(r'../input/ext1data/spdroil.csv')

spdr['DATE'] = pd.to_datetime(spdr['DATE'],format = '%Y-%m-%d')

spdr.set_index("DATE", inplace = True)

dow_oil = pd.read_excel(r'../input/ext1data/dow_oil.xlsx')

dow_oil['DATE'] = pd.to_datetime(dow_oil['DATE'],format = '%Y-%m-%d')

dow_oil.set_index("DATE", inplace = True)

Crude_oil_WTI = pd.read_csv(r'../input/ext1data/Crude_oil_trend.csv')

Crude_oil_WTI['Date'] = pd.to_datetime(Crude_oil_WTI['Date'],format = '%Y-%m-%d')

Crude_oil_WTI.rename(columns={ 'Date':"DATE" }, inplace = True)

wti_1m = Crude_oil_WTI.copy()

Crude_oil_WTI.rename(columns = {'Price':'sp_price'}, inplace = True)

#Crude_oil_WTI['DATE'] = pd.to_datetime(Crude_oil_WTI['DATE'],format = '%Y-%m-%d')

print(list(Crude_oil_WTI.columns))

print(Crude_oil_WTI.head())

Crude_oil_WTI.set_index("DATE", inplace = True)

Crude_oil_WTI = Crude_oil_WTI[start:end]

Crude_oil_WTI = Crude_oil_WTI.join(drip, how='left')

Crude_oil_WTI = Crude_oil_WTI.join(spdr, how='left')

Crude_oil_WTI = Crude_oil_WTI.join(dow_oil, how='left')

for index, row in data_input.iterrows():

    h=row['head']

    t=row['tag']

    #print(h)

    #print(t)

    fred = pd.DataFrame()

    fred = web.get_data_fred(t, start, end) #3-Month London Interbank Offered Rate (LIBOR), based on Euro (EUR3MTD156N)

    fred = pd.DataFrame(fred)

    fred.rename(columns={ fred.columns[0]:h }, inplace = True)

    fred.reset_index(drop=False, inplace=True)

    fred['DATE'] = pd.to_datetime(fred['DATE'],format = '%Y-%m-%d')

    fred.set_index("DATE", inplace = True)

    #if h=='wti':

    #        fred_df = fred

   #else:

    Crude_oil_WTI = Crude_oil_WTI.join(fred, how='left')

print(Crude_oil_WTI.head())

 
wti = Crude_oil_WTI.copy(deep=True)
corrmat = Crude_oil_WTI.corr()

print(corrmat['sp_price'])
wti['sp_price'].max()
wti['sp_price'].min()
wti = wti["2018-01-01":"2020-07-06"]

wti = wti[wti['sp_price'].notna()]

wti=wti.dropna()

wti.head()

wti_1m.head()
wti.reset_index(drop=False, inplace=True)

wti['DATE'] = pd.to_datetime(wti['DATE'],format = '%Y-%m-%d')

wti['1m_DATE'] = wti['DATE'] + pd.DateOffset(days=51)

wti_1m['DATE'] = pd.to_datetime(wti_1m['DATE'],format = '%Y-%m-%d')

wti_1m.rename(columns={ 'DATE':"DATE_2m" }, inplace = True)

wti = pd.merge(wti,wti_1m,left_on='1m_DATE',right_on='DATE_2m',how='left')

wti.head()
wti.set_index("DATE", inplace = True)

wti = wti.drop(columns='DATE_2m',axis=1)

wti['Price'] = wti['Price'].fillna(method='bfill')

wti.head()
wti.head()
wti.iloc[:, 1:16]
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import Normalizer

from sklearn.preprocessing import QuantileTransformer

from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import MinMaxScaler

Minmax = MinMaxScaler(feature_range=(27,70))



wti.iloc[:, 1:16] = Minmax.fit_transform(wti.iloc[:, 1:16])





#wti.loc[:, wti.columns != {'Price','sp_price','1m_DATE'}] = QuantileTransformer(output_distribution='normal').fit_transform(wti.loc[:, wti.columns != {'Price','sp_price','1m_DATE'}])

#wti.iloc[:, 1:17] = QuantileTransformer(output_distribution='normal').fit_transform(wti.iloc[:, 1:17])

    # PowerTransformer(method='box-cox').fit_transform(X)),

   # ('Data after quantile transformation (gaussian pdf)',

     #   QuantileTransformer(output_distribution='normal')

      #  .fit_transform(X)),
wti = wti.drop(columns='spot',axis=1)

corrmat = wti.corr()

print(corrmat['Price'])
wti


wti_c = wti.drop(columns={'drip_vol','spdr_oil_vol','lib_yen','lib_eur','nasdaq'},axis=1)
wti_c.head()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

col = list(wti_c.columns)

col.remove('1m_DATE')

col.remove('Price')

for i in col:

    # multiple line plot

    plt.plot( wti_c.index, wti_c['Price'], marker='', color='Blue',label="Price")

    plt.plot( wti_c.index, wti_c[i] , marker='', color='red', linewidth=2,label=i)

    plt.legend()

    plt.show()
wti_c = wti_c.drop(columns='gold',axis=1)

wti_c.tail()
#wti_c.to_csv(r'E:\python\kaggle map comp\Ext Data\Ext Data\1Jul2020\wti_c.csv')
train = wti_c[:"2020-04-13"]

test1 = wti_c["2020-04-14":"2020-07-06"]

train.tail()
train.head()
from sklearn.linear_model import LinearRegression
model = LinearRegression()
train.reset_index(drop=False,inplace=True)

test1.reset_index(drop=False,inplace=True)

x_train = train.iloc[:, 1:11]



x_test1 = test1.iloc[:, 1:11]

x_train.head()
y_train = train['Price']



y_test1 = test1['Price']

y_train.head()
model = LinearRegression()

model.fit(x_train, y_train)
pred = model.predict(x_test1)
print(pred)
test1.head()
test1['lin_Pred'] = pred

result = test1.iloc[:,11:14]

result.head()
result.tail()
import xgboost as xgb

from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt 
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(x_train,y_train)
preds = xg_reg.predict(x_test1)
preds

test1['xgb_Pred'] = preds

result = test1.iloc[:,11:15]

result.head()
result.set_index('1m_DATE',drop=False,inplace=True)

rm = result["2020-06-04":"2020-07-02"]

rm.tail()

rmse_xgb = np.sqrt(mean_squared_error(rm['Price'], rm['xgb_Pred']))

print("RMSE: %f" % (rmse_xgb))
rmse_lin= np.sqrt(mean_squared_error(rm['Price'], rm['lin_Pred']))

print("RMSE: %f" % (rmse_lin))

xgb_res = result["2020-07-07":"2020-08-21"]
xgb_res = xgb_res.drop(columns = ['Price','lin_Pred'],axis=1)
xgb_res.reset_index(drop=True,inplace=True)
xgb_res.rename(columns = {'1m_DATE':'Date','xgb_Pred':"Price"}, inplace = True) 

xgb_res['Date'] = pd.to_datetime(xgb_res['Date'],format = '%Y-%m-%d')

xgb_res
Crude_oil_WTI.head()
df = Crude_oil_WTI["2020-04-29":"2020-07-06"]
df = df.loc[:, ['sp_price']]
df

xgb_res.rename(columns = {'DATE':'Date'}, inplace = True) 

df.rename(columns = {'sp_price':"Price"}, inplace = True) 

df.reset_index(drop=False,inplace=True)

df.rename(columns = {'DATE':'Date'}, inplace = True) 
df = df.append(xgb_res,ignore_index=True)

df['Date'] = pd.to_datetime(df['Date'],format = '%Y-%m-%d')
df = df[df["Date"].dt.weekday < 5]

df.head()
df
df = df.append({'Date':'2020-07-03','Price': 27.18413333},ignore_index=True)
df = df.append({'Date':'2020-07-13','Price': 26.92253333},ignore_index=True)

df = df.append({'Date':'2020-07-07','Price': 27.4512},ignore_index=True)

df = df.append({'Date':'2020-07-14','Price': 26.92253333},ignore_index=True)

df = df.append({'Date':'2020-07-15','Price': 26.92253333},ignore_index=True)

df = df.append({'Date':'2020-07-20','Price': 26.50946667},ignore_index=True)

df = df.append({'Date':'2020-07-21','Price': 26.50946667},ignore_index=True)

df = df.append({'Date':'2020-07-27','Price': 27.49241829},ignore_index=True)

df = df.append({'Date':'2020-07-28','Price': 27.49241829},ignore_index=True)

df = df.append({'Date':'2020-08-03','Price': 23.95260429},ignore_index=True)

df = df.append({'Date':'2020-08-04','Price': 23.95260429},ignore_index=True)

df = df.append({'Date':'2020-08-10','Price': 23.95260429},ignore_index=True)

df = df.append({'Date':'2020-08-11','Price': 23.95260429},ignore_index=True)

df = df.append({'Date':'2020-08-17','Price': 23.95260429},ignore_index=True)

df = df.append({'Date':'2020-08-18','Price': 23.95260429},ignore_index=True)

df['Date'] = pd.to_datetime(df['Date'],format = '%Y-%m-%d')
df.to_csv(r'/kaggle/working/submission.csv',index=False)