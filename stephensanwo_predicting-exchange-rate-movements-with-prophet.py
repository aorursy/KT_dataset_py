#!pip install fbprophet

#!pip install holidays==0.9.12

#!pip install plotly

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from fbprophet import Prophet
from urllib.request import Request, urlopen



site= "https://www.cbn.gov.ng/Functions/export.asp?tablename=exchange"

hdr = {'User-Agent': 'Mozilla/5.0'}

req = Request(site,headers=hdr)

page = urlopen(req)
cbn_rates = pd.read_csv(page, index_col=False)
cbn_rates
cbn_rates['Rate Date']=pd.to_datetime(cbn_rates['Rate Date'])
cbn_rates.head()
from datetime import datetime, timedelta

five_days_ago = datetime.strftime(datetime.now() - timedelta(7), '%Y-%m-%d')
#USD Cleaning



cbn_usd = cbn_rates[cbn_rates.Currency=='US DOLLAR']

cbn_usd.reset_index(drop=True, inplace=True)



#Try except to adjust for data not available on weekends



try:

    cbn_usd_filtered = cbn_usd.iloc[cbn_usd[cbn_usd['Rate Date']==datetime.strftime(datetime.now() - timedelta(5), '%Y-%m-%d')].index.values[0]:cbn_usd[cbn_usd['Rate Date']==datetime.strftime(datetime.now() - timedelta(5), '%Y-%m-%d')].index.values[0]+365]

except IndexError:

    cbn_usd_filtered = cbn_usd.iloc[cbn_usd[cbn_usd['Rate Date']==datetime.strftime(datetime.now() - timedelta(7), '%Y-%m-%d')].index.values[0]:cbn_usd[cbn_usd['Rate Date']==datetime.strftime(datetime.now() - timedelta(7), '%Y-%m-%d')].index.values[0]+365]

    

cbn_usd_buy = cbn_usd_filtered[['Rate Date','Buying Rate']]

cbn_usd_sell = cbn_usd_filtered[['Rate Date','Selling Rate']]

cbn_usd_buy_p = cbn_usd_buy.rename(columns={'Rate Date' : 'ds', 'Buying Rate': 'y' })

cbn_usd_sell_p = cbn_usd_sell.rename(columns={'Rate Date' : 'ds', 'Selling Rate': 'y' })
#GBP Cleaning



cbn_gbp = cbn_rates[cbn_rates.Currency=='POUNDS STERLING']

cbn_gbp.reset_index(drop=True, inplace=True)

cbn_gbp_filtered = cbn_gbp.iloc[cbn_gbp[cbn_gbp['Rate Date']==five_days_ago].index.values[0]:cbn_gbp[cbn_gbp['Rate Date']==five_days_ago].index.values[0]+365]

cbn_gbp_buy = cbn_gbp_filtered[['Rate Date','Buying Rate']]

cbn_gbp_sell = cbn_gbp_filtered[['Rate Date','Selling Rate']]

cbn_gbp_buy_p = cbn_gbp_buy.rename(columns={'Rate Date' : 'ds', 'Buying Rate': 'y' })

cbn_gbp_sell_p = cbn_gbp_sell.rename(columns={'Rate Date' : 'ds', 'Selling Rate': 'y' })
#EUR Cleaning



cbn_eur = cbn_rates[cbn_rates.Currency=='EURO']

cbn_eur.reset_index(drop=True, inplace=True)

cbn_eur_filtered = cbn_eur.iloc[cbn_eur[cbn_eur['Rate Date']==five_days_ago].index.values[0]:cbn_eur[cbn_eur['Rate Date']==five_days_ago].index.values[0]+365]

cbn_eur_buy = cbn_eur_filtered[['Rate Date','Buying Rate']]

cbn_eur_sell = cbn_eur_filtered[['Rate Date','Selling Rate']]

cbn_eur_buy_p = cbn_eur_buy.rename(columns={'Rate Date' : 'ds', 'Buying Rate': 'y' })

cbn_eur_sell_p = cbn_eur_sell.rename(columns={'Rate Date' : 'ds', 'Selling Rate': 'y' })
#Prediction - USD_Buy

m = Prophet()

m.fit(cbn_usd_buy_p)

future = m.make_future_dataframe(periods = 31)

future.tail()

forecast_usd_buy = m.predict(future)

forecast_usd_buy.tail()
#Prediction - USD_Sell



m1 = Prophet()

m1.fit(cbn_usd_sell_p)



future = m1.make_future_dataframe(periods = 31)

future.tail()



forecast_usd_sell = m1.predict(future)



forecast_usd_sell.tail()
#Prediction - GBP_Buy

m2 = Prophet()

m2.fit(cbn_gbp_buy_p)

future = m2.make_future_dataframe(periods = 31)

future.tail()

forecast_gbp_buy = m2.predict(future)

forecast_gbp_buy.tail()
#Prediction - GBP_Sell



m3 = Prophet()

m3.fit(cbn_gbp_sell_p)



future = m3.make_future_dataframe(periods = 31)

future.tail()



forecast_gbp_sell = m3.predict(future)



forecast_gbp_sell.tail()
#Prediction - EUR_Buy

m4 = Prophet()

m4.fit(cbn_eur_buy_p)

future = m4.make_future_dataframe(periods = 31)

future.tail()

forecast_eur_buy = m4.predict(future)

forecast_eur_buy.tail()
#Prediction - EUR_Sell



m5 = Prophet()

m5.fit(cbn_eur_sell_p)



future = m5.make_future_dataframe(periods = 31)

future.tail()



forecast_eur_sell = m5.predict(future)



forecast_eur_sell.tail()
#Prepare Data for JSON Payload



forecast_usd_buy = forecast_usd_buy[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index(keys = 'ds', drop = True).iloc[-27:]

forecast_usd_sell = forecast_usd_sell[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index(keys = 'ds', drop = True).iloc[-27:]

forecast_gbp_buy = forecast_gbp_buy[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index(keys = 'ds', drop = True).iloc[-27:]

forecast_gbp_sell = forecast_gbp_sell[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index(keys = 'ds', drop = True).iloc[-27:]

forecast_eur_buy = forecast_eur_buy[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index(keys = 'ds', drop = True).iloc[-27:]

forecast_eur_sell = forecast_eur_sell[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index(keys = 'ds', drop = True).iloc[-27:]
forecast_usd_buy.rename(mapper = {'ds':'date', 'yhat':'usd_buy', 'yhat_lower':'usd_buy_lower', 'yhat_upper':'usd_buy_upper'}, axis = 1, inplace = True)

forecast_usd_sell.rename(mapper = {'ds':'date', 'yhat':'usd_sell', 'yhat_lower':'usd_sell_lower', 'yhat_upper':'usd_sell_upper'}, axis = 1, inplace = True)

forecast_gbp_buy.rename(mapper = {'ds':'date', 'yhat':'gbp_buy', 'yhat_lower':'gbp_buy_lower', 'yhat_upper':'gbp_buy_upper'}, axis = 1, inplace = True)

forecast_gbp_sell.rename(mapper = {'ds':'date', 'yhat':'gbp_sell', 'yhat_lower':'gbp_sell_lower', 'yhat_upper':'gbp_sell_upper'}, axis = 1, inplace = True)

forecast_eur_buy.rename(mapper = {'ds':'date', 'yhat':'eur_buy', 'yhat_lower':'eur_buy_lower', 'yhat_upper':'eur_buy_upper'}, axis = 1, inplace = True)

forecast_eur_sell.rename(mapper = {'ds':'date', 'yhat':'eur_sell', 'yhat_lower':'eur_sell_lower', 'yhat_upper':'eur_sell_upper'}, axis = 1, inplace = True)
forecast_usd_buy_T = forecast_usd_buy.transpose()

forecast_usd_sell_T = forecast_usd_sell.transpose()

forecast_gbp_buy_T = forecast_gbp_buy.transpose()

forecast_gbp_sell_T = forecast_gbp_sell.transpose()

forecast_eur_buy_T = forecast_eur_buy.transpose()

forecast_eur_sell_T = forecast_eur_sell.transpose()
#Append app the payloads to consolidate



appended_forecast = forecast_usd_buy_T.append([forecast_usd_sell_T,forecast_gbp_buy_T, forecast_gbp_sell_T,forecast_eur_buy_T ,forecast_eur_sell_T ])
appended_forecast.reset_index(inplace=True)
cbn_predict_30_days = appended_forecast.to_json(date_format = 'iso')
cbn_predict_30_days