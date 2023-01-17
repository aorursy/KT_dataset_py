import numpy as np

import pandas as pd

import os

import time

import datetime

import requests

import urllib

from urllib.request import urlopen

import json
def GetAPI(Size,Symbol,Period):

    url = [r'https://api.huobi.pro/market/history/kline?period=',str(Period),r'&size=',str(Size),'&symbol=',str(Symbol)]

    url = ''.join(url)

    return url
def GetCryptocurrencyPriceBySize(Period,Size,Symbol,headers,timelag):

    url = GetAPI(Size = Size , Symbol = Symbol, Period = Period)

    Json_Response = requests.get(url,headers = headers)

    Json_Text = Json_Response.json()

    Title = 'data'

    Json_DataInformation = Json_Text[Title]

    Open_Price, High_Price, Low_Price, Close_Price ,Amount,Time_List,Count,Volume= [], [], [], [],[],[],[],[]

    for row in Json_DataInformation:

        Open_Price.append(row['open'])

        High_Price.append(row['high'])

        Low_Price.append(row['low'])

        Close_Price.append(row['close'])

        Amount.append(row['amount'])

        Count.append(row['count'])

        Volume.append(row['vol'])

        Temporary = time.localtime(row['id'])

        Tyear,Tmonth,Tday,Thour,Tmin,Tsecond = Temporary.tm_year,Temporary.tm_mon,Temporary.tm_mday,Temporary.tm_hour,Temporary.tm_min,Temporary.tm_sec

        TargetTime = datetime.datetime(Tyear, Tmonth, Tday, Thour, Tmin, Tsecond) + datetime.timedelta(hours=timelag)

        Time_List.append(TargetTime.strftime("%Y-%m-%d %H:%M:%S"))

    DataInformation = pd.DataFrame([Time_List,Open_Price, High_Price, Low_Price, Close_Price,Volume,Count,Amount]).T

    Columns = ['Time','open', 'high', 'low', 'close','volume','count','amount']

    DataInformation.columns = Columns

    DataInformation.sort_values(by=['Time'], ascending=True, inplace=True)

    DataInformation.reset_index(inplace = True)

    del DataInformation['index']

    return DataInformation
def GetAllSymbol(headers):

    url = 'https://api.huobi.pro/v1/common/symbols'

    Json_Response = requests.get(url,headers = headers)

    Json_Text = Json_Response.json()

    Title = 'data'

    Json_DataInformation = Json_Text[Title]

    AllSymbolName = []

    for row in Json_DataInformation:

        AllSymbolName.append(''.join([row['base-currency'],row['quote-currency']]))

    AllSymbolName.sort()

    for x in AllSymbolName:

        print(x)
headers = {'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Mobile Safari/537.36'}



GetAllSymbol(headers = headers)
Size = 2000

Symbol = 'btcusdt'

#headers = {'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Mobile Safari/537.36'}

timelag = 8



Period_5Min = '5min'

Period_1Min = '1min'

Period_1day ='1day'



DataInformation_5Min = GetCryptocurrencyPriceBySize(Period = Period_5Min, Size = Size, Symbol = Symbol , headers = headers , timelag = timelag)

DataInformation_1Min = GetCryptocurrencyPriceBySize(Period = Period_1Min, Size = Size, Symbol = Symbol , headers = headers , timelag = timelag)

DataInformation_1Day = GetCryptocurrencyPriceBySize(Period = Period_1day, Size = Size, Symbol = Symbol , headers = headers , timelag = timelag)
DataInformation_5Min
DataInformation_1Min
DataInformation_1Day