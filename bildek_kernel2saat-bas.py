# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) wil list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    

# Any results you write to the current directory are saved as output.

EURHKD = pd.read_csv("../input/eeeeee7/EURHKD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv") 
CADHKD = pd.read_csv("../input/eeeeee7/CADHKD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv") 
AUDSGD = pd.read_csv("../input/eeeeee7/AUDSGD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv") 
USDSGD = pd.read_csv("../input/eeeeee7/USDSGD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
AUDCHF = pd.read_csv("../input/eeeeee7/AUDCHF_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
EURAUD = pd.read_csv("../input/eeeeee7/EURAUD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
AUDCAD = pd.read_csv("../input/eeeeee7/AUDCAD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
AUDJPY = pd.read_csv("../input/eeeeee7/AUDJPY_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
AUDUSD = pd.read_csv("../input/eeeeee7/AUDUSD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
EURUSD = pd.read_csv("../input/eeeeee7/EURUSD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
GBPAUD = pd.read_csv("../input/eeeeee7/GBPAUD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
CHFSGD = pd.read_csv("../input/eeeeee7/CHFSGD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
EURNZD = pd.read_csv("../input/eeeeee7/EURNZD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
EURSGD = pd.read_csv("../input/eeeeee7/EURSGD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
USDCAD = pd.read_csv("../input/eeeeee7/USDCAD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
USDCHF = pd.read_csv("../input/eeeeee7/USDCHF_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
EURCAD = pd.read_csv("../input/eeeeee7/EURCAD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
USDTRY = pd.read_csv("../input/eeeeee7/USDTRY_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
USDJPY = pd.read_csv("../input/eeeeee7/USDJPY_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
AUDNZD = pd.read_csv("../input/eeeeee7/AUDNZD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
GBPUSD = pd.read_csv("../input/eeeeee7/GBPUSD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
GBPCAD = pd.read_csv("../input/eeeeee7/GBPCAD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
GBPNZD = pd.read_csv("../input/eeeeee7/GBPNZD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
CADCHF = pd.read_csv("../input/eeeeee7/CADCHF_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
NZDUSD = pd.read_csv("../input/eeeeee7/NZDUSD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")

XAUUSD = pd.read_csv("../input/eeeeee7/XAUUSD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
XAGUSD = pd.read_csv("../input/eeeeee7/XAGUSD_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
CHFJPY = pd.read_csv("../input/eeeeee7/CHFJPY_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
EURCHF = pd.read_csv("../input/eeeeee7/EURCHF_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
EURGBP = pd.read_csv("../input/eeeeee7/EURGBP_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
EURJPY = pd.read_csv("../input/eeeeee7/EURJPY_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
EURNOK = pd.read_csv("../input/eeeeee7/EURNOK_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
EURTRY = pd.read_csv("../input/eeeeee7/EURTRY_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
GBPCHF = pd.read_csv("../input/eeeeee7/GBPCHF_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
NZDCHF = pd.read_csv("../input/eeeeee7/NZDCHF_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
USDNOK = pd.read_csv("../input/eeeeee7/USDNOK_Candlestick_2_Hour_BID_01.01.2010-08.02.2020.csv")
AUDSGD.tail(20)

EURHKD1 = EURHKD.iloc[:44254,[1]]
CADHKD1 = CADHKD.iloc[:44254,[1]]
AUDSGD1 = AUDSGD.iloc[:44254,[1]]
USDSGD1 = USDSGD.iloc[:44254,[1]]
AUDCHF1 = AUDCHF.iloc[:44254,[1]]
EURAUD1 = EURAUD.iloc[:44254,[1]]
AUDCAD1 = AUDCAD.iloc[:44254,[1]] 
AUDJPY1 = AUDJPY.iloc[:44254,[1]]
AUDUSD1 = AUDUSD.iloc[:44254,[1]]
EURUSD1 = EURUSD.iloc[:44254,[1]]
GBPAUD1 = GBPAUD.iloc[:44254,[1]]
CHFSGD1 = CHFSGD.iloc[:44254,[1]]
EURNZD1 = EURNZD.iloc[:44254,[1]]
EURSGD1 = EURSGD.iloc[:44254,[1]] 
USDCAD1 = USDCAD.iloc[:44254,[1]]
USDCHF1 = USDCHF.iloc[:44254,[1]]
EURCAD1 = EURCAD.iloc[:44254,[1]]
USDTRY1 = USDTRY.iloc[:44254,[1]]
USDJPY1 = USDJPY.iloc[:44254,[1]]
AUDNZD1 = AUDNZD.iloc[:44254,[1]]
GBPUSD1 = GBPUSD.iloc[:44254,[1]] 
GBPCAD1 = GBPCAD.iloc[:44254,[1]]
GBPNZD1 = GBPNZD.iloc[:44254,[1]]
CADCHF1 = CADCHF.iloc[:44254,[1]]
NZDUSD1 = NZDUSD.iloc[:44254,[1]]
XAUUSD1 = XAUUSD.iloc[:44254,[1]]
XAGUSD1 = XAGUSD.iloc[:44254,[1]]
CHFJPY1 = CHFJPY.iloc[:44254,[1]]
EURCHF1 = EURCHF.iloc[:44254,[1]]
EURGBP1 = EURGBP.iloc[:44254,[1]]
EURJPY1 = EURJPY.iloc[:44254,[1]]
EURNOK1 = EURNOK.iloc[:44254,[1]]
EURTRY1 = EURTRY.iloc[:44254,[1]]
GBPCHF1 = GBPCHF.iloc[:44254,[1]]
NZDCHF1 = NZDCHF.iloc[:44254,[1]]
USDNOK1 = USDNOK.iloc[:44254,[1]]

hedef = EURUSD.iloc[:44254,[4]]
AAA = pd.concat([ EURHKD1 ,CADHKD1,AUDSGD1,USDSGD1,AUDCHF1,EURAUD1,AUDCAD1,AUDJPY1,AUDUSD1,
                 EURUSD1,GBPAUD1,CHFSGD1,EURNZD1,EURSGD1,USDCAD1,USDCHF1,EURCAD1,USDTRY1
                 ,USDJPY1,AUDNZD1,GBPUSD1,GBPCAD1,GBPNZD1,CADCHF1,NZDUSD1,
                 XAUUSD1,XAGUSD1,CHFJPY1,EURCHF1,EURGBP1,EURJPY1,EURNOK1,EURTRY1,GBPCHF1,NZDCHF1,USDNOK1 ],axis=1)

AAA.tail()
aaa = AAA.values
hedef = hedef.values
x_train, x_test, y_train, y_test = train_test_split(aaa,hedef,test_size=0.005)


regressor = LinearRegression()
regressor.fit(x_train,y_train)
TAHMİNLER = regressor.predict(x_test)


x_test.shape

y_test.shape
TAHMİNLER.shape
r2_score(y_test,TAHMİNLER)
r2_score(y_test,TAHMİNLER)

plt.plot(y_test[-100:], label="gerçek fiyat")
plt.plot(TAHMİNLER[-100:],label="tahmin lagoritması")
plt.legend()


regressor.predict([[     
    
    9.01176
    ,5.78378
,0.98202
,1.38332
,0.65519
,1.63762
,0.95241
,75.138
,0.70990
,1.16252
,1.80152
,1.49874
,1.75122
,1.60814
,1.34027
,0.92298
,1.56216
,6.84654
,105.794
,1.06938
,1.27887
,1.71403
,1.92647
,0.68864
,0.66384
,1903.182
,22.83577
,114.680
,1.07298
,0.90903
,123.048
,10.67529
,7.96147
,1.17982
,0.61240
,9.18298

    
    
]])

