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
!pip install alpha_vantage

from alpha_vantage.timeseries import TimeSeries

from alpha_vantage.techindicators import TechIndicators

from alpha_vantage.sectorperformance import SectorPerformances

from alpha_vantage.cryptocurrencies import CryptoCurrencies
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

plt.style.use('bmh')
'''

close_data = data['4. close']

percentage_change = close_data.pct_change()



print(percentage_change)

'''



'''



close_data = data['4. close']

percentage_change = close_data.pct_change()



#print(percentage_change)





last_change = percentage_change[-1]



if abs(last_change) > 0.0004:

    print("GOOGL Alert:" + str(last_change)'

'''
#!pip install yahoo-finance
#!pip install yfinance --upgrade --no-cache-dir
#!pip install fix-yahoo-finance
from datetime import datetime
'''

stockSymbols = ['MMM\n',

 'ABT\n',

 'ABBV\n',

 'ABMD\n',

 'ACN\n',

 'ATVI\n',

 'ADBE\n',

 'AMD\n',

 'AAP\n',

 'AES\n',

 'AFL\n',

 'A\n',

 'APD\n',

 'AKAM\n',

 'ALK\n',

 'ALB\n',

 'ARE\n',

 'ALXN\n',

 'ALGN\n',

 'ALLE\n',

 'ADS\n',

 'LNT\n',

 'ALL\n',

 'GOOGL\n',

 'GOOG\n',

 'MO\n',

 'AMZN\n',

 'AMCR\n',

 'AEE\n',

 'AAL\n',

 'AEP\n',

 'AXP\n',

 'AIG\n',

 'AMT\n',

 'AWK\n',

 'AMP\n',

 'ABC\n',

 'AME\n',

 'AMGN\n',

 'APH\n',

 'ADI\n',

 'ANSS\n',

 'ANTM\n',

 'AON\n',

 'AOS\n',

 'APA\n',

 'AIV\n',

 'AAPL\n',

 'AMAT\n',

 'APTV\n',

 'ADM\n',

 'ANET\n',

 'AJG\n',

 'AIZ\n',

 'T\n',

 'ATO\n',

 'ADSK\n',

 'ADP\n',

 'AZO\n',

 'AVB\n',

 'AVY\n',

 'BKR\n',

 'BLL\n',

 'BAC\n',

 'BK\n',

 'BAX\n',

 'BDX\n',

 'BRK.B\n',

 'BBY\n',

 'BIIB\n',

 'BLK\n',

 'BA\n',

 'BKNG\n',

 'BWA\n',

 'BXP\n',

 'BSX\n',

 'BMY\n',

 'AVGO\n',

 'BR\n',

 'BF.B\n',

 'CHRW\n',

 'COG\n',

 'CDNS\n',

 'CPB\n',

 'COF\n',

 'CAH\n',

 'KMX\n',

 'CCL\n',

 'CARR\n',

 'CAT\n',

 'CBOE\n',

 'CBRE\n',

 'CDW\n',

 'CE\n',

 'CNC\n',

 'CNP\n',

 'CTL\n',

 'CERN\n',

 'CF\n',

 'SCHW\n',

 'CHTR\n',

 'CVX\n',

 'CMG\n',

 'CB\n',

 'CHD\n',

 'CI\n',

 'CINF\n',

 'CTAS\n',

 'CSCO\n',

 'C\n',

 'CFG\n',

 'CTXS\n',

 'CLX\n',

 'CME\n',

 'CMS\n',

 'KO\n',

 'CTSH\n',

 'CL\n',

 'CMCSA\n',

 'CMA\n',

 'CAG\n',

 'CXO\n',

 'COP\n',

 'ED\n',

 'STZ\n',

 'COO\n',

 'CPRT\n',

 'GLW\n',

 'CTVA\n',

 'COST\n',

 'COTY\n',

 'CCI\n',

 'CSX\n',

 'CMI\n',

 'CVS\n',

 'DHI\n',

 'DHR\n',

 'DRI\n',

 'DVA\n',

 'DE\n',

 'DAL\n',

 'XRAY\n',

 'DVN\n',

 'DXCM\n',

 'FANG\n',

 'DLR\n',

 'DFS\n',

 'DISCA\n',

 'DISCK\n',

 'DISH\n',

 'DG\n',

 'DLTR\n',

 'D\n',

 'DPZ\n',

 'DOV\n',

 'DOW\n',

 'DTE\n',

 'DUK\n',

 'DRE\n',

 'DD\n',

 'DXC\n',

 'ETFC\n',

 'EMN\n',

 'ETN\n',

 'EBAY\n',

 'ECL\n',

 'EIX\n',

 'EW\n',

 'EA\n',

 'EMR\n',

 'ETR\n',

 'EOG\n',

 'EFX\n',

 'EQIX\n',

 'EQR\n',

 'ESS\n',

 'EL\n',

 'EVRG\n',

 'ES\n',

 'RE\n',

 'EXC\n',

 'EXPE\n',

 'EXPD\n',

 'EXR\n',

 'XOM\n',

 'FFIV\n',

 'FB\n',

 'FAST\n',

 'FRT\n',

 'FDX\n',

 'FIS\n',

 'FITB\n',

 'FE\n',

 'FRC\n',

 'FISV\n',

 'FLT\n',

 'FLIR\n',

 'FLS\n',

 'FMC\n',

 'F\n',

 'FTNT\n',

 'FTV\n',

 'FBHS\n',

 'FOXA\n',

 'FOX\n',

 'BEN\n',

 'FCX\n',

 'GPS\n',

 'GRMN\n',

 'IT\n',

 'GD\n',

 'GE\n',

 'GIS\n',

 'GM\n',

 'GPC\n',

 'GILD\n',

 'GL\n',

 'GPN\n',

 'GS\n',

 'GWW\n',

 'HRB\n',

 'HAL\n',

 'HBI\n',

 'HOG\n',

 'HIG\n',

 'HAS\n',

 'HCA\n',

 'PEAK\n',

 'HSIC\n',

 'HSY\n',

 'HES\n',

 'HPE\n',

 'HLT\n',

 'HFC\n',

 'HOLX\n',

 'HD\n',

 'HON\n',

 'HRL\n',

 'HST\n',

 'HWM\n',

 'HPQ\n',

 'HUM\n',

 'HBAN\n',

 'HII\n',

 'IEX\n',

 'IDXX\n',

 'INFO\n',

 'ITW\n',

 'ILMN\n',

 'INCY\n',

 'IR\n',

 'INTC\n',

 'ICE\n',

 'IBM\n',

 'IP\n',

 'IPG\n',

 'IFF\n',

 'INTU\n',

 'ISRG\n',

 'IVZ\n',

 'IPGP\n',

 'IQV\n',

 'IRM\n',

 'JKHY\n',

 'J\n',

 'JBHT\n',

 'SJM\n',

 'JNJ\n',

 'JCI\n',

 'JPM\n',

 'JNPR\n',

 'KSU\n',

 'K\n',

 'KEY\n',

 'KEYS\n',

 'KMB\n',

 'KIM\n',

 'KMI\n',

 'KLAC\n',

 'KSS\n',

 'KHC\n',

 'KR\n',

 'LB\n',

 'LHX\n',

 'LH\n',

 'LRCX\n',

 'LW\n',

 'LVS\n',

 'LEG\n',

 'LDOS\n',

 'LEN\n',

 'LLY\n',

 'LNC\n',

 'LIN\n',

 'LYV\n',

 'LKQ\n',

 'LMT\n',

 'L\n',

 'LOW\n',

 'LYB\n',

 'MTB\n',

 'MRO\n',

 'MPC\n',

 'MKTX\n',

 'MAR\n',

 'MMC\n',

 'MLM\n',

 'MAS\n',

 'MA\n',

 'MKC\n',

 'MXIM\n',

 'MCD\n',

 'MCK\n',

 'MDT\n',

 'MRK\n',

 'MET\n',

 'MTD\n',

 'MGM\n',

 'MCHP\n',

 'MU\n',

 'MSFT\n',

 'MAA\n',

 'MHK\n',

 'TAP\n',

 'MDLZ\n',

 'MNST\n',

 'MCO\n',

 'MS\n',

 'MOS\n',

 'MSI\n',

 'MSCI\n',

 'MYL\n',

 'NDAQ\n',

 'NOV\n',

 'NTAP\n',

 'NFLX\n',

 'NWL\n',

 'NEM\n',

 'NWSA\n',

 'NWS\n',

 'NEE\n',

 'NLSN\n',

 'NKE\n',

 'NI\n',

 'NBL\n',

 'JWN\n',

 'NSC\n',

 'NTRS\n',

 'NOC\n',

 'NLOK\n',

 'NCLH\n',

 'NRG\n',

 'NUE\n',

 'NVDA\n',

 'NVR\n',

 'ORLY\n',

 'OXY\n',

 'ODFL\n',

 'OMC\n',

 'OKE\n',

 'ORCL\n',

 'OTIS\n',

 'PCAR\n',

 'PKG\n',

 'PH\n',

 'PAYX\n',

 'PAYC\n',

 'PYPL\n',

 'PNR\n',

 'PBCT\n',

 'PEP\n',

 'PKI\n',

 'PRGO\n',

 'PFE\n',

 'PM\n',

 'PSX\n',

 'PNW\n',

 'PXD\n',

 'PNC\n',

 'PPG\n',

 'PPL\n',

 'PFG\n',

 'PG\n',

 'PGR\n',

 'PLD\n',

 'PRU\n',

 'PEG\n',

 'PSA\n',

 'PHM\n',

 'PVH\n',

 'QRVO\n',

 'PWR\n',

 'QCOM\n',

 'DGX\n',

 'RL\n',

 'RJF\n',

 'RTX\n',

 'O\n',

 'REG\n',

 'REGN\n',

 'RF\n',

 'RSG\n',

 'RMD\n',

 'RHI\n',

 'ROK\n',

 'ROL\n',

 'ROP\n',

 'ROST\n',

 'RCL\n',

 'SPGI\n',

 'CRM\n',

 'SBAC\n',

 'SLB\n',

 'STX\n',

 'SEE\n',

 'SRE\n',

 'NOW\n',

 'SHW\n',

 'SPG\n',

 'SWKS\n',

 'SLG\n',

 'SNA\n',

 'SO\n',

 'LUV\n',

 'SWK\n',

 'SBUX\n',

 'STT\n',

 'STE\n',

 'SYK\n',

 'SIVB\n',

 'SYF\n',

 'SNPS\n',

 'SYY\n',

 'TMUS\n',

 'TROW\n',

 'TTWO\n',

 'TPR\n',

 'TGT\n',

 'TEL\n',

 'FTI\n',

 'TFX\n',

 'TXN\n',

 'TXT\n',

 'TMO\n',

 'TIF\n',

 'TJX\n',

 'TSCO\n',

 'TT\n',

 'TDG\n',

 'TRV\n',

 'TFC\n',

 'TWTR\n',

 'TSN\n',

 'UDR\n',

 'ULTA\n',

 'USB\n',

 'UAA\n',

 'UA\n',

 'UNP\n',

 'UAL\n',

 'UNH\n',

 'UPS\n',

 'URI\n',

 'UHS\n',

 'UNM\n',

 'VFC\n',

 'VLO\n',

 'VAR\n',

 'VTR\n',

 'VRSN\n',

 'VRSK\n',

 'VZ\n',

 'VRTX\n',

 'VIAC\n',

 'V\n',

 'VNO\n',

 'VMC\n',

 'WRB\n',

 'WAB\n',

 'WMT\n',

 'WBA\n',

 'DIS\n',

 'WM\n',

 'WAT\n',

 'WEC\n',

 'WFC\n',

 'WELL\n',

 'WST\n',

 'WDC\n',

 'WU\n',

 'WRK\n',

 'WY\n',

 'WHR\n',

 'WMB\n',

 'WLTW\n',

 'WYNN\n',

 'XEL\n',

 'XRX\n',

 'XLNX\n',

 'XYL\n',

 'YUM\n',

 'ZBRA\n',

 'ZBH\n',

 'ZION\n',

 'ZTS\n''

'''
stockStartDate = '2010-3-14'
today = datetime.today().strftime('%y-%m-%d')
numAssets = len(stockSymbols)
def getMyPortfolio (stocks = stockSymbols, start = '2010-3-14', end = '2019-3-14', col = 'Adj Close'):

    data = web.DataReader(stocks,data_source = 'yahoo', start = start, end = end)[col]

    return data
'''

my_stocks = getMyPortfolio(stockSymbols)

my_stocks.to_csv('s&p 500.csv')

'''
import math

import pandas_datareader as web

from sklearn.preprocessing import MinMaxScaler

#from tensorflow.keras import Sequential

from keras.models import Sequential

from keras.layers import Dense,LSTM

#from keras.layers import 

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
#stock_code = input('Enter Stock Code : ')

stock_code = 'AAPL'
start = '2000-01-01'

end = datetime.today().strftime('%y-%m-%d')
df = web.DataReader(stock_code,data_source = 'yahoo', start = '2000-01-01', end = '2019-05-07')

df.tail(1)
plt.figure(figsize = (16,8))

plt.title('Close price History')

plt.plot(df['Close'])

plt.xlabel('Date')

plt.ylabel('Close price USD')

plt.show()
from alpha_vantage.timeseries import TimeSeries

import matplotlib.pyplot as plt



ts_intraday = TimeSeries(key='GVH8MT18HCNA4PC4M',output_format='pandas')

data_intraday, meta_data_intraday = ts_intraday.get_intraday(symbol=stock_code,interval='1min', outputsize='full')

#print(data_intraday)



data_intraday['4. close'].plot()

plt.title('Intraday TimeSeries of Stock')

plt.show()




'''

from alpha_vantage.techindicators import TechIndicators





api_key = 'GVH8MT18HCNA4PC4M'



ts_sma = TimeSeries(key=api_key, output_format='pandas')

data_ts, meta_data_ts = ts_sma.get_intraday(symbol=stock_code,interval='1min', outputsize='full')



periods = 60



ti = TechIndicators(key=api_key, output_format='pandas')

data_ti, meta_data_ti = ti.get_sma(symbol=stock_code, interval='1min', 

                         time_period=60, series_type='close')





df1 = data_ti

df2 = data_ts['4. close'].iloc[periods-1::]



#print(df1)

df2.index = df1.index

#print(df2)



total_df = pd.concat([df1,df2], axis=1)

#print(total_df)



total_df.plot()

plt.title('Simple Moving avarage vs Close')

plt.show()

'''

api_key = 'GVH8MT18HCNA4PC4M'
ti_ema = TechIndicators(key=api_key, output_format='pandas')

data_ema, meta_data_ema = ti_ema.get_ema(symbol=stock_code, interval='1min', 

                         time_period=60, series_type='close')



data_ema.plot()

plt.title('Exponential Moving Average (60 Minutes) of Stock')

plt.show()
ti_stoachastic = TechIndicators(key=api_key, output_format='pandas')

data_stoachastic, meta_data_stoachastic = ti_stoachastic.get_stoch(symbol=stock_code, interval='1min')

data_stoachastic.plot()

plt.title('Stochastic Oscillator')

plt.show()
ti_wma = TechIndicators(key=api_key, output_format='pandas')

data_wma, meta_data_wma = ti_wma.get_wma(symbol=stock_code, interval='1min', time_period=60, series_type='close')

data_wma.plot()

plt.title('Weighted Moving Average')

plt.show()
ti_rsi = TechIndicators(key=api_key, output_format='pandas')

data_rsi, meta_data_rsi = ti_rsi.get_rsi(symbol=stock_code, interval='1min', time_period=60, series_type='close')

data_rsi.plot()

plt.title('Relative Strength Index')

plt.show()


'''

periods = 60



ti_smavsrsi = TechIndicators(key=api_key, output_format='pandas')

data_smavsrsi, meta_data_smavsrsi = ti_smavsrsi.get_rsi(symbol='MSFT', interval='1min',

                         time_period=periods, series_type='close')



tismavsrsi1 = TechIndicators(key=api_key, output_format='pandas')

data_smavsrsi1, meta_data_smavsrsi1 = tismavsrsi1.get_sma(symbol='MSFT', interval='1min',

                         time_period=periods, series_type='close')





df1 = data_smavsrsi1.iloc[1::]

df2 = data_smavsrsi



#print(df1)

df1.index = df2.index

#print(df2)



fig, ax1 = plt.subplots()

ax1.plot(df1, 'b-')

ax2 = ax1.twinx()

ax2.plot(df2, 'r.')

plt.title('SMA (In blue) vs RSI (Red)')

plt.show()



'''


'''

from alpha_vantage.timeseries import TimeSeries

from alpha_vantage.techindicators import TechIndicators



ts_rsivsclose = TimeSeries(key=api_key, output_format='pandas')

data_ts_rsivsclose, meta_data_ts_rsivsclose = ts_rsivsclose.get_intraday(symbol=stock_code,interval='1min', outputsize='full')



periods = 14



ti_rsivsclose1 = TechIndicators(key=api_key, output_format='pandas')

data_ti_rsivsclose1, meta_data_ti_rsivsclose1 = ti_rsivsclose1.get_rsi(symbol=stock_code, interval='1min',

                         time_period=periods, series_type='close')





df1 = data_ti

df2 = data_ts['4. close'].iloc[periods::]



#print(df1)

#df1.index = df2.index

#print(df2)



fig, ax1 = plt.subplots()

ax1.plot(df1, 'b-')

ax2 = ax1.twinx()

ax2.plot(df2, 'r.')

plt.title('RSI vs Close Price')

plt.show()



'''
'''

from alpha_vantage.sectorperformance import SectorPerformances

import matplotlib.pyplot as plt



sp1 = SectorPerformances(key='GVH8MT18HCNA4PC4', output_format='pandas')

data1, meta_data = sp.get_sector()

data1['Rank A: Real-Time Performance'].plot(kind='bar')

plt.title('Real Time Performance (%) per Sector')

plt.tight_layout()

plt.grid()

plt.show()



'''
data = df.filter(['Close'])

dataset = data.values

training_data_len = math.ceil(len(dataset)*.8)

training_data_len
scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(dataset)

scaled_data
train_data = scaled_data[0:training_data_len,:]

x_train = []

y_train = []



for i in range(60,len(train_data)):

    x_train.append(train_data[i-60:i,0])

    y_train.append(train_data[i,0])

    if i <= 60:

        print(x_train)

        print(y_train)

        print()

    
x_train,y_train = np.array(x_train),np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

x_train.shape
import math

import pandas_datareader as web

from sklearn.preprocessing import MinMaxScaler

#from tensorflow.keras import Sequential

from keras.models import Sequential

from keras.layers import Dense,LSTM

#from keras.layers import LSTM

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
model = Sequential()

model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1],1)))

model.add(LSTM(50, return_sequences = False))

model.add(Dense(25))

model.add(Dense(1))
model.compile(optimizer = 'adam',loss = 'mean_squared_error')

model.fit(x_train,y_train,batch_size = 1,epochs = 1)
test_data = scaled_data[training_data_len - 60:,:]

x_test = []

y_test = dataset[training_data_len:,:]

for i in range(60, len(test_data)):

    x_test.append(test_data[i-60:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1)) 
predictions = model.predict(x_test)

predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)**2)

rmse
train = data[:training_data_len]

valid = data[training_data_len:]

valid['Predictions'] = predictions





plt.figure(figsize = (16,18))

plt.title('Model')

plt.xlabel('Date')

plt.ylabel('Close price USD')

plt.plot(train['Close'])

plt.plot(valid[['Close','Predictions']])

plt.legend(['Train','Val','Predictions'],loc= 'lower right')

plt.show()
valid
stock_qoute = web.DataReader(stock_code,data_source = 'yahoo', start = '2000-01-01', end = '2019-05-07')

new_df = stock_qoute.filter(['Close'])

last_60_days = new_df[-60:].values

last_60_days_scaled = scaler.transform(last_60_days)

X_test = []

X_test.append(last_60_days_scaled)

X_test = np.array(X_test)

X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

pred_price = model.predict(X_test)

pred_price = scaler.inverse_transform(pred_price)

print('Predicted price of the stock for the current date is : ',pred_price)

print('Predicted price of the stock for the current date is : ',pred_price)
stock_actual_price1 = web.DataReader(stock_code,data_source = 'yahoo', start = '2000-01-01', end = '2019-05-07')

print('Actual price of the stock is :',stock_actual_price1['Close'].tail(1))