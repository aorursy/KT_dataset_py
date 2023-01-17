### Get all the libraries

import datetime as dt                    ## To work with date time

import pandas as pd                      ## Work with datasets

import pandas_datareader.data as web     ## Get data from web

import numpy as np                       ## Linear Algebra

import plotly.express as px              ## Graphing/Plotting- Visualization

import plotly.graph_objects as go        ## Graphing/Plotting- Visualization

pd.set_option('display.max_columns', 50) ## Display All Columns

import warnings

warnings.filterwarnings("ignore")       ## Mute all the warnings
## Names of companies on NIFTY index

company_list = ['ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS',

                'BHARTIARTL.NS', 'INFRATEL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'EICHERMOT.NS',

                'GAIL.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS',

                'HDFC.NS', 'ICICIBANK.NS', 'ITC.NS', 'IOC.NS', 'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS','KOTAKBANK.NS',

                'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS',

                'SHREECEM.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 

                'TITAN.NS', 'UPL.NS', 'ULTRACEMCO.NS', 'VEDL.NS', 'WIPRO.NS', 'ZEEL.NS']

print(company_list)
start = dt.datetime(2010, 1, 1) ## Start Date of data

end = dt.datetime(2020, 5, 5)  ## End Date of data



target = 'BAJAJFINSV.NS'

df = web.DataReader(target, 'yahoo', start, end)

columns = df.columns

index = df.index



df.head()
## It's safe to have a copy of the dataset and work on that.

Bj_fs = df.copy()

Bj_fs = Bj_fs[['Close']]

Bj_fs['10_ma'] = Bj_fs['Close'].rolling(10).mean()  ## 10 days moving average.

Bj_fs['30_ma'] = Bj_fs['Close'].rolling(30).mean()  ## 30 days moving average.





fig = go.Figure(data=[

    go.Scatter(x = Bj_fs.index, y=Bj_fs.Close, name='Close', fillcolor='blue'),

    go.Scatter(x = Bj_fs.index, y=Bj_fs['10_ma'], name='10_ma', fillcolor='red'),

    go.Scatter(x = Bj_fs.index, y=Bj_fs['30_ma'], name='30ma', fillcolor='green'),

])

fig.update_layout(title="Moving Average 10 and Moving Average 30 with Close",

                 xaxis_title='Date', yaxis_title='Value')

fig.show()
Bj_fs.reset_index(inplace=True)
## -1: share bught :: 1 = share sold

## Strategy:: if intersection goes down: Sell; if intersection goes up: Buy



temp = [0]

Shares=[0]

for i in Bj_fs.index[1:]:

    if (Bj_fs.iloc[i-1, 2]<=Bj_fs.iloc[i-1, 3]) and (Bj_fs.iloc[i, 2]>Bj_fs.iloc[i, 3]):

        temp.append(-1)

        Shares.append(Shares[-1]+1)

    elif (Bj_fs.iloc[i-1, 2]>=Bj_fs.iloc[i-1, 3]) and (Bj_fs.iloc[i, 2]<Bj_fs.iloc[i, 3]):

        if Shares[-1]>0:

            temp.append(Shares[-1])

            Shares.append(0)

        else:

            temp.append(0)

            Shares.append(0)

    else:

        temp.append(0)

        Shares.append(Shares[-1])

temp = np.array(temp)

Shares = np.array(Shares)
Bj_fs['Direction'] = temp

Bj_fs['Cur_Shares'] = Shares

Bj_fs.dropna(inplace=True) ## Time when we don't have moving avergae of

Bj_fs['Transection'] = Bj_fs['Direction'] * Bj_fs['Close'] ## Bought or sold shares

Bj_fs['Profits'] = Bj_fs['Transection'].cumsum() ## Track of pfofit

Bj_fs.set_index('Date', inplace=True)
fig = go.Figure(data=[

    go.Scatter(x = Bj_fs.index, y=Bj_fs['Close'], name="Daily Close"),

    go.Scatter(x = Bj_fs.index, y=Bj_fs['10_ma'], name="10 MA"),

    go.Scatter(x = Bj_fs.index, y=Bj_fs['30_ma'], name = "30 MA"),

    go.Scatter(x = Bj_fs.index, y=Bj_fs['Transection'], name = "Transections"),

    go.Scatter(x = Bj_fs.index, y=Bj_fs['Profits'], name="Total Profit")

])

fig.update_layout(title = "Trading History:",

                 xaxis_title="Date_Time", yaxis_title="Value")

fig.show()

print("Total Profit Till Day:", Bj_fs['Profits'][-1])
## Transection history

bought = Bj_fs.loc[Bj_fs['Direction']==-1]

sold = Bj_fs.loc[Bj_fs['Direction']==1]

trans = pd.concat([bought, sold]).sort_index()

trans
investment = abs( trans['Profits'].iloc[0])

inflation = 0.055



fd_return = investment * (1+0.06)**10

trading_profit = trans['Profits'].iloc[-1]



cur_value = investment * (1-inflation)**10



print("Current value of our money if we woulld have just saved it:", cur_value)

print("Total value of our money if we woulld have deposited in bank:", fd_return)

print("Total value of our money with trading:", trading_profit)
target = 'BHARTIARTL.NS'

df = web.DataReader(target, 'yahoo', start, end)

columns = df.columns

index = df.index



airtel = df.copy()

airtel = airtel[['Close']]

airtel['10_ma'] = airtel['Close'].rolling(10).mean()

airtel['30_ma'] = airtel['Close'].rolling(30).mean()





fig = go.Figure(data=[

    go.Scatter(x = airtel.index, y=airtel.Close, name='Close'),

    go.Scatter(x = airtel.index, y=airtel['10_ma'], name='10_ma'),

    go.Scatter(x = airtel.index, y=airtel['30_ma'], name='30ma'),

])

fig.update_layout(title="Moving Average 10 and Moving Average 30 with Close",

                 xaxis_title='Date', yaxis_title='Value')

fig.show()







airtel.reset_index(inplace=True)



## -1: share bught :: 1 = share sold

## Strategy:: if intersection goes down: Sell; if intersection goes up: Buy



temp = [0]

Shares=[0]

for i in airtel.index[1:]:

    if (airtel.iloc[i-1, 2]<=airtel.iloc[i-1, 3]) and (airtel.iloc[i, 2]>airtel.iloc[i, 3]):

        temp.append(-1)

        Shares.append(Shares[-1]+1)

    elif (airtel.iloc[i-1, 2]>=airtel.iloc[i-1, 3]) and (airtel.iloc[i, 2]<airtel.iloc[i, 3]):

        if Shares[-1]>0:

            temp.append(Shares[-1])

            Shares.append(0)

        else:

            temp.append(0)

            Shares.append(0)

    else:

        temp.append(0)

        Shares.append(Shares[-1])

temp = np.array(temp)

Shares = np.array(Shares)



airtel['Direction'] = temp

airtel['Cur_Shares'] = Shares

airtel.dropna(inplace=True) ## Time when we don't have moving avergae of

airtel['Transection'] = airtel['Direction'] * airtel['Close'] ## Bought or sold shares

airtel['Profits'] = airtel['Transection'].cumsum() ## Track of pfofit

airtel.set_index('Date', inplace=True)





fig = go.Figure(data=[

    go.Scatter(x = airtel.index, y=airtel['Close'], name="Daily Close"),

    go.Scatter(x = airtel.index, y=airtel['10_ma'], name="10 MA"),

    go.Scatter(x = airtel.index, y=airtel['30_ma'], name = "30 MA"),

    go.Scatter(x = airtel.index, y=airtel['Transection'], name = "Transections"),

    go.Scatter(x = airtel.index, y=airtel['Profits'], name="Total Profit")

])

fig.update_layout(title = "Trading History:",

                 xaxis_title="Date_Time", yaxis_title="Value")

fig.show()



print("Total Profit Till Day:", airtel['Profits'][-1])

print("Maximum Profit made in history:", airtel['Profits'].max())