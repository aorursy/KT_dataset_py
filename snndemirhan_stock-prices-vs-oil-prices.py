!pip install yfinance
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import yfinance as yf

%matplotlib inline

plt.style.use("fivethirtyeight")
path = '/kaggle/input/ntt-data-global-ai-challenge-06-2020/'

time_series=pd.read_csv(path+"Crude_oil_trend_From1986-10-16_To2020-03-31.csv")

sample_submission=pd.read_csv(path+"sampleSubmission.csv")
print('Number of data points : ', time_series.shape[0])

print('Number of features : ', time_series.shape[1])

time_series.head() # to print first 5 rows
Yahoo_indeces=["^GSPC","^DJI","NQ=F","^FTSE","^GDAXI","^ISEQ","^N225","^XU100",

               "^FCHI","IMOEX.ME","^OMX","^OSEAX","XIU.TO","^BVSP","^MXX","^BSESN","^SSE50",

               "^KS11","^NZ50","^AXJO","^BFX","^ATX","^PSI20","BTCUSD=X","EURUSD=X",

              "MSFT","AAPL","AMZN","FB",

        "BLK","JNJ","V","PG","UNH",

        "JPM","INTC","HD","MA","VZ",

        "PFE","T","MRK","NVDA","NFLX","DIS",

        "CSCO","PEP","XOM","BAC","WMT","ADBE","CVX","BA","KO","CMCSA","ABT","WFC",

         "BMY","CRM","AMGN","TMO","LLY","COST","MCD","MDT","ORCL","ACN","NEE","NKE","UNP","AVGO","PM","IBM","LMT","QCOM","DAL"]





Stock_indeces=["S&P500","DowJones","NASDAQ100","FTSE100","DAX","ISEQ","NIKKEI225","BIST100",

               "CAC40","MOEX","OMXS30","Oslo BorsAll-Share","TSX","IBOVESPA","IPCMexico","SHANGHAI50",

               "SENSEX","KOSPI","NZX50","ASX200","BEL20","ATX","PSI20","BTC/USD","EUR/USD",

              "Microsoft Corporation","Apple Inc.","Amazon","Facebook","BlackRock Inc.","Johnson & Johnson",

             "Visa Inc.","Procter & Gamble","UnitedHealth Group","JPMorgan",

            "Intel Corporation","Home Depot Inc.","Mastercard","Verizon Communications Inc.","Pfizer","AT&T Inc.","Merck & Co. Inc",

             "NVIDIA Corporation","Netflix Inc.","Walt Disney Company","Cisco Systems Inc.",

            "PepsiCo Inc.","Exxon Mobil Corporation","Bank of America Corp","Walmart Inc.","Adobe Inc.","Chevron Corporation",

             "Boeing Company","Coca-Cola Company","Comcast Corporation Class A","Abbott Laboratories","Wells Fargo & Company",

            "Bristol-Myers Squibb Company","Salesforce","Amgen Inc.","Thermo Fisher Scientific Inc.","Eli Lilly and Company",

             "Costco Wholesale Corporation","McDonald's Corporation","Medtronic Plc","Oracle Corporation","Accenture Plc Class A"

             ,"NextEra Energy Inc.",

            "NIKE Inc. Class B","Union Pacific Corporation","Broadcom Inc.","Philip Morris International","International Business Machines Corporation"

             ,"Lockheed Martin Corporation","QUALCOMM Incorporated","Delta Air"]



for i in range(len(Stock_indeces)):

    data = yf.download(Yahoo_indeces[i], start="2019-01-02", end="2020-05-23")

    stock_hist=pd.DataFrame(data[['Close','Volume']])

    stock_hist['Stock_indeces']=Stock_indeces[i]

    if i == 0:

        stock_history=stock_hist

    else:

        stock_history=pd.concat([stock_history,stock_hist])
oil=time_series[time_series['Date']>='2019-01-02']

oil=oil.reset_index(drop=True, inplace=False)

oil=pd.concat([oil,sample_submission])

oil=oil.reset_index(drop=True, inplace=False)

oil.head()
stock_history.head()
stock_history['Stock_indeces'].unique()
stock_history['date']=stock_history.index

country_indeces=stock_history.pivot_table(index='date', columns='Stock_indeces', values='Close',fill_value=0)

country_indeces['Date']=country_indeces.index

country_indeces=country_indeces.reset_index(drop=True, inplace=False)

country_indeces['Date']=country_indeces['Date'].dt.strftime('%Y-%m-%d') #date column in downloaded data is not in wanted type.So I changed its type in here

country_indeces.head()
all_stocks=pd.merge(oil,country_indeces, on='Date')

#all_stocks.astype(bool).sum(axis=0)

all_stocks.astype(bool).sum(axis=0).unique()
all_stocks=all_stocks.reset_index(drop=True, inplace=False)

all_stocks.head()
def equal_up_value(df,my_len):

    

    for i in range(len(df.columns)):

        if df[df.columns[i]][2]==0:

            df[df.columns[i]][2]=df[df.columns[i]][3]

        if df[df.columns[i]][1]==0:

            df[df.columns[i]][1]=df[df.columns[i]][2]

        if df[df.columns[i]][0]==0:

            df[df.columns[i]][0]=df[df.columns[i]][1]

            

    for i in range(len(df.columns)):

        for j in range(my_len-1):

            if df[df.columns[i]][j+1]==0:

                df[df.columns[i]][j+1]=df[df.columns[i]][j]



equal_up_value(all_stocks,len(all_stocks))    
all_stocks.astype(bool).sum(axis=0).unique()
dates=all_stocks['Date']

all_stocks=all_stocks.drop(['Date'], axis=1)
def normalize(dataset):

    dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))

    #dataNorm["Price"]=dataset["Price"]

    return dataNorm



new_data=normalize(all_stocks[0:len(all_stocks)-37])

new_data.head()
pd.concat([new_data.T[0:1],new_data.T[1:10]]).T.plot(figsize=(16,8)).legend(bbox_to_anchor=(1, 0.8))
pd.concat([new_data.T[0:1],new_data.T[10:20]]).T.plot(figsize=(16,8)).legend(bbox_to_anchor=(1, 0.8))
correlations=pd.DataFrame(new_data[200:].corr()['Price'])

correlations[(correlations['Price']>0.92)]
pd.concat([pd.DataFrame(new_data.T.loc['Price']).T,

           pd.DataFrame(new_data.T.loc['ATX']).T,

          pd.DataFrame(new_data.T.loc['Chevron Corporation']).T,

          pd.DataFrame(new_data.T.loc['Delta Air']).T,

          pd.DataFrame(new_data.T.loc['Exxon Mobil Corporation']).T,

          pd.DataFrame(new_data.T.loc['FTSE100']).T,

          pd.DataFrame(new_data.T.loc['JPMorgan']).T,

          pd.DataFrame(new_data.T.loc['Oslo BorsAll-Share']).T,

          pd.DataFrame(new_data.T.loc['Pfizer']).T,

          pd.DataFrame(new_data.T.loc['SHANGHAI50']).T,

          pd.DataFrame(new_data.T.loc['Walt Disney Company']).T,

          ]).T.plot(figsize=(16,8)).legend(bbox_to_anchor=(1, 0.8))