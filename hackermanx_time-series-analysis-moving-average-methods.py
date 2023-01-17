#importing the python libraries - pandas and numpy 
import pandas as pd
import numpy as np
#Importing & reading the dataset
#This creates a dataframe from the CSV file:
df=pd.read_csv("../input/TCS.NS.csv",index_col='Date')
df
#Checking if any column has any empty value:
df.isna().sum()
#dropping the empty data entries:
df.dropna(inplace=True)
df.isna().sum()
df
#Importing the matplot library for data visualization
import matplotlib.pyplot as plt
%matplotlib inline
#this selects the 'Adj Close' column
close=df['Adj Close']

#This converts the date strings in the index into pandas datetime format:
close.index=pd.to_datetime(close.index)
close
#Plot the price series for visual inspection:
close.plot()
product = {'month' : [1,2,3,4,5,6,7,8,9,10,11,12],'demand':[290,260,288,300,310,303,329,340,316,330,308,310]}
dfe = pd.DataFrame(product)
dfe.head()

for i in range(0,dfe.shape[0]-2):
    dfe.loc[dfe.index[i+2],'SMA_3'] = np.round(((dfe.iloc[i,1]+ dfe.iloc[i+1,1] +dfe.iloc[i+2,1])/3),1)
dfe.head()    
dfe['pandas_SMA_3'] = np.round(dfe.iloc[:,1].rolling(window=3).mean(),1)
dfe.head()

for i in range(0,dfe.shape[0]-3):
    dfe.loc[dfe.index[i+3],'SMA_4'] = np.round(((dfe.iloc[i,1]+ dfe.iloc[i+1,1] +dfe.iloc[i+2,1]+dfe.iloc[i+3,1])/4),1)
dfe.head()    
dfe['pandas_SMA_4'] = dfe.iloc[:,1].rolling(window=4).mean()
dfe.head()
sma_50=close.rolling(window=50).mean()
sma_50
#As we expect the first 49 values of the sma_50 series are empty
sma_50.iloc[45:55]
#To mprove our plot appearence ,we can use a pre-defined style 
plt.style.use('fivethirtyeight')
#Our Chart size
plt.figure(figsize = (13,7))

#Plotting 'Adj Close' Price and SMA_50 lines
plt.plot(close,label='TCS Adj Close',linewidth=3)
plt.plot(sma_50,label='50 day Rolling SMA',linewidth=1.5)

#Adding title, labels on the axis:
#Legend on the left-upper corner
plt.xlabel('Date')
plt.ylabel('TCS Adjusted Closing Price ($)')
plt.title('Pot of Adj Close v/s Simple_MA_50')
plt.legend()

sma_20 = close.rolling(window=20).mean()

plt.figure(figsize=(15,8))

#Plotting 'AdjClose' price with 2 SMA's ie. sma_20 & sma_50
plt.plot(close,label='TCS Adj Close',linewidth=3)
plt.plot(sma_20,label='20 day Rolling SMA',linewidth=1.5)
plt.plot(sma_50,label='50 day Rolling SMA',linewidth=1.5)

plt.xlabel('DATE')
plt.ylabel('Adjusted Closing Price ($)')
plt.title('Plot of Adjusted Closing Price V/S Simple Moving Averages')
priceSma_df = pd.DataFrame({
      'Adj Close' : close,
      'SMA 20' : sma_20,
      'SMA 50' : sma_50
     })
priceSma_df
priceSma_df.plot()
plt.show()
plt.figure(figsize = (12,6))
#Plotting the Adjusted Price and two SMAs for The Corona Period:
plt.plot(priceSma_df['2019-10-01':'2020-05-28']['Adj Close'], label='TCS Adj Close', linewidth = 2)
plt.plot(priceSma_df['2019-10-01':'2020-05-28']['SMA 20'], label='20 days rolling SMA', linewidth = 1.5)
plt.plot(priceSma_df['2019-10-01':'2020-05-28']['SMA 50'], label='50 days rolling SMA', linewidth = 1.5)
plt.xlabel('Date')
plt.ylabel('Adjusted closing price ($)')
plt.title('Price with Two Simple Moving Averages - The Corona Period')
plt.legend()
plt.show()
sma_200=close.rolling(window=200).mean()
priceSma_df['SMA 200']=sma_200
priceSma_df
#defining the start and end dates beforehand to avoid complexity:
start='2015-01-01'
end='2020-05-28'

plt.figure(figsize=(12,7))

#Plotting the Adjusted Closing Price with SMA_50 & SMA_200
plt.plot(priceSma_df[start:end]['Adj Close'],label='TCS Adj Closing Price',linewidth=3)
plt.plot(priceSma_df[start:end]['SMA 200'],label='200 day rolling SMA',linewidth=1)
plt.plot(priceSma_df[start:end]['SMA 50'],label='50 day rolling SMA',linewidth=1)

plt.xlabel('Date')
plt.ylabel('Adjusted closing price {$}')
plt.title('Price with SMA_200 and SMA_50')
plt.legend()

plt.show()


cma_50 = close.expanding(min_periods=50).mean()
cma_50
#So, we can see that the last row value of CMA is just the average(mean) of all the previous data up until the last data point.
close.mean()
plt.style.use('fivethirtyeight')
#Our Chart size
plt.figure(figsize = (13,7))

#Plotting 'Adj Close' Price and SMA_50 lines
plt.plot(close,label='TCS Adj Close',linewidth=3)
plt.plot(cma_50,label='50 day Expanding CMA',linewidth=1.5)
plt.plot(sma_50,label='50 day Rolling SMA',linewidth=1.5)

#Adding title, labels on the axis:
#Legend on the left-upper corner
plt.xlabel('Date')
plt.ylabel('TCS Adjusted Closing Price ($)')
plt.title('Pot of Adj Close & Cumulative_MA_50 V/S Simple_MA_50')
plt.legend()
ema_50 = close.ewm(span=50,adjust=False).mean()
ema_50
plt.style.use('fivethirtyeight')
#Our Chart size
plt.figure(figsize = (13,7))

#Plotting 'Adj Close' Price and SMA_50 lines
plt.plot(close,label='TCS Adj Close',linewidth=3)
plt.plot(cma_50,label='50 day Expanding CMA',linewidth=1.5)
plt.plot(sma_50,label='50 day Rolling SMA',linewidth=1.5)
plt.plot(ema_50,label='50 day Exponential EMA',linewidth=2.5)

#Adding title, labels on the axis:
#Legend on the left-upper corner
plt.xlabel('Date')
plt.ylabel('TCS Adjusted Closing Price ($)')
plt.title('Pot of Adj Close & Cumulative_MA_50 V/S Simple_MA_50 V/S Exponential_MA_50')
plt.legend()
plt.figure(figsize = (12,6))
#Plotting the Adjusted Price & SMA V/S CMA V/S EMA for The Corona Period:
plt.plot(close['2019-10-01':'2020-05-28'], label='TCS Adj Close', linewidth = 2)
plt.plot(sma_50['2019-10-01':'2020-05-28'], label='50 days rolling SMA', linewidth = 1.5)
plt.plot(cma_50['2019-10-01':'2020-05-28'], label='50 days expanding CMA', linewidth = 1.5)
plt.plot(ema_50['2019-10-01':'2020-05-28'], label='50 days exponential CMA', linewidth = 1.5)

plt.xlabel('Date')
plt.ylabel('Adjusted closing price ($)')
plt.title('Price with Various Moving Averages - The Corona Period')
plt.legend()
plt.show()
ema_20 = close.ewm(span=20,adjust=True).mean()
ema_20
ema_5 = close.ewm(span=5,adjust=True).mean()
ema_5
plt.figure(figsize = (12,6))
#Plotting the Adjusted Price & SMA V/S CMA V/S EMA for The Corona Period:
plt.plot(close['2019-10-01':'2020-05-28'], label='TCS Adj Close', linewidth = 5)
plt.plot(sma_20['2019-10-01':'2020-05-28'], label='20 days rolling SMA', linewidth = 1.5)
plt.plot(ema_20['2019-10-01':'2020-05-28'], label='20 days exponential EMA', linewidth = 1.5,color='green')
plt.plot(ema_5['2019-10-01':'2020-05-28'], label='5 days exponential EMA', linewidth = 2.5,color='red')

plt.xlabel('Date')
plt.ylabel('Adjusted closing price ($)')
plt.title('Price with SMA_20 & EMA_20 & EMA_5- The Corona Period')
plt.legend()
plt.show()