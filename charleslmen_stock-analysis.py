# The first step import all dataset and set the Date as index except two all_stock files
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
% matplotlib inline
all_stock1=pd.read_csv('../input/all_stocks_2006-01-01_to_2018-01-01.csv')
all_stock1.head()
all_stock2=pd.read_csv('../input/all_stocks_2017-01-01_to_2018-01-01.csv')
all_stock2.head()
AABA=pd.read_csv('../input/AABA_2006-01-01_to_2018-01-01.csv').set_index('Date')
AABA.head()
AAPL=pd.read_csv('../input/AAPL_2006-01-01_to_2018-01-01.csv').set_index('Date')
AAPL.head()
AMZN=pd.read_csv('../input/AMZN_2006-01-01_to_2018-01-01.csv').set_index('Date')
AMZN.head()
AXP=pd.read_csv('../input/AXP_2006-01-01_to_2018-01-01.csv').set_index('Date')
AXP.head()
BA=pd.read_csv('../input/BA_2006-01-01_to_2018-01-01.csv').set_index('Date')
BA.head()
CAT=pd.read_csv('../input/CAT_2006-01-01_to_2018-01-01.csv').set_index('Date')
CAT.head()
CSCO=pd.read_csv('../input/CSCO_2006-01-01_to_2018-01-01.csv').set_index('Date')
CSCO.head()
CVX=pd.read_csv('../input/CVX_2006-01-01_to_2018-01-01.csv').set_index('Date')
CVX.head()
DIS=pd.read_csv('../input/DIS_2006-01-01_to_2018-01-01.csv').set_index('Date')
DIS.head()
GE=pd.read_csv('../input/GE_2006-01-01_to_2018-01-01.csv').set_index('Date')
GE.head()
GOOGL=pd.read_csv('../input/GOOGL_2006-01-01_to_2018-01-01.csv').set_index('Date')
GOOGL.head()
GS=pd.read_csv('../input/GS_2006-01-01_to_2018-01-01.csv').set_index('Date')
GS.head()
HD=pd.read_csv('../input/HD_2006-01-01_to_2018-01-01.csv').set_index('Date')
HD.head()
IBM=pd.read_csv('../input/IBM_2006-01-01_to_2018-01-01.csv').set_index('Date')
IBM.head()
INTC=pd.read_csv('../input/INTC_2006-01-01_to_2018-01-01.csv').set_index('Date')
INTC.head()
JNJ=pd.read_csv('../input/JNJ_2006-01-01_to_2018-01-01.csv').set_index('Date')
JNJ.head()
JPM=pd.read_csv('../input/JPM_2006-01-01_to_2018-01-01.csv').set_index('Date')
JPM.head()
KO=pd.read_csv('../input/KO_2006-01-01_to_2018-01-01.csv').set_index('Date')
KO.head()
MCD=pd.read_csv('../input/MCD_2006-01-01_to_2018-01-01.csv').set_index('Date')
MCD.head()
MMM=pd.read_csv('../input/MMM_2006-01-01_to_2018-01-01.csv').set_index('Date')
MMM.head()
MRK=pd.read_csv('../input/MRK_2006-01-01_to_2018-01-01.csv').set_index('Date')
MRK.head()
MSFT=pd.read_csv('../input/MSFT_2006-01-01_to_2018-01-01.csv').set_index('Date')
MSFT.head()
NKE=pd.read_csv('../input/NKE_2006-01-01_to_2018-01-01.csv').set_index('Date')
NKE.head()
PEE=pd.read_csv('../input/PFE_2006-01-01_to_2018-01-01.csv').set_index('Date')
PEE.head()
PG=pd.read_csv('../input/PG_2006-01-01_to_2018-01-01.csv').set_index('Date')
PG.head()
TRV=pd.read_csv('../input/TRV_2006-01-01_to_2018-01-01.csv').set_index('Date')
TRV.head()
UNH=pd.read_csv('../input/UNH_2006-01-01_to_2018-01-01.csv').set_index('Date')
UNH.head()
UTX=pd.read_csv('../input/UTX_2006-01-01_to_2018-01-01.csv').set_index('Date')
UTX.head()
VZ=pd.read_csv('../input/VZ_2006-01-01_to_2018-01-01.csv').set_index('Date')
VZ.head()
WMT=pd.read_csv('../input/WMT_2006-01-01_to_2018-01-01.csv').set_index('Date')
WMT.head()
XOM=pd.read_csv('../input/XOM_2006-01-01_to_2018-01-01.csv').set_index('Date')
XOM.head()
tickers=['AABA','AAPL','AMZN','AXP','BA','CAT','CSCO','CVX'
        ,'DIS','GE','GOOGL','GS','HD','IBM','INTC','JNJ',
         'JPM','KO','MCD','MMM','MRK','MSFT','NKE','PEE',
         'PG','TRV','UNH','UTX','VZ','WMT','XOM']
all_stock=pd.concat([AABA.iloc[:,:-1],AAPL.iloc[:,:-1],AMZN.iloc[:,:-1],AXP.iloc[:,:-1],BA.iloc[:,:-1],CAT.iloc[:,:-1],
                     CSCO.iloc[:,:-1],CVX.iloc[:,:-1],DIS.iloc[:,:-1],GE.iloc[:,:-1],GOOGL.iloc[:,:-1],GS.iloc[:,:-1],
                     HD.iloc[:,:-1],IBM.iloc[:,:-1], INTC.iloc[:,:-1],JNJ.iloc[:,:-1],JPM.iloc[:,:-1],KO.iloc[:,:-1],
                     MCD.iloc[:,:-1],MMM.iloc[:,:-1],MRK.iloc[:,:-1],MSFT.iloc[:,:-1],NKE.iloc[:,:-1],PEE.iloc[:,:-1],
                     PG.iloc[:,:-1],TRV.iloc[:,:-1], UNH.iloc[:,:-1],UTX.iloc[:,:-1],VZ.iloc[:,:-1],WMT.iloc[:,:-1],
                     XOM.iloc[:,:-1]],axis=1,keys=tickers,sort=True)
all_stock.columns.names = ['Bank Ticker','Stock Info']
all_stock.head()
# Try to find the average open,close price for each stock from 2006-01-01 to 2018-01-01
all_stock1.groupby('Name',as_index=False).mean()[['Name','Open','Close']]
plt.figure()
L=all_stock1.groupby('Name',as_index=False).mean()[['Name','Open','Close']]
x=np.array(L['Name'])
y=np.array(L['Open'])
z=np.array(L['Close'])
plt.subplot(3,1,1)
plt.plot(x,y)
plt.xticks(L['Name'],rotation='vertical')
plt.title('Average open price for each stock')
plt.xlabel('Stock Name')
plt.ylabel('Average open price')
plt.subplot(3, 1, 3)
plt.plot(x,z)
plt.xticks(L['Name'],rotation='vertical')
plt.title('Average close price for each stock')
plt.xlabel('Stock Name')
plt.ylabel('Average close price')
plt.show()
'''for all stocks in the time period from 2006-01-01 to 2009-12-31'''
# What is the max Open price for each stock throughout the time period
new_df_allstock1=all_stock1[(all_stock1['Date']>='2006-01-01')&(all_stock1['Date']<='2009-12-31')]#Find the data in the time period
new_df_allstock1.loc[new_df_allstock1.groupby('Name')['Open'].idxmax()][['Name','Date','Open']]#show the dataframe with max open price for each stock 
L=new_df_allstock1.loc[new_df_allstock1.groupby('Name')['Open'].idxmax()][['Name','Date','Open']]
x=np.array(L['Name'])
y=np.array(L['Open'])
plt.plot(x,y,'o',color='black')
plt.xticks(L['Name'],rotation='vertical')
plt.title('Max open price for each stock from 2006 to 2009')
plt.xlabel('Stock Name')
plt.ylabel('Max open price')
plt.show()
# Try to find returns for every stock each day which means how much the stock earns all losses each day
returns=pd.DataFrame()
for tick in tickers:
    returns[tick+' Return'] = all_stock[tick]['Close'].pct_change()
returns.head()
# Shows the return of GOOGL in year 2017
sns.distplot(returns.loc['2017-01-01':'2017-12-31']['GOOGL Return'],color='blue',bins=100)
# The trend of close price from 2006-01-01 to 2018-01-01 for all stocks
sns.set_style('whitegrid')
for tick in tickers:
    all_stock[tick]['Close'].plot(figsize=(12,11),label=tick)
plt.legend()
# Combine the GOOGL and AMZN as one DataFrame
ticker1=['GOOGL','AMZN']
two_stock=pd.concat([AMZN,GOOGL],axis=1,keys=ticker1)
two_stock.columns.names=['Stock ticker','Stock Info']
two_stock.head()
# try to find the highest open price of both stocks for each year
Year=[2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
x=np.array(Year)
AMZN_open_max=[AMZN['2006':'2007']['Open'].max(),AMZN['2007':'2008']['Open'].max(),AMZN['2008':'2009']['Open'].max(),
    AMZN['2009':'2010']['Open'].max(),AMZN['2010':'2011']['Open'].max(),AMZN['2011':'2012']['Open'].max(),
    AMZN['2012':'2013']['Open'].max(),AMZN['2013':'2014']['Open'].max(),AMZN['2014':'2015']['Open'].max(),
    AMZN['2015':'2016']['Open'].max(),AMZN['2016':'2017']['Open'].max(),AMZN['2017':'2018']['Open'].max()]
y=np.array(AMZN_open_max)
GOOGL_open_max=[GOOGL['2006':'2007']['Open'].max(),GOOGL['2007':'2008']['Open'].max(),GOOGL['2008':'2009']['Open'].max(),
    GOOGL['2009':'2010']['Open'].max(),GOOGL['2010':'2011']['Open'].max(),GOOGL['2011':'2012']['Open'].max(),
    GOOGL['2012':'2013']['Open'].max(),GOOGL['2013':'2014']['Open'].max(),GOOGL['2014':'2015']['Open'].max(),
    GOOGL['2015':'2016']['Open'].max(),GOOGL['2016':'2017']['Open'].max(),GOOGL['2017':'2018']['Open'].max()]
z=np.array(GOOGL_open_max)
plt.figure()
plt.plot(x, y, linestyle='solid',label="AMZN")
plt.plot(x, z, linestyle='dashed',label="GOOGL")
plt.title('Highest Open Price Each year')
plt.xlabel('Year')
plt.ylabel('Open Price')
plt.legend()
# try to find the lowest volume of both stocks for each year
Year=[2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
x=np.array(Year)
AMZN_open_min=[AMZN['2006':'2007']['Volume'].min(),AMZN['2007':'2008']['Volume'].min(),AMZN['2008':'2009']['Volume'].min(),
    AMZN['2009':'2010']['Volume'].min(),AMZN['2010':'2011']['Volume'].min(),AMZN['2011':'2012']['Volume'].min(),
    AMZN['2012':'2013']['Volume'].min(),AMZN['2013':'2014']['Volume'].min(),AMZN['2014':'2015']['Volume'].min(),
    AMZN['2015':'2016']['Volume'].min(),AMZN['2016':'2017']['Volume'].min(),AMZN['2017':'2018']['Volume'].min()]
y=np.array(AMZN_open_min)
GOOGL_open_min=[GOOGL['2006':'2007']['Volume'].min(),GOOGL['2007':'2008']['Volume'].min(),GOOGL['2008':'2009']['Volume'].min(),
    GOOGL['2009':'2010']['Volume'].min(),GOOGL['2010':'2011']['Volume'].min(),GOOGL['2011':'2012']['Volume'].min(),
    GOOGL['2012':'2013']['Volume'].min(),GOOGL['2013':'2014']['Volume'].min(),GOOGL['2014':'2015']['Volume'].min(),
    GOOGL['2015':'2016']['Volume'].min(),GOOGL['2016':'2017']['Volume'].min(),GOOGL['2017':'2018']['Volume'].min()]
z=np.array(GOOGL_open_min)

plt.figure()
plt.plot(x, y, linestyle='solid',label="AMZN")
plt.plot(x, z, linestyle='dashed',label="GOOGL")
plt.title('Lowest Volume')
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.legend()
# compare the close price
for ticker in ticker1:
        two_stock[ticker]['Close'].plot(figsize=(12,4),label=ticker)
plt.legend()
# compare the open price
for ticker in ticker1:
        two_stock[ticker]['Open'].plot(figsize=(12,4),label=ticker)
plt.legend()
# try to find out the relationship between close price and 30-day average close price for stocks from 2010 to 2011
plt.figure(figsize=(12,6))
AMZN['Close'].loc['2010-01-01':'2011-01-01'].rolling(window=30).mean().plot(label='30 Day Avg AMZN')
AMZN['Close'].loc['2010-01-01':'2011-01-01'].plot(label='AMZN CLOSE')
GOOGL['Close'].loc['2010-01-01':'2011-01-01'].rolling(window=30).mean().plot(label='30 Day Avg GOOGL')
GOOGL['Close'].loc['2010-01-01':'2011-01-01'].plot(label='GOOGL CLOSE')
plt.legend()
# compare the volume trend
plt.figure(figsize=(12,6))
AMZN['Volume'].loc['2017-01-01':'2018-01-01'].plot(label='AMZN Volume')
GOOGL['Volume'].loc['2017-01-01':'2018-01-01'].plot(label='GOOGL Volume')
plt.legend()
# show the correlation between open price of stocks
sns.heatmap(two_stock.xs(key='Open',axis=1,level='Stock Info').corr(),annot=True)
# Combine GS BA IBM CVX and MMM as one DataFrame
ticker2=['GS','BA','IBM','CVX','MMM']
five_stock=pd.concat([GS,BA,IBM,CVX,MMM],axis=1,keys=ticker2)
five_stock.columns.names=['Stock ticker','Stock Info']
five_stock.head()
# try to find the highest open price of both stocks for each year
Year=[2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
x=np.array(Year)
GS_open_max=[GS['2006':'2007']['Open'].max(),GS['2007':'2008']['Open'].max(),GS['2008':'2009']['Open'].max(),
    GS['2009':'2010']['Open'].max(),GS['2010':'2011']['Open'].max(),GS['2011':'2012']['Open'].max(),
    GS['2012':'2013']['Open'].max(),GS['2013':'2014']['Open'].max(),GS['2014':'2015']['Open'].max(),
    GS['2015':'2016']['Open'].max(),GS['2016':'2017']['Open'].max(),GS['2017':'2018']['Open'].max()]
y=np.array(GS_open_max)

BA_open_max=[BA['2006':'2007']['Open'].max(),BA['2007':'2008']['Open'].max(),BA['2008':'2009']['Open'].max(),
    BA['2009':'2010']['Open'].max(),BA['2010':'2011']['Open'].max(),BA['2011':'2012']['Open'].max(),
    BA['2012':'2013']['Open'].max(),BA['2013':'2014']['Open'].max(),BA['2014':'2015']['Open'].max(),
    BA['2015':'2016']['Open'].max(),BA['2016':'2017']['Open'].max(),BA['2017':'2018']['Open'].max()]
z=np.array(BA_open_max)

IBM_open_max=[IBM['2006':'2007']['Open'].max(),IBM['2007':'2008']['Open'].max(),IBM['2008':'2009']['Open'].max(),
    IBM['2009':'2010']['Open'].max(),IBM['2010':'2011']['Open'].max(),IBM['2011':'2012']['Open'].max(),
    IBM['2012':'2013']['Open'].max(),IBM['2013':'2014']['Open'].max(),IBM['2014':'2015']['Open'].max(),
    IBM['2015':'2016']['Open'].max(),IBM['2016':'2017']['Open'].max(),IBM['2017':'2018']['Open'].max()]
m=np.array(IBM_open_max)

CVX_open_max=[CVX['2006':'2007']['Open'].max(),CVX['2007':'2008']['Open'].max(),CVX['2008':'2009']['Open'].max(),
    CVX['2009':'2010']['Open'].max(),CVX['2010':'2011']['Open'].max(),CVX['2011':'2012']['Open'].max(),
    CVX['2012':'2013']['Open'].max(),CVX['2013':'2014']['Open'].max(),CVX['2014':'2015']['Open'].max(),
    CVX['2015':'2016']['Open'].max(),CVX['2016':'2017']['Open'].max(),CVX['2017':'2018']['Open'].max()]
n=np.array(CVX_open_max)

MMM_open_max=[MMM['2006':'2007']['Open'].max(),MMM['2007':'2008']['Open'].max(),MMM['2008':'2009']['Open'].max(),
    MMM['2009':'2010']['Open'].max(),MMM['2010':'2011']['Open'].max(),MMM['2011':'2012']['Open'].max(),
    MMM['2012':'2013']['Open'].max(),MMM['2013':'2014']['Open'].max(),MMM['2014':'2015']['Open'].max(),
    MMM['2015':'2016']['Open'].max(),MMM['2016':'2017']['Open'].max(),MMM['2017':'2018']['Open'].max()]
l=np.array(MMM_open_max)

plt.figure()
plt.plot(x, y, linestyle='solid',label="GS")
plt.plot(x, z, linestyle='dashed',label="BA")
plt.plot(x, m, linestyle='dashdot',label="IBM")
plt.plot(x, n, linestyle='dotted',label="CVX")
plt.plot(x, l, linestyle='dashed',label="MMM")
plt.title('Highest Open Price Each year')
plt.xlabel('Year')
plt.ylabel('Open Price')
plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
# try to find the lowest volume of both stocks for each year
Year=[2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
x=np.array(Year)
GS_volume_min=[GS['2006':'2007']['Volume'].min(),GS['2007':'2008']['Volume'].min(),GS['2008':'2009']['Volume'].min(),
    GS['2009':'2010']['Volume'].min(),GS['2010':'2011']['Volume'].min(),GS['2011':'2012']['Volume'].min(),
    GS['2012':'2013']['Volume'].min(),GS['2013':'2014']['Volume'].min(),GS['2014':'2015']['Volume'].min(),
    GS['2015':'2016']['Volume'].min(),GS['2016':'2017']['Volume'].min(),GS['2017':'2018']['Volume'].min()]
y=np.array(GS_volume_min)

BA_volume_min=[BA['2006':'2007']['Volume'].min(),BA['2007':'2008']['Volume'].min(),BA['2008':'2009']['Volume'].min(),
    BA['2009':'2010']['Volume'].min(),BA['2010':'2011']['Volume'].min(),BA['2011':'2012']['Volume'].min(),
    BA['2012':'2013']['Volume'].min(),BA['2013':'2014']['Volume'].min(),BA['2014':'2015']['Volume'].min(),
    BA['2015':'2016']['Volume'].min(),BA['2016':'2017']['Volume'].min(),BA['2017':'2018']['Volume'].min()]
z=np.array(GS_volume_min)

IBM_volume_min=[IBM['2006':'2007']['Volume'].min(),IBM['2007':'2008']['Volume'].min(),IBM['2008':'2009']['Volume'].min(),
    IBM['2009':'2010']['Volume'].min(),IBM['2010':'2011']['Volume'].min(),IBM['2011':'2012']['Volume'].min(),
    IBM['2012':'2013']['Volume'].min(),IBM['2013':'2014']['Volume'].min(),IBM['2014':'2015']['Volume'].min(),
    IBM['2015':'2016']['Volume'].min(),IBM['2016':'2017']['Volume'].min(),IBM['2017':'2018']['Volume'].min()]
m=np.array(IBM_volume_min)

CVX_volume_min=[CVX['2006':'2007']['Volume'].min(),CVX['2007':'2008']['Volume'].min(),CVX['2008':'2009']['Volume'].min(),
    CVX['2009':'2010']['Volume'].min(),CVX['2010':'2011']['Volume'].min(),CVX['2011':'2012']['Volume'].min(),
    CVX['2012':'2013']['Volume'].min(),CVX['2013':'2014']['Volume'].min(),CVX['2014':'2015']['Volume'].min(),
    CVX['2015':'2016']['Volume'].min(),CVX['2016':'2017']['Volume'].min(),CVX['2017':'2018']['Volume'].min()]
n=np.array(CVX_volume_min)

MMM_volume_min=[MMM['2006':'2007']['Volume'].min(),MMM['2007':'2008']['Volume'].min(),MMM['2008':'2009']['Volume'].min(),
    MMM['2009':'2010']['Volume'].min(),MMM['2010':'2011']['Volume'].min(),MMM['2011':'2012']['Volume'].min(),
    MMM['2012':'2013']['Volume'].min(),MMM['2013':'2014']['Volume'].min(),MMM['2014':'2015']['Volume'].min(),
    MMM['2015':'2016']['Volume'].min(),MMM['2016':'2017']['Volume'].min(),MMM['2017':'2018']['Volume'].min()]
l=np.array(MMM_volume_min)

plt.figure()
plt.plot(x, y, linestyle='solid',label="GS")
plt.plot(x, z, linestyle='dashed',label="BA")
plt.plot(x, m, linestyle='dashdot',label="IBM")
plt.plot(x, n, linestyle='dotted',label="CVX")
plt.plot(x, l, linestyle='dashed',label="MMM")
plt.title('Highest Open Price Each year')
plt.xlabel('Year')
plt.ylabel('Open Price')
plt.legend(loc='best',frameon=False)
# compare close price over year
for ticker in ticker2:
        five_stock[ticker]['Close'].plot(figsize=(12,4),label=ticker)
plt.legend()
# compare open price over year
for ticker in ticker2:
        five_stock[ticker]['Open'].plot(figsize=(12,4),label=ticker)
plt.legend()
# try to find out the relationship between close price and 30-day average close price for stocks from 2010 to 2011
plt.figure(figsize=(12,6))
GS['Close'].loc['2010-01-01':'2011-01-01'].rolling(window=30).mean().plot(label='30 Day Avg GS')
GS['Close'].loc['2010-01-01':'2011-01-01'].plot(label='GS CLOSE')
BA['Close'].loc['2010-01-01':'2011-01-01'].rolling(window=30).mean().plot(label='30 Day Avg BA')
BA['Close'].loc['2010-01-01':'2011-01-01'].plot(label='BA CLOSE')
IBM['Close'].loc['2010-01-01':'2011-01-01'].rolling(window=30).mean().plot(label='30 Day Avg IBM')
IBM['Close'].loc['2010-01-01':'2011-01-01'].plot(label='IBM CLOSE')
CVX['Close'].loc['2010-01-01':'2011-01-01'].rolling(window=30).mean().plot(label='30 Day Avg CVX')
CVX['Close'].loc['2010-01-01':'2011-01-01'].plot(label='CVX CLOSE')
MMM['Close'].loc['2010-01-01':'2011-01-01'].rolling(window=30).mean().plot(label='30 Day Avg MMM')
MMM['Close'].loc['2010-01-01':'2011-01-01'].plot(label='MMM CLOSE')
plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
# compare volume trend
plt.figure(figsize=(12,6))
GS['Volume'].loc['2017-01-01':'2018-01-01'].plot(label='GS Volume')
BA['Volume'].loc['2017-01-01':'2018-01-01'].plot(label='BA Volume')
IBM['Volume'].loc['2017-01-01':'2018-01-01'].plot(label='IBM Volume')
CVX['Volume'].loc['2017-01-01':'2018-01-01'].plot(label='CVX Volume')
MMM['Volume'].loc['2017-01-01':'2018-01-01'].plot(label='MMM Volume')
plt.legend()
# show the correlation between open price of stocks
sns.heatmap(five_stock.xs(key='Open',axis=1,level='Stock Info').corr(),annot=True)
