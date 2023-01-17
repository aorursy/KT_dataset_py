import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
# Importing DJIA index data for checking stationarity
time_series_data=pd.read_csv('../input/dataset-financial/DJIA_table.csv',parse_dates=[0],usecols=['Date','Adj Close'],date_parser=lambda x: pd.datetime.strptime(x,'%Y-%m-%d'))
# Setting 'Date' as the index of dataframe
time_series_data.set_index('Date',inplace=True)
time_series_data.reindex()
time_series_data.sort_index(ascending=True, inplace=True)
# Have a look at the data
time_series_data['Direction'] = np.where(time_series_data['Adj Close'].shift(-5) <= time_series_data['Adj Close'], 0, 1)
time_series_data=time_series_data.loc['2011-04-01':'2015-04-01']
time_series_data.dropna(inplace=True)
#time_series_data.sort_index(ascending=True, inplace=True)
#time_series_data.fillna(method='ffill',inplace=True)
time_series_data.drop(['Adj Close'],axis=1,inplace=True)
time_series_data.head()
print(time_series_data['Direction'])
# Defining ACF and PACF plot functions-
def ACF_plot(data,title):
    plot_acf(data,lags=20)
    plt.title(title)
    plt.show()
def PACF_plot(data,title):
    plot_pacf(data,lags=20)
    plt.title(title)
    plt.show()
ACF_plot(time_series_data,'DJIA index ACF plot')
PACF_plot(time_series_data,'DJIA index PACF plot')
articles = pd.read_csv('../input/dataset-financial/all_results.csv')

articles.Date = pd.to_datetime(articles.Date, format='%d/%m/%Y %H:%M:%S')

articles.set_index('Date', inplace=True)
articles.reindex()
articles.sort_index(ascending=True, inplace=True)
articles.drop(columns=['Original', 'Original', 'TotalWords', 'TotalSentimentWords', 'Id', 'Anger',
                           'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust'], inplace=True)

#print(articles.head())
#articles['Calculated']=articles['Calculated']*1000

articles['Calculated'] = articles['Calculated'].apply(lambda x: (x-3)/2)
#articles['Calculated'] = articles['Calculated'].apply(lambda x: 1 if x>0 else -1)

#articles.fillna(value=0,inplace=True)

articles = articles.groupby(pd.Grouper(freq = 'D')).mean()
articles=articles.loc['2011-04-01':'2015-04-01']
articles.fillna(value=0,inplace=True)
#articles['Calculated'] = articles['Calculated'].apply(lambda x: 1 if x>0 else 0)
print(articles.isna().sum())
print(articles)
#articles.reindex()
#articles.sort_index(ascending=True, inplace=True)

articles.head()
ACF_plot(articles,'Sentiment Attitudes ACF plot')
PACF_plot(articles,'Sentiment Attitudes PACF plot')
# ADF test for checking stationarity:
def adf_test(data,title,name):
    print('Results of Augmented Dickey Fuller Test for {}:'.format(title))
    dftest = adfuller(data[name], autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)
def import_time_series_data(stock_name):
    # Importing DJIA index data for checking stationarity
    time_series_data=pd.read_csv('../input/quandl-time-series-data/{}_quandl_stock.csv'.format(stock_name),parse_dates=[0],usecols=['date','Close'],date_parser=lambda x: pd.datetime.strptime(x,'%Y-%m-%d'))
    # Setting 'Date' as the index of dataframe
    time_series_data.set_index('date',inplace=True)
    time_series_data.reindex()
    time_series_data.sort_index(ascending=True, inplace=True)
    # Have a look at the data
    time_series_data['Direction'] = np.where(time_series_data['Close'].shift(-5) <= time_series_data['Close'], 0, 1)
    #time_series_data['Direction']=time_series_data['Close']
    time_series_data.drop(['Close'],axis=1,inplace=True)
    time_series_data=time_series_data.loc['2011-04-01':'2015-04-01']
    time_series_data.sort_index(ascending=True, inplace=True)
    #time_series_data.fillna(value=,inplace=True)
    return time_series_data
DJIA_time_series=time_series_data
adf_test(DJIA_time_series,'DJIA index','Direction')
AAPL_time_series=import_time_series_data('AAPL')
adf_test(AAPL_time_series,'AAPL','Direction')
GOOGL_time_series=import_time_series_data('GOOGL')
adf_test(GOOGL_time_series,'GOOGL','Direction')
JPM_time_series=import_time_series_data('JPM')
adf_test(JPM_time_series,'JPM','Direction')
HPQ_time_series=import_time_series_data('HPQ')
adf_test(HPQ_time_series,'HPQ','Direction')


def cross_correlation(x,y,title):
    plt.xcorr(x,y,maxlags=20,)
    plt.title(title)
    plt.show()
# importing S&p data
# Importing DJIA index data for checking stationarity
spy=pd.read_csv('../input/dataset-financial/spy.csv',parse_dates=[0],usecols=['Date','Value'],date_parser=lambda x: pd.datetime.strptime(x,'%d/%m/%Y'))
# Setting 'Date' as the index of dataframe
spy.set_index('Date',inplace=True)
spy.reindex()
spy.sort_index(ascending=True, inplace=True)
# Have a look at the data
spy['Direction'] = np.where(spy['Value'].shift(-5) <= spy['Value'], 0, 1)
spy=spy.loc['2011-04-01':'2015-04-01']
spy.dropna(inplace=True)
#time_series_data.sort_index(ascending=True, inplace=True)
#time_series_data.fillna(method='ffill',inplace=True)
spy.drop(['Value'],axis=1,inplace=True)
spy.head()
print(spy['Direction'])

# for financial news 1.
spy1=spy.loc['2011-04-01':'2011-12-26']
financial_news_1=articles.loc[spy1.index]
print(spy1.shape,financial_news_1.shape)
cross_correlation(financial_news_1.values.flatten(),spy1.values.flatten(),'Sentiment attitude (FT-1) && S&P 500')

# financial news 2
spy2=spy.loc['2014-04-01':'2014-10-26']
financial_news_2=articles.loc[spy2.index]
print(spy2.shape,financial_news_2.shape)
cross_correlation(financial_news_2.values.flatten(),spy2.values.flatten(),'Sentiment attitude (FT-2) && S&P 500')

# financial news 3
spy3=spy.loc['2014-10-26':'2015-3-08']
financial_news_3=articles.loc[spy3.index]
print(spy3.shape,financial_news_3.shape)
cross_correlation(financial_news_3.values.flatten(),spy3.values.flatten(),'Sentiment attitude (FT-3) && S&P 500')
# Granger causality test
from statsmodels.tsa.stattools import grangercausalitytests
def granger_causality(stock,sentiment,stock_name):
    print('Sentiment causing price of {} stock'.format(stock_name))
    grangercausalitytests(np.hstack([stock,sentiment]),maxlag=2,verbose=0.25)
    print('\nPrice of {} stock causing sentiment'.format(stock_name))
    grangercausalitytests(np.hstack([sentiment,stock]),maxlag=2,verbose=0.25)
ft1_dates=articles.loc['2011-04-01':'2011-12-25'].index
stocktoseries={'AAPL':AAPL_time_series,
              'GOOGL':GOOGL_time_series,
              'SPY':spy,
              'JPM':JPM_time_series,
              'HPQ':HPQ_time_series}
stocks=['AAPL','GOOGL','SPY','JPM','HPQ']
stocks=['HPQ']
for stock in stocks:
    dates=stocktoseries[stock].loc['2011-04-01':'2011-12-25'].index
    stock_data=stocktoseries[stock].loc['2011-04-01':'2011-12-25'].values
    sentiment_data=articles.loc[dates].values
    #print(np.hstack([stock_data,sentiment_data]))
    granger_causality(stock_data,sentiment_data,stock)
    
    
    

