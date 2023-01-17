!pip install arch
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from datetime import timedelta

from sklearn.cluster import KMeans 

import matplotlib.pyplot as plt 



from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.stattools import adfuller

from arch import arch_model

from tensorflow.keras import Sequential

from tensorflow.keras.layers import LSTM, Dense, Bidirectional,TimeDistributed, RepeatVector, Input



import seaborn as sns

import os

import warnings

warnings.filterwarnings('ignore')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
bar_S = pd.read_csv('/kaggle/input/stock-market-small-wide-dataset/bar-S.csv')

quote = pd.read_csv('/kaggle/input/stock-market-small-wide-dataset/quote-S.csv')
bar_S.head()
quote.head()
bar_S = bar_S.sort_values(by='time')

bar_S['date'] = pd.to_datetime(bar_S['time']).apply(lambda x: x.strftime('%Y-%m-%d'))
stock_list = list(set(bar_S['symbol']))

df_list = []

df_list_2 = []

window = 30

n_minutes = 10

date_list = list(set(bar_S['date']))

features = ['open_price','high_price','low_price','close_price',

            'average_price','VWAP','volume','accumulated_volume','symbol','date','time']

bar_S_window_time = 0

for i in stock_list:

    new_bar_S = bar_S[bar_S['symbol']==i]

    for j in date_list: 

        try:

            lower_limit = list(new_bar_S[new_bar_S['date']==j].sort_values(by='time')['time'])[0]

            lower_limit = pd.to_datetime(lower_limit)

            upper_limit = lower_limit + timedelta(minutes=window*n_minutes)

            df = new_bar_S[(new_bar_S['date']==j) & (new_bar_S['time']>=str(lower_limit)) & (new_bar_S['time']<str(upper_limit))]

            sorted_df = df.sort_values(by='time')[features]

            df_list_2.append(sorted_df)

            sorted_df = sorted_df.groupby('date').mean()

            sorted_df['symbol'] = i

            df_list.append(sorted_df)

        except Exception as e:

            pass

bar_S_windowed = pd.concat(df_list)

bar_S_windowed = bar_S_windowed.sort_index()

bar_S_window_time = pd.concat(df_list_2)

bar_S_col_list = list(bar_S_windowed)

bar_S_windowed.columns = ['mean '+col if col!='symbol' else 'symbol' for col in bar_S_col_list]

bar_S_windowed.head()
price_features = ['mean open_price','mean high_price','mean low_price','mean close_price',

                  'mean average_price','mean VWAP']

count = 0

cm = plt.get_cmap('rainbow')

colors = cm(np.linspace(0, 1, 24))

plt.rcParams['figure.figsize'] = (10,10)

for index in date_list: 

    bar_S_price_data = bar_S_windowed.loc[index,price_features]

    cluster_list = range(1,21)

    wss_list = []

    for i in cluster_list:

        km = KMeans(n_clusters=i)

        km.fit(bar_S_price_data)

        wss_list.append(km.inertia_)

    plt.plot(cluster_list,wss_list,marker='o',color=colors[count])

    plt.xlabel('No. of clusters')

    plt.ylabel('Within sum of squares (WSS)')

    plt.title('Elbow Curve')

    plt.legend(date_list)

    count+=1

    for i, txt in enumerate(cluster_list):

        plt.annotate(txt, (cluster_list[i], wss_list[i])) 
km = KMeans(n_clusters=4)

bar_S_price_data =  bar_S_windowed.loc[:,price_features]

km.fit(bar_S_price_data)

label = km.predict(bar_S_price_data)
stock_cluster = dict(zip(bar_S_windowed.loc[:,'symbol'],label))

cluster_0_N1=[]

cluster_1_N1=[]

cluster_2_N1= []

cluster_3_N1=[]

for key in stock_cluster:

    if stock_cluster[key]==0:

        cluster_0_N1.append(key)    

    elif stock_cluster[key]==1:

        cluster_1_N1.append(key)

    elif stock_cluster[key]==2:

        cluster_2_N1.append(key)

    else:

        cluster_3_N1.append(key)

print('Stocks belonging to cluster 0:',','.join(set(cluster_0_N1)))

print('\nStocks belonging to cluster 1:',','.join(set(cluster_1_N1)))

print('\nStocks belonging to cluster 2:',','.join(set(cluster_2_N1)))

print('\nStocks belonging to cluster 3:',','.join(set(cluster_3_N1)))
def plot_price_curve(cluster,n_stocks):

    plt.rcParams['figure.figsize']=(50,20)

    count = 1

    for stock in cluster[:n_stocks]:

        df = bar_S_windowed[bar_S_windowed['symbol']==stock]

        features = ['mean open_price','mean high_price','mean low_price','mean close_price',

                    'mean average_price','mean VWAP','mean volume','mean accumulated_volume']

        for feature in features:

            plt.title('For stock '+stock)

            plt.subplot(n_stocks,8,count)

            sns.lineplot(data = df,x = df.index,y = feature)

            count+=1  
print('For cluster 0 of N1')

plot_price_curve(cluster_0_N1,4)
print('For cluster 1 of N1')

plot_price_curve(cluster_1_N1,4)
print('For cluster 2 of N1')

plot_price_curve(cluster_2_N1,4)
print('For cluster 3 of N1')

plot_price_curve(cluster_3_N1,4)
def stationary_test(timeseries):

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    return dftest[2]



def ARIMA_prediction_plot(timeseries,split):

    train_ts = timeseries[:split]

    test_ts = timeseries[split:]

    p_val = stationary_test(timeseries)

    model = ARIMA(train_ts, order=(p_val,1,2))  

    results_ARIMA = model.fit(disp=-1)

    #Future Forecasting

    history = list(train_ts)

    predictions = []

    test_data = list(test_ts)

    for i in range(len(test_data)):

        model = ARIMA(history, order=(p_val,1,2))

        model_fit = model.fit(disp=-1)

        output = model_fit.forecast()

        yhat = output[0]

        predictions.append(float(yhat))

        history.append(float(yhat))

    plt.rcParams['figure.figsize'] = (20,10)

    plt.plot(timeseries)

    plt.plot(test_ts.index,predictions,color='green')

    plt.axvline(train_ts.index[-1],color='orange',dashes=(5,2,1,2))

    plt.xlabel('Average price')

    plt.ylabel('time')

    plt.legend(['actual values','predicted values for test data'])
bar_S_window_time.index = bar_S_window_time['time']

bar_S_window_time = bar_S_window_time.sort_index()
stock = cluster_0_N1[0]

print('For N1 cluster 0 stock: ',stock)

timeseries = bar_S_window_time[bar_S_window_time['symbol']==stock]['average_price']

ARIMA_prediction_plot(timeseries,int(0.90*len(timeseries)))
stock = cluster_1_N1[0]

print('For N1 cluster 1 stock: ',stock)

timeseries = bar_S_window_time[bar_S_window_time['symbol']==stock]['average_price']

ARIMA_prediction_plot(timeseries,int(0.90*len(timeseries)))
stock = cluster_2_N1[0]

print('For N1 cluster 2 stock: ',stock)

timeseries = bar_S_window_time[bar_S_window_time['symbol']==stock]['average_price']

ARIMA_prediction_plot(timeseries,int(0.90*len(timeseries)))
stock = cluster_3_N1[0]

print('For N1 cluster 3 stock: ',stock)

timeseries = bar_S_window_time[bar_S_window_time['symbol']==stock]['average_price']

ARIMA_prediction_plot(timeseries,int(0.90*len(timeseries)))
cluster_list = [cluster_0_N1, cluster_1_N1, cluster_2_N1, cluster_3_N1]

plt.rcParams['figure.figsize']=(20,5)

for index,cluster in enumerate(cluster_list):

    text = 0

    stock = cluster[0]

    if cluster==cluster_0_N1:

        text='For N1 cluster 0 stock: '+stock

    elif cluster==cluster_1_N1:

        text='For N1 cluster 1 stock: '+stock

    elif cluster==cluster_2_N1:

        text='For N1 cluster 2 stock: '+stock

    else:

        text='For N1 cluster 3 stock: '+stock

    timeseries = bar_S_window_time[bar_S_window_time['symbol']==stock]['average_price']

    model = arch_model(timeseries, mean='ARX', vol='GARCH', p=9)

    model_fit = model.fit(disp=-1)

    y = model_fit.forecast(horizon=30)

    plt.subplot(1,4,index+1)

    plt.title(text)

    plt.ylabel('variance')

    plt.xlabel('epochs')

    plt.plot(y.variance.values[-1],color='red')
# split a univariate sequence into samples

def split_sequence(sequence, n_steps_in, n_steps_out):

    X, y = [],[]

    for i in range(len(sequence)):

        end_ix = i + n_steps_in

        out_end_ix = end_ix + n_steps_out

        if out_end_ix > len(sequence):

            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]

        X.append(seq_x)

        y.append(seq_y)

    return np.array(X), np.array(y)
def R2_score(y_true, y_pred):

    from tensorflow.keras import backend as K

    SS_res = K.sum(K.square(y_true-y_pred))

    SS_tot = K.sum(K.square(y_true-K.mean(y_true)))

    return (1-SS_res/(SS_tot+K.epsilon()))
def LSTM_prediction_plot(timeseries):

    train_ts = timeseries[:int(0.90*len(timeseries))]

    test_ts = timeseries[int(0.90*len(timeseries)):]

    X_train, Y_train = split_sequence(train_ts,3,1)

    Y_train = np.squeeze(Y_train)

    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)

    # defining LSTM model

    model = Sequential()

    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(3, 1)))

    model.add(LSTM(100, activation='relu'))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=[R2_score])

    model.summary()

    training = model.fit(X_train,Y_train,epochs=10,verbose=0)

    history = list(train_ts)

    predictions = []

    for i in range(len(test_ts)):

        pred = np.squeeze(model.predict(np.array(history[-3:]).reshape(1,3,1)))

        history.append(float(pred))

        predictions.append(float(pred))

    plt.plot(timeseries)

    plt.plot(test_ts.index,predictions,color='green')

    plt.axvline(train_ts.index[-1],color='orange',dashes=(5,2,1,2))

    plt.xlabel('time')

    plt.ylabel('average price')

    plt.rcParams['figure.figsize'] = (25,10)
stock = cluster_0_N1[0]

print('For N1 cluster 0 stock: '+stock)

timeseries = bar_S_window_time[bar_S_window_time['symbol']==stock]['average_price']

LSTM_prediction_plot(timeseries)
stock = cluster_1_N1[0]

print('For N1 cluster 1 stock: '+stock)

timeseries = bar_S_window_time[bar_S_window_time['symbol']==stock]['average_price']

LSTM_prediction_plot(timeseries)
stock = cluster_2_N1[0]

print('For N1 cluster 2 stock: '+stock)

timeseries = bar_S_window_time[bar_S_window_time['symbol']==stock]['average_price']

LSTM_prediction_plot(timeseries)
stock = cluster_3_N1[0]

print('For N1 cluster 3 stock: '+stock)

timeseries = bar_S_window_time[bar_S_window_time['symbol']==stock]['average_price']

LSTM_prediction_plot(timeseries)
quote = quote.sort_values(by='time')

quote['date'] = pd.to_datetime(quote['time']).apply(lambda x: x.strftime('%Y-%m-%d'))
stock_list = list(set(quote['ticker']))

df_list = []

for i in stock_list:

    df = quote[quote['ticker']==i]

    df['bid_price_change'] = df['bid_price'].diff().values

    df['bid_price_returns'] = df['bid_price_change']/df['bid_price']

    df['ask_price_returns'] = df['ask_price'].diff().values/df['ask_price'].values

    average_bid_price = df['bid_price'].median()

    df['bid_price_volatility'] = (((df['bid_price'] - average_bid_price)**2)/len(df))**0.5 

    df_list.append(df)

quote = pd.concat(df_list)
df_list = []

window = 30

n_minutes = 10

date_list = list(set(quote['date']))

for i in stock_list:

    new_quote_S = quote[quote['ticker']==i]

    for j in date_list: 

        try:

            lower_limit = list(new_quote_S[new_quote_S['date']==j].sort_values(by='time')['time'])[0]

            lower_limit = pd.to_datetime(lower_limit)

            upper_limit = lower_limit + timedelta(minutes=window*n_minutes)

            df = new_quote_S[(new_quote_S['date']==j) & (new_quote_S['time']>=str(lower_limit)) & 

                             (new_quote_S['time']<str(upper_limit))]

            #Filling null values in records with 0 as default

            df['bid_price_change'] = df['bid_price_change'].fillna(0)

            df['bid_price_returns'] = df['bid_price_returns'].fillna(0)

            df['ask_price_returns'] = df['ask_price_returns'].fillna(0)

            df = df.sort_values(by='time')

            df['cumulative bid_price'] = np.cumsum(df['bid_price'])

            df_list.append(df)

        except Exception as e:

            pass

quote_S_windowed = pd.concat(df_list)

quote_S_windowed = quote_S_windowed.sort_index()

quote_S_windowed.index = quote_S_windowed['date']

quote_S_windowed.drop('date',axis=1,inplace=True)

quote_S_windowed.head()
bar_quote_data = quote_S_windowed.merge(bar_S,left_on=['ticker','time'],right_on=['symbol','time'],how='inner')
plt.title('cumulative bid price vs volume bubble chart based on ticker-wise bid size')

sns.scatterplot(data=bar_quote_data,x='volume',y='cumulative bid_price',hue='ticker',size='bid_size')
def plot_elbow(features):

    cm = plt.get_cmap('rainbow')

    date_list = list(set(quote_S_windowed.index))

    colors = cm(np.linspace(0, 1, len(date_list)))

    count=0

    plt.rcParams['figure.figsize'] = (10,10)

    for index in date_list: 

        quote_S_data = quote_S_windowed.loc[index,features]

        cluster_list = range(1,21)

        wss_list = []

        for i in cluster_list:

            km = KMeans(n_clusters=i)

            km.fit(quote_S_data)

            wss_list.append(km.inertia_)

        plt.plot(cluster_list,wss_list,marker='o',color=colors[count])

        plt.xlabel('No. of clusters')

        plt.ylabel('Within sum of squares (WSS)')

        plt.title('Elbow Curve')

        plt.legend(date_list)

        count+=1

        for i, txt in enumerate(cluster_list):

            plt.annotate(txt, (cluster_list[i], wss_list[i]))
returns_features = ['bid_price_returns','ask_price_returns']

plot_elbow(returns_features)
km = KMeans(n_clusters=2)

quote_S_returns_data = quote_S_windowed.loc[:,returns_features]

km.fit(quote_S_returns_data)

label = km.predict(quote_S_returns_data)
stock_cluster = dict(zip(quote_S_windowed.loc[:,'ticker'],label))

cluster_0_N2=[]

cluster_1_N2=[]

for key in stock_cluster:

    if stock_cluster[key]==0:

        cluster_0_N2.append(key)    

    else:

        cluster_1_N2.append(key)

print('Stocks belonging to cluster 0:',','.join(set(cluster_0_N2)))

print('\nStocks belonging to cluster 1:',','.join(set(cluster_1_N2)))
bid_size_feature = ['bid_size']

plot_elbow(bid_size_feature)
km = KMeans(n_clusters=3)

quote_S_size_data = quote_S_windowed.loc[:,bid_size_feature]

km.fit(quote_S_size_data)

label = km.predict(quote_S_size_data)
stock_cluster = dict(zip(quote_S_windowed.loc[:,'ticker'],label))

cluster_0_N3=[]

cluster_1_N3=[]

cluster_2_N3=[]

for key in stock_cluster:

    if stock_cluster[key]==0:

        cluster_0_N3.append(key)    

    elif stock_cluster[key]==1:

        cluster_1_N3.append(key)

    else:

        cluster_2_N3.append(key)

cluster_0_N3 = list(set(cluster_0_N3))

cluster_1_N3 = list(set(cluster_1_N3))

cluster_2_N3 = list(set(cluster_2_N3))

print('Stocks belonging to cluster 0:',','.join(cluster_0_N3))

print('\nStocks belonging to cluster 1:',','.join(cluster_1_N3))

print('\nStocks belonging to cluster 2:',','.join(cluster_2_N3))
def plot_bid_curve(cluster,n_stocks):

    count = 1

    cluster = list(set(cluster))

    plt.rcParams['figure.figsize'] = (20,20)

    for stock in cluster[:n_stocks]: 

        df = quote_S_windowed[['time','bid_price_change','ticker','bid_size','bid_price_volatility','cumulative bid_price']]

        df = df[df['ticker']==stock].sort_values(by='time')

        plt.subplot(n_stocks,4,count)

        plt.plot(list(df['bid_price_change']))

        plt.xlabel('time')

        plt.ylabel('bid price change')

        plt.title('For stock '+stock)

        count+=1

        plt.subplot(n_stocks,4,count)

        plt.plot(list(df['bid_size']))

        plt.xlabel('time')

        plt.ylabel('bid size')

        plt.title('For stock '+stock)

        count+=1

        plt.subplot(n_stocks,4,count)

        plt.plot(list(df['bid_price_volatility']))

        plt.xlabel('time')

        plt.ylabel('bid price volatility')

        plt.title('For stock '+stock)

        count+=1

        plt.subplot(n_stocks,4,count)

        plt.plot(list(df['cumulative bid_price']))

        plt.xlabel('time')

        plt.ylabel('cumulative bid price')

        plt.title('For stock '+stock)

        count+=1
print('Cluster 0 bid plot:')

plot_bid_curve(cluster_0_N3,4)
print('Cluster 1 of N3 bid plot:')

plot_bid_curve(cluster_1_N3,4)
print('Cluster 2 of N3 bid plot:')

plot_bid_curve(cluster_2_N3,4)