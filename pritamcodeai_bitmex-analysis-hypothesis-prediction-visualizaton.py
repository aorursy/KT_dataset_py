import pandas as pd

import cufflinks as cf

import datetime

import numpy as np

import pandas as pd

import keras.callbacks

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.recurrent import LSTM

from keras.models import Sequential, load_model

from sklearn import preprocessing

from datetime import timedelta

import time

from collections import Counter

import os

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

%matplotlib inline

#resize charts to fit screen if using Jupyter Notebook

plt.rcParams['figure.figsize']=[15,5]

stocks = pd.read_csv("../input/bitmex-btcusd-candles-1m-price/bitmex_BTCUSD_candles_1m_price.csv", index_col= [0], parse_dates= [0]).Close
print(type(stocks))

print(stocks[0])

stocks = stocks.resample('D').mean()

stocks.head()
norm = stocks.div(stocks[0]).mul(100)
norm.head(10)
cf.set_config_file(offline = True)
cf.go_offline()
norm.iplot()
norm.head()
norm.iplot(kind = "line", fill = True)
cf.colors.scales()
norm.iplot(kind = "line", fill = True, colorscale= "reds")
cf.getThemes()
norm.iplot(kind = "line", fill = True, colorscale= "rdylbu", theme= "solar", 

             title= "bitmex_BTC", xTitle= "Time", yTitle= "Stock Price")
stocks.head()
ret = stocks.pct_change()

ret = ret.dropna()

ret
ret.head()
ret.iplot(kind = "histogram", bins = (-0.15, 0.1, 0.001), histnorm= "percent")
bitmex_BTC = pd.read_csv("../input/bitmex-btcusd-candles-1m-price/bitmex_BTCUSD_candles_1m_price.csv", index_col= [0], parse_dates= [0])
bitmex_BTC = bitmex_BTC[['Open', 'High', 'Low', 'Close']]
bitmex_BTC = bitmex_BTC.resample('D').mean()

bitmex_BTC.index = np.array([datetime.date(*date_tuple) for date_tuple in zip(bitmex_BTC.index.year, bitmex_BTC.index.month, bitmex_BTC.index.day)])

bitmex_BTC.head()
bitmex_BTC.loc[pd.to_datetime("2018-01-01").date():pd.to_datetime("2018-01-26").date()].iplot(kind= "candle")

time.sleep(600)
qf = cf.QuantFig(df = bitmex_BTC.loc[pd.to_datetime("2018-01-01").date():pd.to_datetime("2018-01-26").date()])
type(qf)
qf.iplot(title = "bitmex_BTC", name = "bitmex_BTC")

time.sleep(600)
qf = cf.QuantFig(df = bitmex_BTC.loc[pd.to_datetime("2018-01-01").date():pd.to_datetime("2018-01-26").date()])
qf.add_bollinger_bands(periods = 10, boll_std= 2)

qf.add_sma(periods = 10)

qf.add_macd()
qf.iplot(title = "bitmex_BTC", name = "bitmex_BTC")

time.sleep(600)
df = stocks

df.head()
#To model returns we will use daily % change

ret = df.resample('D').mean()

daily_ret = ret.pct_change()

#drop the 1st value - nan

daily_ret.dropna(inplace=True)

#daily %change

daily_ret.head()
print(daily_ret.describe())

return_for_chart = daily_ret.describe()

print(type(return_for_chart))
print(return_for_chart.loc["mean"])

print(return_for_chart.loc["std"])

yearly_check = [""]

yearly_check_mean = return_for_chart.loc["mean"]*252

yearly_check_std = return_for_chart.loc["std"] * np.sqrt(252)

print("yearly_check_mean = ",yearly_check_mean)

print("yearly_check_std = ",yearly_check_std)

yearly_check = pd.DataFrame({"stock":["bitmex_BTC"],"mean":[yearly_check_mean],"std":[yearly_check_std]}).set_index("stock")

yearly_check
yearly_check.plot(kind = "scatter", x = "std", y = "mean", figsize = (15,12), s = 1000, fontsize = 26)

for i in yearly_check.index:

    plt.annotate(i, xy=(yearly_check.loc[i, "std"]+0.002, yearly_check.loc[i, "mean"]+0.002), size = 26)

plt.xlabel("ann. Risk(std)", fontsize = 100)

plt.ylabel("ann. Return", fontsize = 100)

plt.title("Risk/Return", fontsize = 100)

plt.show()
#use pandas to resample returns per month and take Standard Dev as measure of Volatility

#then annualize by multiplying by sqrt of number of periods (12)

mnthly_annu = daily_ret.resample('M').std()* np.sqrt(12)



print(mnthly_annu.head())

#we can see major market events show up in the volatility

plt.plot(mnthly_annu)

plt.axvspan('2-2018','3-2018',color='r',alpha=.5)

plt.axvspan('12-2018','1-2019',color='r',alpha=.5)

plt.axvspan('7-2019','8-2019',color='r',alpha=.5)

plt.title('Monthly Annualized volatility')

labs = mpatches.Patch(color='red',alpha=.5, label="more volatility")

plt.legend(handles=[labs])
#for each year rank each month based on volatility lowest=1 Highest=12

ranked = mnthly_annu.groupby(mnthly_annu.index.year).rank()



#average the ranks over all years for each month

final = ranked.groupby(ranked.index.month).mean()



final.describe()
#the final average results over 2 years

final
#plot results for ranked bitmex data volatility

#clearly november has the highest AMVR

#and october has the lowest

#mean of 5.62 is plotted



b_plot = plt.bar(x=final.index,height=final)

b_plot[10].set_color('g')

b_plot[9].set_color('r')

for i,v in enumerate(round(final,2)):

    plt.text(i+.8,1,str(v), color='black', fontweight='bold')

plt.axhline(final.mean(),ls='--',color='k',label=round(final.mean(),2))

plt.title('Average Monthly Volatility Ranking bitmex data')



plt.legend()

plt.show()

#take abs value move from the mean

#we see october and november are the biggest abs moves



fin = abs(final - final.mean())

print(fin.sort_values())

november_value = fin[11]

october_value = fin[10]

print('Extreme october value:', october_value)

print('Extreme november value:', november_value)
#as Null is that no seasonality exists or alternatively that the 

#month does not matter in terms of AMVR,

#it can be shuffled 'date' labels

#for simplicity, I will shuffle the 'daily' return data, 

#which has the same effect as shuffling 'month' labels



#generate null data



new_df_sim = pd.DataFrame()

highest_only = []



count=0

n=300

for i in range(n):

    #sample same size as dataset, drop timestamp

    daily_ret_shuffle = daily_ret.sample(588).reset_index(drop=True)

    #add new timestamp to shuffled data

    daily_ret_shuffle.index = (pd.bdate_range(start='2017-12-31',periods=588))



    #then follow same data wrangling as before...

    mnthly_annu = daily_ret_shuffle.resample('M').std()* np.sqrt(12)



    ranked = mnthly_annu.groupby(mnthly_annu.index.year).rank()

    sim_final = ranked.groupby(ranked.index.month).mean()

    #add each of 1000 sims into df

    new_df_sim = pd.concat([new_df_sim,sim_final],axis=1)



    #also record just highest AMVR for each year 

    maxi_month = max(sim_final)

    highest_only.append(maxi_month)







#calculate absolute deviation in AMVR from the mean

all_months = new_df_sim.values.flatten()

mu_all_months = all_months.mean()

abs_all_months = abs(all_months-mu_all_months)    



#calculate absolute deviation in highest only AMVR from the mean

mu_highest = np.mean(highest_only)

abs_highest = [abs(x - mu_all_months) for x in highest_only]
#count number of months in sim data where ave-vol-rank is >= october

#Note: we are using october not november, as october 

#has highest absolute deviation from the mean

count=0

for i in abs_all_months:

    if i> october_value:

        count+=1

ans = count/len(abs_all_months)        

print('p-value:', ans )
#same again but just considering highest AMVR 

count=0

for i in abs_highest:

    if i> october_value:

        count+=1

ans = count/len(abs_highest)        

print('p-value:', ans)
abs_all_months_92 = np.quantile(abs_all_months,0.92)

abs_highest_92 = np.quantile(abs_highest,.92)



fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,sharex='col',figsize=(20,20))





#plot 1

ax1.hist(abs_all_months,histtype='bar',color='#42a5f5')

ax1.set_title('AMVR all months',fontsize=30)

ax1.set_ylabel('Frequency',fontsize=20)

ax3.hist(abs_all_months,density=1,histtype='bar',cumulative=True,bins=30,color='#42a5f5')

ax3.set_ylabel('Cumulative probability',fontsize=20)

ax1.axvline(october_value,color='b',label='october Result',lw=10)

ax3.axvline(october_value,color='b',lw=10)

ax3.axvline(abs_all_months_92,color='r',ls='--',label='8% Sig level',lw=10)





#plot2

ax2.hist(abs_highest,histtype='bar',color='g')

ax2.set_title('AMVR highest only',fontsize=30)

ax2.axvline(october_value,color='b',lw=10)

ax4.hist(abs_highest,density=1,histtype='bar',cumulative=True,bins=30,color='g')

ax4.axvline(october_value,color='b',lw=10)

ax4.axvline(abs_highest_92,color='r',ls='--',lw=10)



ax1.legend(fontsize=15)

ax3.legend(fontsize=15)
#for normalizing data

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

bitmex_BTC.index 
"""

#sorting

data = bitmex_BTC.sort_index(ascending=True, axis=0)

data.insert(0, 'Date', bitmex_BTC.index)

data.head()

"""
bitmex_data_detail_of_close_column = bitmex_BTC[['Close']]
bitmex_data_detail_of_close_column.head()
def bitmex_data_load(df, lookback=25):

    ''' scale data and split into training/test sets '''

    data = df.values

    n_train = list(df.index).index(df.index[-1]+timedelta(-365))

    scaler = preprocessing.StandardScaler() #normalize mean-zero, unit-variance

    scaler.fit(data[:n_train,:])

    data = scaler.transform(data)

    dataX, dataY = [], []

    for timepoint in range(data.shape[0]-lookback):

        dataX.append(data[timepoint:timepoint+lookback,:])

        dataY.append(data[timepoint+lookback,0])

    X_train, X_test = dataX[:n_train], dataX[n_train:]

    y_train, y_test = dataY[:n_train], dataY[n_train:]

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), scaler



# load practice stock, AAPL

lookback = 25

X_train, y_train, X_test, y_test, scaler = bitmex_data_load(df=bitmex_data_detail_of_close_column, lookback=lookback)
def bitmex_data_better_ax(ax):

    ''' better axis '''

    for spine in ax.spines.values():

        spine.set_visible(False)

    ax.set_frameon=True

    ax.patch.set_facecolor('#eeeeef')

    ax.grid('on', color='w', linestyle='-', linewidth=1)

    ax.tick_params(direction='out')

    ax.set_axisbelow(True)

    

def bitmex_data_easy_ax(figsize=(6,4), **kwargs):

    ''' easy axis '''

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111, **kwargs)

    bitmex_data_better_ax(ax)

    return fig, ax



def bitmex_data_inverse_price_transform(normalized_data, scaler):

    ''' inverse from normalized price to raw price '''

    m = scaler.mean_[0]

    s = scaler.scale_[0]

    return s*np.array(normalized_data)+m
# denormalize training and test price data and plot

print("{} training examples, {} test examples".format(len(y_train), len(y_test)))



f,a = bitmex_data_easy_ax(figsize=(10,6))

a.plot(range(len(y_train)), bitmex_data_inverse_price_transform(y_train, scaler), c='b', label='Training Data')

a.plot(range(len(y_train),len(y_test)+len(y_train)), bitmex_data_inverse_price_transform(y_test, scaler), c='r', label='Test Data')

a.set_title('bitmex stock price data of close column')

a.set_xlabel('Day')

a.set_ylabel('Closing price')

plt.legend()

plt.show()
# build model

model = Sequential()

model.add(LSTM(128, input_shape=(X_train.shape[1],X_train.shape[2]), return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(128, return_sequences=False))

model.add(Dropout(0.2))

#model.add(Dense(32, kernel_initializer="uniform", activation='relu'))        

model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop')

model.summary()
# fit model

t0 = time.time()

history = model.fit(

            X_train,

            y_train,

            batch_size=33,

            epochs=200,

            validation_split=0.05,

            verbose=1)

print("TRAINING DONE. {} seconds to train.".format(int(time.time()-t0)))

# plot loss and validation loss

f,a = bitmex_data_easy_ax(figsize=(10,6))

a.plot(history.history['loss'], label='loss')

a.plot(history.history['val_loss'], label='val_loss')

a.set_title('Training Losses')

a.set_xlabel('Epoch')

a.set_ylabel('MSE Loss')

plt.legend()

plt.show()
##### SAVE MODEL #####

model.save('bitmex_trained_model.h5')
#### LOAD MODEL #####

model = load_model('bitmex_trained_model.h5')
# predict test set

predictions = model.predict(X_test)

print("RMSE: ", np.sqrt(np.mean((predictions-y_test)**2)))



f, a = bitmex_data_easy_ax(figsize=(10,6))

a.plot(predictions, c='b', label='predictions')

a.plot(y_test, c='r', label='actual')

a.set_ylabel('Normalized closing price')

a.set_xlabel('Day')

a.set_title('bitmex test set predictions')

plt.legend()

plt.show()
def bitmex_data_predict_days(startday, days_topredict, data, model):

    ''' starting from startday predict days_topredict stock prices '''

    curr_data = data[startday,:,:]

    predictions = []

    for day in range(days_topredict):

        prediction = model.predict(curr_data.reshape(1,curr_data.shape[0],curr_data.shape[1]))[0][0]

        predictions.append(prediction)

        new_row = curr_data[-1,:]

        new_row[0] = prediction

        curr_data = np.vstack((curr_data[1:,:], new_row))

    return predictions
# predict multiple days in future and plot price projection path

days = 100

f, a = bitmex_data_easy_ax(figsize=(10,6))

a.plot(bitmex_data_inverse_price_transform(y_test,scaler), c='k')

for segment in range(int(len(y_test)/days)):

    predictions = bitmex_data_predict_days(segment*days, days, X_test, model)

    a.plot(range(segment*days, segment*days+days), bitmex_data_inverse_price_transform(predictions, scaler))

    a.axvline(segment*days, c='k', linestyle='dashed', linewidth=1)

    a.axvline(segment*days+days, c='k', linestyle='dashed', linewidth=1)

a.set_xlabel('Day')

a.set_ylabel('Price')

a.set_title('bitmex test set 100 day lookahead')

plt.show()
def decision_on_bitmex_stock_buy_sell(startpoint, days_topredict, data, model, return_threshold):

    '''

    predict future prices and return a market decision

    - returns True: "buy long"

    - returns False: "sell short"

    - returns None: "do nothing"

    '''

    predictions = bitmex_data_predict_days(startpoint, days_topredict, data, model)

    startprice, maxprice, minprice = predictions[0], max(predictions), min(predictions)

    buyreturn = (maxprice-startprice)/startprice

    sellreturn = (startprice-minprice)/startprice

    if buyreturn>=sellreturn and buyreturn>=return_threshold:

        return True

    elif sellreturn>buyreturn and sellreturn>=return_threshold:

        return False

    return None



def bitmex_data_walk_buy_sell(data, model, return_threshold=.06, days_topredict=26):

    ''' walk data making buy/sell decisions '''

    buy_dates, sell_dates = [], []

    for t in range(len(y_test)):

        decision = decision_on_bitmex_stock_buy_sell(t, days_topredict, data, model, return_threshold)

        if decision is True:

            buy_dates.append(t)

        elif decision is False:

            sell_dates.append(t)

        if t%62==0:

            print("{}/{} timepoints calculated.".format(t+1,len(y_test)))

    print("Data walk complete.")

    return buy_dates, sell_dates



buy_dates, sell_dates = bitmex_data_walk_buy_sell(X_test, model, return_threshold=0.62, days_topredict=26)
# plot buy/sell timepoint decisions

f,a = bitmex_data_easy_ax(figsize=(10,6))

a.plot(bitmex_data_inverse_price_transform(y_test, scaler), c='k')

a.scatter(buy_dates, bitmex_data_inverse_price_transform(y_test[buy_dates],scaler), c='b')

a.scatter(sell_dates, bitmex_data_inverse_price_transform(y_test[sell_dates],scaler), c='y')

a.set_xlabel('Day')

a.set_ylabel('Price')

a.set_title('Buy/Sell Decisions for bitmex Test Set')

recs = [mpatches.Rectangle((0,0),1,1,fc='b'), mpatches.Rectangle((0,0),1,1,fc='y')]

a.legend(recs,['buy', 'sell'], loc=2, prop={'size':14})

plt.show()
# simulate portfolio value using buy/sell decisions

init_value = 1000000

stocks_per_trade = 5

cash = init_value

portfolio = Counter()

returns = [0]

for date in range(max(buy_dates+sell_dates)+1):

    if date in buy_dates: #buy

        portfolio['bitmex'] += stocks_per_trade

        cash = cash - stocks_per_trade*bitmex_data_inverse_price_transform(y_test[date], scaler)

    elif date in sell_dates: #sell

        portfolio['bitmex'] -= stocks_per_trade

        cash = cash + stocks_per_trade*bitmex_data_inverse_price_transform(y_test[date], scaler)

    curr_value = cash + portfolio['bitmex']*bitmex_data_inverse_price_transform(y_test[date],scaler)

    curr_return = 100*(curr_value - init_value)/init_value

    returns.append(curr_return)

    

f,a = bitmex_data_easy_ax(figsize=(10,6))

a.plot(returns, linewidth=2)

a.set_xlabel('Day')

a.set_ylabel('Portfolio Percent Return')

a.set_title('Portfolio Value Over Time Trading of bitmex BTC on Test Set')

plt.show()