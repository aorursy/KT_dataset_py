# Basic libaries

import numpy as np

from datetime import datetime as dt, timedelta as td

# Data manipulation

import pandas as pd

import csv

# Modelling

import statsmodels.api as sm

import sklearn as sk

from sklearn.preprocessing import MinMaxScaler

import scipy as sp

import tensorflow as tf

# Visualizations

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

matplotlib.use('SVG')
# Changing Frequency

freq=td(hours=1)

# Choose missing_data_handling to be True to fill missing data points with a concave function and False to remove missing data points.

missing_data_handling=False
import os

print(os.listdir("../input"))
# reading in stock index date

prices=pd.read_csv("../input/twitter-investor-sentiment-analysis-dataset/prices_60m.csv",parse_dates=['Dates'])

prices.rename(columns={'Dates':'dates'},inplace=True)

prices=prices.set_index('dates')

prices=prices.dropna(how='any')

prices



if missing_data_handling:

    prices=prices.resample(freq,base=15.5).asfreq()

    

# calculating logreturns and Z Scores

def logreturn(x):

    return np.log(x/x.shift(1))



def zscore(x):

    return (x-x.mean())/x.std()



prices['DJIA LogRet']=logreturn(prices['DJIA CLOSE'])

prices['SP500 LogRet']=logreturn(prices['SP500 CLOSE'])

prices['DJIA Z Scores']=zscore(prices['DJIA CLOSE'])

prices['SP500 Z Scores']=zscore(prices['SP500 CLOSE'])

prices['DJIA Log']=np.log(prices['DJIA CLOSE'])

prices['SP500 Log']=np.log(prices['SP500 CLOSE'])

prices=prices.dropna(how='any')

prices
# Reading in csv file containing tweets and converting it to a pandas dataframe

tweets=pd.read_csv("../input/twitter-investor-sentiment-analysis-dataset/Raw_tweets_09_30.csv",parse_dates=['created_at'])

tweets = tweets.drop(columns="id")



# Converting time from UTC to CEST +2 hours

cest_date=[]

for date in tweets['created_at']:

    cest_date.append(date+td(hours=2))

tweets['created_at']=cest_date



# Changing index to Datetimeindex

tweets.rename(columns={'created_at':'date'},inplace=True)

tweets=tweets.set_index('date')

#tweets.index = pd.DatetimeIndex(tweets.index)



tweets
len(tweets)
tweets['Time']=tweets.index.hour
# Calculating average number of tweets per day

df=tweets.groupby(tweets.index.date).count()

df['Total']=df['tweet_text']+df['bullish']

df['Total'].mean()
#tweets['Time'].value_counts().sort_index().plot.bar()

fig, ax = plt.subplots()

counts, bins, patches = ax.hist(tweets.Time, facecolor='gray', edgecolor='black', bins=range(0,25))

ax.set_xticks(bins)

ax.set_title('Tweetek óránkénti eloszlása', fontsize=9)

ax.tick_params(axis='both', which='major', labelsize=7)

plt.savefig("hourly_tweet_distribution.svg", format="svg")

print(counts)

print(bins)
if not missing_data_handling:

    # Removing tweets that were created on weekends and bank holidays

    #getting trading dates from DJIA intraday prices

    t_dates=prices.index.map(pd.Timestamp.date).unique()

    str_t_dates=[str(x) for x in t_dates]

    tweets=tweets[tweets.index.floor('D').isin(str_t_dates)]

    # Removing tweets that are not in trading analysis interval

    start=str(td(hours=15,minutes=30)-freq)

    end=str(td(hours=21,minutes=29))

    tweets = tweets.between_time(start, end, include_start=True, include_end=True)
# Separating sentiment tweets from bullish/bearish index tweets

text_tweets=tweets.loc[:, ['tweet_text']]

market_tweets=tweets.loc[:, ['bullish']]
# Filtering out NaNs

fmarket_tweets=market_tweets[market_tweets['bullish'].notnull()]

ftext_tweets=text_tweets[text_tweets['tweet_text'].notnull()]

ftext_tweets =ftext_tweets.drop_duplicates(subset="tweet_text",keep=False)

print('Total number of observations including both text and bullish/bearish tweets:',ftext_tweets.tweet_text.count()+fmarket_tweets.bullish.count())

ftext_tweets
# Filtering out tweets containing a link

#ftext_tweets=ftext_tweets[~ftext_tweets['tweet_text'].str.contains("https")]
text = " ".join(tweet for tweet in ftext_tweets.tweet_text)

print ('There are %s words in the combination of all review.' % (len(text)))
# Create stopword list:

stopwords = set(STOPWORDS)

stopwords_raw=["one","co","amp","https","bullish", "bearish", "stock market", "I think economy", "I feel economy", "I am feeling economy", "It feels economy","inflation", "unemployment rate", "recession","SP500","S&P500","DJIA","Dow jones"]

stopwords_splited=[]

for phrase in stopwords_raw:

    split=phrase.split()

    stopwords_splited=list(set(stopwords_splited+split))

stopwords.update(stopwords_splited)



# Generate a word cloud image

wordcloud = WordCloud(stopwords=stopwords, background_color="white",width=800, height=400).generate(text)



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()

fig = plt.gcf() #get current figure

#fig.set_size_inches(10,10)

#plt.savefig("tweets_word_cloud.png", format="png")
pd.set_option('display.max_colwidth', -1)

ftext_tweets[ftext_tweets['tweet_text'].str.contains("now")].sample(10)
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

#nltk.download('vader_lexicon')



sia=SentimentIntensityAnalyzer()
del sia.lexicon['thank']

del sia.lexicon['thanks']
stock_lex = pd.read_csv('VADER +/stock_lex.csv')

stock_lex['sentiment'] = (stock_lex['Aff_Score'] + stock_lex['Neg_Score'])/2

stock_lex = dict(zip(stock_lex.Item, stock_lex.sentiment))

#filtering out multiple words expressions from the lexicon

stock_lex = {k:v for k,v in stock_lex.items() if len(k.split(' '))==1}

stock_lex_scaled = {}

for k, v in stock_lex.items():

    if v > 0:

        stock_lex_scaled[k] = v / max(stock_lex.values()) * 4

    else:

        stock_lex_scaled[k] = v / min(stock_lex.values()) * -4
positive = []

with open('VADER +/lm_positive.csv', 'r') as f:

    reader = csv.reader(f)

    for row in reader:

        positive.append(row[0].strip())

        

negative = []

with open('VADER +/lm_negative.csv', 'r') as f:

    reader = csv.reader(f)

    for row in reader:

        entry = row[0].strip().split(" ")

        if len(entry) > 1:

            negative.extend(entry)

        else:

            negative.append(entry[0])
final_lex = {}

final_lex.update({word:2.0 for word in positive})

final_lex.update({word:-2.0 for word in negative})

final_lex.update(stock_lex_scaled)

final_lex.update(sia.lexicon)

sia.lexicon = final_lex
sentiment=[]

for tweet in ftext_tweets['tweet_text']:

    sentiment.append(sia.polarity_scores(tweet)['compound'])

ftext_tweets['Sentiment Score']=sentiment
pd.set_option('display.max_colwidth', -1)

ftext_tweets[ftext_tweets['tweet_text'].str.len()<150].sample(10)
# Downsample the tweet sentiment scores

iss = ftext_tweets['Sentiment Score'].resample(freq,base=15.5,label='right').mean().to_frame()

iss = iss[iss['Sentiment Score'].notnull()]

iss['Z Scores']=(iss['Sentiment Score']-iss['Sentiment Score'].mean())/iss['Sentiment Score'].std()
iss
fmarket_tweets
bbi=pd.DataFrame()

bbi['nbull']=(fmarket_tweets['bullish'] == True).resample(freq,base=15.5,label='right').sum()

bbi['nbear']=(fmarket_tweets['bullish'] == False).resample(freq,base=15.5,label='right').sum()

bbi['BBI']=np.log(bbi['nbull']/bbi['nbear'])

bbi = bbi[bbi['BBI'].notnull()]

bbi['Z Scores']=(bbi['BBI']-bbi['BBI'].mean())/bbi['BBI'].std()
#Calculating rolling window Z Scores
# Defining columns that are used in the analysis

djiaval=prices['DJIA Z Scores']

spval=prices['SP500 Z Scores']

bbival=bbi['Z Scores']

issval=iss['Z Scores']



timeseries=pd.DataFrame({'DJIA': djiaval, 'SP500': spval,'BBI': bbival,'ISS': issval})

timeseries=timeseries.dropna(how='any')



bbival=timeseries['BBI']

issval=timeseries['ISS']

djiaval=timeseries['DJIA']

spval=timeseries['SP500']
df_graphs=timeseries.reset_index()

df_graphs.rename(columns={'index':'dates'},inplace=True)

df_graphs
# Visualizing DJIA and SP500 prices

daily=pd.read_csv("Input data\Import\prices_daily.csv",parse_dates=["Dates"])



fig, axs = plt.subplots(1,2,figsize=(20,5))



xlabel=[0]+list(daily[daily.index.isin([0,10,20,30,40,50,60,70])]['Dates'].dt.date)



axs[0].plot(daily['DJIA CLOSE'],color='blue', label='DJIA')

axs[1].plot(daily['SP500 CLOSE'],color='blue', label='SP500')

axs[0].set_title('DJIA')

axs[1].set_title('S&P 500')

axs[0].set_xticklabels(xlabel)

axs[1].set_xticklabels(xlabel)

#plt.savefig("DJIA_SP500_hist_price.svg",format="svg")
fig, axs = plt.subplots(2,figsize=(18,10))



xlabel=[0]+list(df_graphs[df_graphs.index.isin([0,50,100,150,200,250,300,350])]['dates'].dt.date)



axs[0].plot(df_graphs['BBI'],color='green', label='BBI',alpha=0.5)

axs[0].plot(df_graphs['ISS'],color='black', label='ISS')

axs[0].plot(df_graphs['DJIA'],color='red', label='DJIA')

axs[0].set_title('Z Scores')

axs[0].legend()

axs[0].set_xticklabels(xlabel)

axs[1].plot(df_graphs['BBI'],color='green', label='BBI',alpha=0.5)

axs[1].plot(df_graphs['ISS'],color='black', label='ISS')

axs[1].plot(df_graphs['SP500'],color='red', label='S&P500')

axs[1].set_title('Z Scores')

axs[1].legend()

axs[1].set_xticklabels(xlabel)

plt.savefig("Z-Scores.svg", format="svg")
def diffs(ts):

    diffs=(ts-ts.shift(1))**2

    diffs=diffs[1:]

    return (diffs.sum()/diffs.count())**(1/2)



print(diffs(bbival))

print(diffs(issval))

print(diffs(djiaval))

print(diffs(spval))

print(np.var(iss['Sentiment Score']))
print(np.var(bbival)/bbival.mean())

print(np.var(issval)/issval.mean())

print(np.var(djiaval)/djiaval.mean())

print(np.var(spval)/spval.mean())
c=timeseries.corr()

#c.to_csv("correlation_matrix.csv")

c
pd.plotting.scatter_matrix(timeseries, figsize=(6, 6))

plt.savefig("Scatter_Matrix.svg", format="svg")
prices['DJIA Z Scores'].hist()
testprices=pd.read_csv("DJI.csv",parse_dates=['Date'])

testprices['Close Z']=zscore(testprices['Close'])

testprices['Close Z'].hist()
variables=[djiaval,spval,bbival,issval]

pvals=np.zeros((len(variables),len(variables)))

for i,n in zip(variables,range(0,len(variables))):

    for j,k in zip(variables,range(0,len(variables))):

        corr, p = sp.stats.pearsonr(i,j)

        pvals[n][k]=p

#np.savetxt("Correlation Coefficients P values.csv", pvals, delimiter=",")

pd.DataFrame(pvals)
# Plotting the Autocorrelation Function (ACF) for the indices

fig, axs = plt.subplots(1,2,figsize=(15,5))

print(sm.graphics.tsa.plot_acf(djiaval,lags=50,ax=axs[0],title="DJIA"))

print(sm.graphics.tsa.plot_acf(spval,lags=50,ax=axs[1],title="S&P 500"))

#plt.savefig("Correlograms.svg", format="svg")
lbvals,pvals=sm.stats.diagnostic.acorr_ljungbox(djiaval,4)

nlags=5

pd.DataFrame({'DJIA p-values':sm.stats.diagnostic.acorr_ljungbox(djiaval,nlags)[1], 'SP500 p-values':sm.stats.diagnostic.acorr_ljungbox(spval,nlags)[1]},index=range(1,nlags+1))
fig, axs = plt.subplots(5,figsize=(10,30))



lr=sk.linear_model.LinearRegression()

        

def plotlinreg(x,y,n):

    X=x.values.reshape(-1,1)

    Y=y.values.reshape(-1,1)

    lr.fit(X,Y)

    axs[n].scatter(X,Y,color='black')

    axs[n].plot(X,lr.predict(X),color='red')



plotlinreg(bbival,djiaval,0)

axs[0].set_title('BBI - DJIA')

plotlinreg(bbival,spval,1)

axs[1].set_title('BBI - SP500')

plotlinreg(issval,djiaval,2)

axs[2].set_title('ISS - DJIA')

plotlinreg(issval,spval,3)

axs[3].set_title('ISS - SP500')

plotlinreg(issval,bbival,4)

axs[4].set_title('ISS - BBI')



#plt.savefig("Linear_Regressions.svg", format="svg")
def crosscorr(x,y,lag=0):

    return x.corr(y.shift(lag))

rs=[]

lag=70

f,ax=plt.subplots(2,figsize=(14,8))



# DJIA BBI

rs=[crosscorr(djiaval,bbival,lag) for lag in range(-lag,lag+1)]

offset = np.ceil(len(rs)/2)-np.argmax(rs)

ax[0].plot(rs,color='blue',label='DJIA')

ax[0].axvline(lag,color='k',linestyle='-',label='Center')

ax[0].axvline(np.argmax(rs),color='r',linestyle='--')



# DJIA ISS

rs=[crosscorr(djiaval,issval, lag) for lag in range(-lag,lag+1)]

offset = np.ceil(len(rs)/2)-np.argmax(rs)

ax[1].plot(rs,color='green',label='DJIA')

ax[1].axvline(lag,color='k',linestyle='-',label='Center')

ax[1].axvline(np.argmax(rs),color='r',linestyle='--')



#SP500 BBI

rs=[crosscorr(spval,bbival, lag) for lag in range(-lag,lag+1)]

offset = np.ceil(len(rs)/2)-np.argmax(rs)

ax[0].plot(rs,color='orange',label='SP500')

ax[0].axvline(np.argmax(rs),color='r',linestyle='--')

print(np.argmax(rs)-70)



#SP500 ISS

rs=[crosscorr(spval,issval, lag) for lag in range(-lag,lag+1)]

offset = np.ceil(len(rs)/2)-np.argmax(rs)

ax[1].plot(rs,color='purple',label='SP500')

ax[1].axvline(np.argmax(rs),color='r',linestyle='--')

print(np.argmax(rs)-70)



tick_range=range(0,2*lag+1,10)

label_range=range(-lag,lag+1,10)

ax[0].set_xticks(tick_range)

ax[0].set_xticklabels(label_range)

ax[1].set_xticks(tick_range)

ax[1].set_xticklabels(label_range)

ax[0].set_title('BBI')

ax[1].set_title('ISS')

ax[0].legend()

ax[1].legend()

#plt.savefig("TLCC.svg", format="svg")
crosscorr(djiaval,issval,34)
class StationarityTests:

    def __init__(self, significance=.05):

        self.SignificanceLevel = significance

        self.pValue = None

        self.isStationary = None

    def ADF_Stationarity_Test(self, tslist):

        pValues=[]

        adf=pd.DataFrame()

        names=[]

        #Dickey-Fuller test:

        for ts in tslist:

            adfTest = sm.tsa.stattools.adfuller(ts, autolag='AIC')

            self.pValue = adfTest[1]

            pValues.append(self.pValue)

            names.append(ts.name)

        print(names)

        adf['P values']=pValues

        adf=adf.set_index(pd.Index(names))

        return adf

    

    def KPSS_Stationarity_Test(self, tslist):

        pValues=[]

        kpss=pd.DataFrame()

        names=[]

        #Dickey-Fuller test:

        for ts in tslist:

            kpssTest = sm.tsa.stattools.kpss(ts, regression='c', store=False)

            self.pValue = kpssTest[1]

            pValues.append(self.pValue)

            names.append(ts.name)

        kpss['P values']=pValues

        kpss=kpss.set_index(pd.Index(names))

        return kpss
tslist=[djiaval,spval,bbival,issval]

sTest = StationarityTests()

sTest.ADF_Stationarity_Test(tslist)
def diff(data):

    diff_var=data[1:]-data.shift(1)[1:]

    return diff_var



sTest.ADF_Stationarity_Test([diff(djiaval),diff(spval),diff(bbival),diff(issval)])
sTest.KPSS_Stationarity_Test([diff(djiaval),diff(spval),bbival,diff(bbival),diff(issval)])
granger_ts=pd.DataFrame({'dDJIA':diff(djiaval),'dSP500':diff(spval),'dBBI':diff(bbival),'dISS':diff(issval)})

granger_ts=granger_ts.dropna(how='any')

granger_ts[['dDJIA','dISS']]
sm.tsa.stattools.coint(granger_ts['dSP500'],granger_ts['dDJIA'], trend='c', method='aeg', autolag='aic')[1]
result=[]

nlags=5

granger_ts=granger_ts.reset_index()

df_granger=pd.DataFrame()

df_granger['Lags']=range(1,nlags+1)

df_granger=df_granger.set_index('Lags')

for ix in ['dDJIA', 'dSP500']:

    for metric in ['dBBI','dISS']:

        pairs=[ix+"-"+metric+" "+'F']

        result.append(pairs)

        for start, end in zip(range(len(granger_ts)-199),range(199,len(granger_ts))):

            granger=sm.tsa.stattools.grangercausalitytests(granger_ts[[ix,metric]].loc[start:end],maxlag=nlags,verbose=False)

            vallist=[]

            for lag in df_granger.index:

                vallist.append(granger[lag][0]['ssr_ftest'][1])

            pairs.append(vallist)

result
result[0][0]
fig, axs = plt.subplots(4,figsize=(10,30))



#for axis, in [axs[0,0],axs[0,1],axs[1,0],axs[1,1]]

#    axis.plot(X,lr.predict(X),color='red')



axs[0].set_title('p-értékek (dDJIA - dBBI)')

axs[1].set_title('p-értékek (dDJIA - dISS)')

axs[2].set_title('p-értékek (dSP500 - dBBI)')

axs[3].set_title('p-értékek (dSP500 - dISS)')

for pair in range(4):

    for lag,color in zip(range(len(vallist)),['blue','red','orange','purple','green']):

        lagvals=[]

        for j in result[pair][1:]:

            lagvals.append(j[lag])

        axs[pair].plot(lagvals,color=color,label='Lag '+str(lag+1))

    axs[pair].legend()

plt.savefig("Window_Granger.svg", format="svg")
price_data=prices[prices.index.floor('T').isin(iss.index)]

input_data=pd.DataFrame()

input_data['DJIA']=price_data['DJIA CLOSE']

nlags=10

for i in range(1,nlags+1):

    input_data['DJIA-'+str(i)]=price_data['DJIA CLOSE'].shift(i)

    input_data['ISS-'+str(i)]=iss['Sentiment Score'].shift(i)

input_data=input_data.dropna()

#input_data=input_data.reset_index()

#input_data=input_data.drop(columns='dates')

input_data.tail()
#Defining metrics

def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def mean_absolute_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred)))



def mean_square_error(y_true, y_pred): 

    return np.mean((y_true - y_pred)**2)



def root_mean_square_error(y_true, y_pred): 

    return np.sqrt(np.mean((y_true - y_pred)**2))



def directional_accuracy(y_true, y_pred):

    success=0

    for i,j,k,l in zip(list(y_true)[:-1],list(y_true)[1:],list(np.ndarray.tolist(y_pred)[0])[:-1],list(np.ndarray.tolist(y_pred)[0])[1:]):

        if j-i > 0:

            dir_test=1

        elif j-i<0:

            dir_test=-1

        else:

            dir_test=0

        if l-k > 0:

            dir_pred=1

        elif l-k<0:

            dir_pred=-1

        else:

            dir_pred=0

        if dir_pred==dir_test:

            success=success+1

    return success/len(list(y_true)[1:])



def trading_performance(y_curr, y_true, y_pred): 

    port_start=1000000

    port=port_start

    cost=0.0002

    position='neutral'

    for ycurr,ytrue,ypred in zip(y_curr,y_true,y_pred):

        expr=(ypred-ycurr)/ycurr

        actr=(ytrue-ycurr)/ycurr

        # Trading

        if expr > cost:

            if position=='long': # Hold condition

                port=(1+actr)*port

                position='long'

            elif position=='neutral': # Long condition

                port=(1+actr-cost)*port

                position='long'

            elif position=='short': # Cover condition

                port=(1-cost)*port

                position='neutral'

        elif expr < -cost:

            if position=='long': # Sell condition

                port=(1-cost)*port

                position='neutral'

            elif position=='neutral': # Short condition

                port=(1-actr-cost)*port

                position='short'

            elif position=='short': # Hold condition

                port=(1-actr)*port

                position='short'

        else: # Hold condition

            if position=='long':

                port=(1+actr)*port

            elif position=='short':

                port=(1-actr)*port

        print(port)

    return port/port_start-1
scaler2 = MinMaxScaler()

scaler2.fit_transform(y_act.reshape(-1, 1))

y_pred=scaler2.inverse_transform(pred[0].reshape(-1, 1))

trading_performance(data_test[:,1].tolist(),y_test,invscaled_y_pred.tolist())
mse_test=[]

rmse_test=[]

mae_test=[]

mape_test=[]

da_test=[]

tradingperf_test=[]

#Dimensions

n = data.shape[0]

p = data.shape[1]

data = input_data.values    



#Splitting to training and test data

train_start = 0

train_end = int(np.floor(0.8*n))

test_start = train_end

test_end = n

data_train = data[np.arange(train_start, train_end), :]

data_test = data[np.arange(test_start, test_end), :]

y_curr=data_test[:,1]

y_act=data_test[:,0]



#Scaling data

scaler = MinMaxScaler()

data_train = scaler.fit_transform(data_train)

data_test = scaler.transform(data_test)



# Build X and y

X_train = data_train[:, 1:]

y_train = data_train[:, 0]

X_test = data_test[:, 1:]

y_test = data_test[:, 0]



# Number of stocks in training data

n_inputs = X_train.shape[1]



# Neurons

n_neurons_1 = 64

n_neurons_2 = 32

n_neurons_3 = 16

n_neurons_4 = 8

n_target = 1



# Session

net = tf.InteractiveSession()



# Placeholder

X = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs])

Y = tf.placeholder(dtype=tf.float32, shape=[None])



# Initializers

sigma = 1

weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)

bias_initializer = tf.zeros_initializer()



# Hidden weights

W_hidden_1 = tf.Variable(weight_initializer([n_inputs, n_neurons_1]))

bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))

bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))

bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))

W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))

bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))



# Output weights

W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))

bias_out = tf.Variable(bias_initializer([1]))



# Hidden layer

hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))

hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))

hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))

hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))



# Output layer (transpose!)

out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))



# Cost function

mse = tf.reduce_mean(tf.squared_difference(out, Y))



# Optimizer

opt = tf.train.AdamOptimizer().minimize(mse)



# Init

net.run(tf.global_variables_initializer())



# Fit neural net

batch_size = 5

mse_train_l = []

mse_test_l = []



# Run

epochs = 25

for e in range(epochs):



    # Shuffle training data

    shuffle_indices = np.random.permutation(np.arange(len(y_train)))

    X_train = X_train[shuffle_indices]

    y_train = y_train[shuffle_indices]



    # Minibatch training

    for i in range(0, len(y_train) // batch_size):

        start = i * batch_size

        batch_x = X_train[start:start + batch_size]

        batch_y = y_train[start:start + batch_size]

        # Run optimizer with batch

        net.run(opt, feed_dict={X: batch_x, Y: batch_y})



        # Show progress

        if np.mod(i, 5) == 0:

            # MSE train and test

            mse_train_l.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))

            mse_test_l.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))

            #print('MSE Train: ', mse_train_l[-1])

            #print('MSE Test: ', mse_test_l[-1])

            # Prediction

            pred = net.run(out, feed_dict={X: X_test})

            plt.pause(0.001)



# Metrics

print("Calculating Metrics")

print(p)

#print(data_test[:,1])

mse_test.append(mean_square_error(y_test,pred))

rmse_test.append(root_mean_square_error(y_test,pred))

mae_test.append(mean_absolute_error(y_test,pred))

mape_test.append(mean_absolute_percentage_error(y_test,pred))

da_test.append(directional_accuracy(y_test,pred))

scaler2 = MinMaxScaler()

scaler2.fit_transform(y_act.reshape(1, -1))

y_pred=scaler2.inverse_transform(pred[0].reshape(1, -1))

print(y_pred[0])

print(tradingperf_test)

trading_performance(y_curr.tolist(),y_act,y_pred[0])
trading_performance(y_curr.tolist(),y_act,y_curr.tolist())
def run_nn(input_data, modeltype):

    

    global pred, y_test, X_test, y_act, y_curr, mse_train_l, mse_test_l

    

    mse_test=[]

    rmse_test=[]

    mae_test=[]

    mape_test=[]

    da_test=[]

    tradingperf_test=[]

    if modeltype=='u': # Unrestricted model

        step=2

    else: # Restricted model

        step=1

    for i in range(0,len(input_data.columns)-1,step):

        if i==0:

            data=input_data.iloc[:,:]

        else:

            data=input_data.iloc[:,:-i]

        

        #Dimensions

        n = data.shape[0]

        p = data.shape[1]

        data = data.values    



        #Splitting to training and test data

        train_start = 0

        train_end = int(np.floor(0.8*n))

        test_start = train_end

        test_end = n

        data_train = data[np.arange(train_start, train_end), :]

        data_test = data[np.arange(test_start, test_end), :]

        y_curr=data_test[:,1]

        y_act=data_test[:,0]



        #Scaling data

        scaler = MinMaxScaler()

        data_train = scaler.fit_transform(data_train)

        data_test = scaler.transform(data_test)



        # Build X and y

        X_train = data_train[:, 1:]

        y_train = data_train[:, 0]

        X_test = data_test[:, 1:]

        y_test = data_test[:, 0]



        # Number of stocks in training data

        n_inputs = X_train.shape[1]



        # Neurons

        n_neurons_1 = 64

        n_neurons_2 = 32

        n_neurons_3 = 16

        n_neurons_4 = 8

        n_target = 1



        # Session

        net = tf.InteractiveSession()



        # Placeholder

        X = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs])

        Y = tf.placeholder(dtype=tf.float32, shape=[None])



        # Initializers

        sigma = 1

        weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)

        bias_initializer = tf.zeros_initializer()



        # Hidden weights

        W_hidden_1 = tf.Variable(weight_initializer([n_inputs, n_neurons_1]))

        bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

        W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))

        bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

        W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))

        bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))

        W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))

        bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))



        # Output weights

        W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))

        bias_out = tf.Variable(bias_initializer([1]))



        # Hidden layer

        hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))

        hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))

        hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))

        hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))



        # Output layer (transpose!)

        out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))



        # Cost function

        mse = tf.reduce_mean(tf.squared_difference(out, Y))



        # Optimizer

        opt = tf.train.AdamOptimizer().minimize(mse)



        # Init

        net.run(tf.global_variables_initializer())



        # Fit neural net

        batch_size = 5

        mse_train_l = []

        mse_test_l = []



        # Run

        epochs = 25

        for e in range(epochs):



            # Shuffle training data

            shuffle_indices = np.random.permutation(np.arange(len(y_train)))

            X_train = X_train[shuffle_indices]

            y_train = y_train[shuffle_indices]



            # Minibatch training

            for i in range(0, len(y_train) // batch_size):

                start = i * batch_size

                batch_x = X_train[start:start + batch_size]

                batch_y = y_train[start:start + batch_size]

                # Run optimizer with batch

                net.run(opt, feed_dict={X: batch_x, Y: batch_y})



                # Show progress

                if np.mod(i, 5) == 0:

                    # MSE train and test

                    mse_train_l.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))

                    mse_test_l.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))

                    #print('MSE Train: ', mse_train_l[-1])

                    #print('MSE Test: ', mse_test_l[-1])

                    # Prediction

                    pred = net.run(out, feed_dict={X: X_test})

                    plt.pause(0.001)



        # Metrics

        print("Calculating Metrics")

        print(p)

        mse_test.append(mean_square_error(y_test,pred))

        rmse_test.append(root_mean_square_error(y_test,pred))

        mae_test.append(mean_absolute_error(y_test,pred))

        mape_test.append(mean_absolute_percentage_error(y_test,pred))

        da_test.append(directional_accuracy(y_test,pred))

        scaler2 = MinMaxScaler()

        scaler2.fit_transform(y_act.reshape(-1, 1))

        y_pred=scaler2.inverse_transform(pred[0].reshape(1, -1))

        tradingperf_test.append(trading_performance(y_curr.tolist(),y_act,y_pred[0]))

        print(pred[0])

        print(y_pred[0])

        print(tradingperf_test)

        

    return list(zip(mae_test,mape_test,mse_test,rmse_test,da_test,tradingperf_test))
nn_performance=pd.DataFrame(run_nn(input_data,'u'),columns =['MAE','MAPE','MSE','RMSE','Dir. Acc.','Tr. Perf.'])

cols = [c for c in input_data.columns if c[:3] != 'ISS']

data=input_data[cols]

nn_performance_naive=pd.DataFrame(run_nn(data,'r'),columns =['MAE','MAPE','MSE','RMSE','Dir. Acc.','Tr. Perf.'])

nn_performance=nn_performance.append(nn_performance_naive)

index_range=['U'+str(i) for i in range(10,0,-1)]+['R'+str(i) for i in range(10,0,-1)]

nn_performance.set_index(pd.Index(index_range))
plt.plot(range(1,len(y_test)+1), y_test,label='Test')

plt.plot(range(1,len(y_test)+1),pred[0],label='Prediction')

plt.legend()

plt.savefig("NN_test_pred.svg", format="svg")
plt.plot(range(1,len(mse_train_l[30:])+1), mse_train_l[30:],label='MSE - tanító készlet')

plt.plot(range(1,len(mse_test_l[30:])+1),mse_test_l[30:],label='MSE - tesztelő készlet')

plt.legend()

plt.title('R1-es modell MSE')

plt.savefig("NN_test_pred.svg", format="svg")
price_data=prices[prices.index.floor('T').isin(iss.index)]

input_data=pd.DataFrame()

input_data['SP500']=price_data['SP500 CLOSE']

nlags=10

for i in range(1,nlags+1):

    input_data['SP500-'+str(i)]=price_data['SP500 CLOSE'].shift(i)

    input_data['ISS-'+str(i)]=iss['Sentiment Score'].shift(i)

input_data=input_data.dropna()

input_data
nn_performance=pd.DataFrame(run_nn(input_data,'u'),columns =['MAE','MAPE','MSE','RMSE','Dir. Acc.','Tr. Perf.'])

cols = [c for c in input_data.columns if c[:3] != 'ISS']

data=input_data[cols]

nn_performance_naive=pd.DataFrame(run_nn(data,'r'),columns =['MAE','MAPE','MSE','RMSE','Dir. Acc.','Tr. Perf.'])

nn_performance=nn_performance.append(nn_performance_naive)

index_range=['U'+str(i) for i in range(10,0,-1)]+['R'+str(i) for i in range(10,0,-1)]

nn_performance.set_index(pd.Index(index_range))
trading_performance(y_curr.tolist(),y_act,y_act)