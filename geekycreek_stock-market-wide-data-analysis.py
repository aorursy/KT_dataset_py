!pip install arch
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime,timedelta



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.svm import SVC, LinearSVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import OneHotEncoder 

from sklearn.feature_extraction.text import TfidfVectorizer



from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.stattools import adfuller

from arch import arch_model



from tensorflow.keras import Sequential

from tensorflow.keras.layers import LSTM, Dense, Bidirectional,TimeDistributed, RepeatVector, Input

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor



import warnings

warnings.filterwarnings('ignore')

import time

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Q_data = pd.read_csv('/kaggle/input/stock-market-wide-datasets/000000000000Q',nrows=50000)

AM_data = pd.read_csv('/kaggle/input/stock-market-wide-datasets/AM',nrows=50000)

news = pd.read_csv('/kaggle/input/stock-market-wide-datasets/news')

event = pd.read_csv('/kaggle/input/stock-market-wide-datasets/event')
Q_data.head()
Q_data.describe()
AM_data.head()
AM_data.describe()
news.head()
event.head()
combined_data = AM_data.merge(Q_data,how='inner',left_on='symbol',right_on='ticker')

combined_data.drop(['time_y','symbol'],axis=1,inplace=True)
combined_data.head()
combined_data.columns
print("Unique tickers available in the dataset",combined_data['ticker'].unique())
AAL_data = combined_data[combined_data['ticker']=='AAL']
AAL_data['hour'] = pd.to_datetime(AAL_data['time_x']).apply(lambda x:x.strftime('%H'))

AAL_data['date'] = pd.to_datetime(AAL_data['time_x']).apply(lambda x:x.strftime('%Y-%m-%d'))
dependent_variables = ['volume','accumulated_volume','VWAP','open_price','high_price','low_price',

                      'close_price','average_price','ask_price','ask_size','bid_price','bid_size']

one_hour_data = AAL_data[AAL_data['hour']=='15']

mean_one_hour_data = one_hour_data.groupby('time_x').mean()[dependent_variables]

mean_one_hour_data['time_x'] = list(mean_one_hour_data.index)

mean_one_hour_data['time_x'] = pd.to_datetime(mean_one_hour_data['time_x']).apply(lambda x:x.strftime('%H:%M:%S'))
plt.rcParams['figure.figsize'] = (20,30)

count = 1

for val in dependent_variables:

    plt.subplot(4,3,count)

    sns.lineplot(x='time_x',y=val,data=mean_one_hour_data)

    plt.ylabel('Mean '+val)    

    plt.xlabel('time')

    plt.xticks(rotation=90)

    count+=1

print()
df = one_hour_data[dependent_variables]

sns.pairplot(data = df)
features = ['time_x','volume','accumulated_volume','VWAP','open_price','high_price','low_price','close_price',

            'average_price','bid_price','bid_size','ask_price','ask_size','ticker']

corr_df = combined_data[:100000][features]

corr_df['hour'] = pd.to_datetime(corr_df['time_x']).apply(lambda x:int(x.strftime('%H')))

corr_df['weekday'] = pd.to_datetime(corr_df['time_x']).apply(lambda x:int(x.strftime('%w')))

corr_df['year'] = pd.to_datetime(corr_df['time_x']).apply(lambda x:int(x.strftime('%Y')))

corr_df['time'] = pd.to_datetime(corr_df['time_x']).apply(lambda x:x.strftime('%H:%M:%S'))
cat_features = corr_df['ticker'].unique()

encoder = OneHotEncoder(categories=[cat_features])

corr_df[cat_features] = encoder.fit_transform(corr_df['ticker'].values.reshape(-1,1)).toarray()
plt.rcParams['figure.figsize'] = (20,10)

sns.heatmap(corr_df.corr(),cmap='Blues',annot=True)
corr_df = corr_df.sort_values(by='time_x')
selected_features = ['volume','accumulated_volume','VWAP','open_price','low_price','high_price','close_price',

                     'average_price','bid_price','bid_size','ask_price','hour','weekday']

X = corr_df[selected_features]

Y = corr_df[['AMD','AES','MO']]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state=0)
print("X_train shape: {}, Y_train shape: {}".format(X_train.shape,Y_train.shape))

print("X_test shape: {}, Y_test shape: {}".format(X_test.shape, Y_test.shape))
#Function to convert one-hot encoded output to label encoded output

def convert_label(val):

    label_dict = {0:[1,0,0], 1:[0,1,0], 2:[0,0,1]}

    result = 0

    for i in label_dict:

        if (label_dict[i]==val).all():

            result = i

    return result
log_reg_clf = MultiOutputClassifier(LogisticRegression())

log_reg_clf.fit(X_train,Y_train)

Y_train_pred = log_reg_clf.predict(X_train) 

Y_test_pred = log_reg_clf.predict(X_test)

print("Train accuracy",accuracy_score(Y_train,Y_train_pred))

print("Test accuracy",accuracy_score(Y_test,Y_test_pred))
Y_test_label = [convert_label(val) for val in Y_test.values]

Y_test_pred_label = [convert_label(val) for val in Y_test_pred]

label_list = ['AMD','AES','MO']

confusion_matrix_df = pd.DataFrame(data = confusion_matrix(Y_test_label,Y_test_pred_label), 

                                   columns=label_list, index=label_list)

sns.heatmap(confusion_matrix_df,cmap='Blues_r',annot=True)
KNN_clf = MultiOutputClassifier(KNeighborsClassifier(7))

KNN_clf.fit(X_train,Y_train)

Y_train_pred = KNN_clf.predict(X_train) 

Y_test_pred = KNN_clf.predict(X_test)

print("Train accuracy",accuracy_score(Y_train,Y_train_pred))

print("Test accuracy",accuracy_score(Y_test,Y_test_pred))
Y_test_label = [convert_label(val) for val in Y_test.values]

Y_test_pred_label = [convert_label(val) for val in Y_test_pred]

label_list = ['AMD','AES','MO']

confusion_matrix_df = pd.DataFrame(data = confusion_matrix(Y_test_label,Y_test_pred_label), 

                                   columns=label_list, index=label_list)

sns.heatmap(confusion_matrix_df,cmap='Blues_r',annot=True)
svc_clf = MultiOutputClassifier(SVC())

svc_clf.fit(X_train,Y_train)

Y_train_pred = svc_clf.predict(X_train) 

Y_test_pred = svc_clf.predict(X_test)

print("Train accuracy",accuracy_score(Y_train,Y_train_pred))

print("Test accuracy",accuracy_score(Y_test,Y_test_pred))
Y_test_label = [convert_label(val) for val in Y_test.values]

Y_test_pred_label = [convert_label(val) for val in Y_test_pred]

label_list = ['AMD','AES','MO']

confusion_matrix_df = pd.DataFrame(data = confusion_matrix(Y_test_label,Y_test_pred_label), 

                                   columns=label_list, index=label_list)

sns.heatmap(confusion_matrix_df,cmap='Blues_r',annot=True)
directory = '/kaggle/input/stock-market-wide-datasets/AM'

AM_data = pd.read_csv(directory)
AAL_data = AM_data[AM_data['symbol']=='AAL']
AAL_data_ts = AAL_data[['time','open_price']]

AAL_data_ts['time'] = pd.to_datetime(AAL_data_ts['time'])

AAL_data_ts = AAL_data_ts.groupby('time').mean()
plt.title('Time Series plot for AAL ticker')

plt.plot(AAL_data_ts)

plt.ylabel('mean open price')

plt.xlabel('time')
def stationary_test(timeseries):

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)

stationary_test(AAL_data_ts)
train_ts = AAL_data_ts[:3501]

test_ts = AAL_data_ts[3501:]

model = ARIMA(train_ts, order=(10,0,2))  

results_ARIMA = model.fit(disp=-1)  



#Future Forecasting

history = list(train_ts['open_price'])

predictions = []

test_data = list(test_ts['open_price'])

for i in range(len(test_data)):

    model = ARIMA(history, order=(10,0,2))

    model_fit = model.fit(disp=-1)

    output = model_fit.forecast()

    yhat = output[0]

    predictions.append(float(yhat))

    obs = test_data[i]

    history.append(obs)

plt.rcParams['figure.figsize'] = (20,10)

plt.plot(AAL_data_ts)

plt.plot(results_ARIMA.fittedvalues, color='red')

plt.plot(test_ts.index,predictions,color='green')

plt.axvline(AAL_data_ts.index[3501],color='orange',dashes=(5,2,1,2))

plt.legend(['actual values','predicted values for train data','predicted values for test data'])
print(results_ARIMA.summary())

# plot residual errors

residuals = pd.DataFrame(results_ARIMA.resid)

residuals.plot(kind='kde')

print(residuals.describe())
plt.rcParams['figure.figsize'] = (10,5)

#10 days rolling standard deviation

AAL_data_ts.rolling(10).std().plot(style='b')

plt.title('Moving Standard Deviation')

plt.ylabel('Standard Deviation')

#10 days rolling mean

AAL_data_ts.rolling(10).mean().plot(style='r')

plt.title('Moving Mean')

plt.ylabel('Mean')
model = arch_model(AAL_data_ts, mean='ARX', vol='GARCH', p=10)
model_fit = model.fit()
y = model_fit.forecast(horizon=30)

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
train_ts = AAL_data_ts['open_price'][:2001]

test_ts = AAL_data_ts['open_price'][2001:]

X_train, Y_train = split_sequence(train_ts,3,1)

Y_train = np.squeeze(Y_train)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
def r2_score(y_true, y_pred):

    from tensorflow.keras import backend as K

    SS_res = K.sum(K.square(y_true-y_pred))

    SS_tot = K.sum(K.square(y_true-K.mean(y_true)))

    return (1-SS_res/(SS_tot+K.epsilon()))
# define model

model = Sequential()

model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(3, 1)))

model.add(LSTM(100, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=[r2_score])

model.summary()
training = model.fit(X_train,Y_train,epochs=10)
history = list(train_ts)

predictions = []

for i in range(len(test_ts)):

    pred = np.squeeze(model.predict(np.array(history[-3:]).reshape(1,3,1)))

    history.append(float(pred))

    predictions.append(float(pred))
plt.plot(AAL_data_ts['open_price'])

plt.plot(train_ts,color='red')

plt.plot(test_ts.index,predictions,color='red')

plt.axvline(AAL_data_ts.index[2001],color='orange',dashes=(5,2,1,2))

plt.rcParams['figure.figsize'] = (25,10)
model = Sequential()

model.add(Input((3, 1)))

model.add(Bidirectional(LSTM(100, activation='relu', return_sequences=True)))

model.add(Bidirectional(LSTM(100, activation='relu')))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=[r2_score])

model.summary()
training = model.fit(X_train,Y_train,epochs=10)
history = list(train_ts)

predictions = []

for i in range(len(test_ts)):

    pred = np.squeeze(model.predict(np.array(history[-3:]).reshape(1,3,1)))

    history.append(float(pred))

    predictions.append(float(pred))

plt.plot(AAL_data_ts['open_price'])

plt.plot(train_ts,color='red')

plt.plot(test_ts.index,predictions,color='red')

plt.axvline(AAL_data_ts.index[2001],color='orange',dashes=(5,2,1,2))

plt.rcParams['figure.figsize'] = (25,10)
AAL_AM_data = AM_data[AM_data['symbol']=='AAL']

AAL_AM_data = AAL_AM_data.groupby('time').mean()

AAL_AM_data = AAL_AM_data.sort_values(by='time')

AAL_AM_data_ts = AAL_AM_data['open_price']
model = ARIMA(AAL_AM_data_ts, order=(10,0,2))

time_diff = (datetime(2020,10,22)-datetime(2020,8,21)).days

results_ARIMA = model.fit(disp=-1)  

datetime_list = []

#Future Forecasting

history = list(AAL_AM_data_ts)

predictions = []

for i in range(time_diff):

    model = ARIMA(history, order=(10,0,2))

    model_fit = model.fit(disp=-1)

    datetime_list.append(str(pd.to_datetime(list(AAL_AM_data_ts.index)[-1])+timedelta(days=i+1)))

    output = model_fit.forecast()

    yhat = output[0]

    predictions.append(float(yhat))

    history.append(yhat)

plt.rcParams['figure.figsize'] = (20,10)

plt.plot(results_ARIMA.fittedvalues, color='red')

plt.plot(datetime_list,predictions, color='green')

plt.axvline(datetime_list[0],color='orange',dashes=(5,2,1,2))

plt.legend(['actual values','forecasted values'])
print("Predicted value of AAL ticker 5 days prior to report date:",predictions[-5])
from sklearn.metrics import r2_score
directory = '/kaggle/input/stock-market-wide-datasets/AM'

AM_data = pd.read_csv(directory)
stock_list = list(set(AM_data.symbol))[:10]

df_list = []

for stock in stock_list:

    stock_AM_data = AM_data[AM_data['symbol']==stock]

    stock_AM_data = stock_AM_data.groupby('time').mean()

    stock_AM_data = stock_AM_data.sort_values(by='time')

    stock_AM_data['time'] = list(stock_AM_data.index)

    stock_AM_data_ts = stock_AM_data['open_price']

    stock_AM_data.index = range(len(stock_AM_data))

    stock_news_data = news[news['stock']==stock]

    stock_AM_data['time'] = pd.to_datetime(stock_AM_data['time']).apply(lambda x:x.strftime('%Y-%m-%d'))

    stock_news_data['time'] = pd.to_datetime(stock_news_data['datetime']).apply(lambda x:x.strftime('%Y-%m-%d'))

    stock_news_AM = stock_AM_data.merge(stock_news_data,how='inner',right_on='time',left_on='time')

    stock_news_AM = stock_news_AM[['stock','average_price','summary']]

    df_list.append(stock_news_AM)

stock_news_AM = pd.concat(df_list)
corpus = list(stock_news_AM['summary'].apply(lambda x: x.lower()))
tfidf = TfidfVectorizer(stop_words={'english'},max_df=0.3,ngram_range=(5,5),min_df=7)

X = tfidf.fit_transform(corpus).toarray()

Y = stock_news_AM['average_price']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.5,random_state=0)
lin_reg = LinearRegression()

lin_reg.fit(X_train,Y_train)
print("Train R2-score: ",r2_score(Y_train,lin_reg.predict(X_train)))

print("Test R2-score:",r2_score(Y_test,lin_reg.predict(X_test)))
def create_keyword_weight_df(model):

    keywords = np.array(sorted(tfidf.vocabulary_.items(), key=lambda x: x[1]))[:,0]

    weights = model.coef_

    keywords_weights = []

    for k,w in zip(keywords,weights):

        keywords_weights.append([k,w])

    keywords_weights = pd.DataFrame(data = keywords_weights,columns=['keywords','weights'])

    keywords_weights = keywords_weights.sort_values(by='weights',ascending=False)

    return keywords_weights
keywords_weights = create_keyword_weight_df(lin_reg)

keywords_weights[:50]
keywords_weights[-50:]
lin_svr = LinearSVR()

lin_svr.fit(X_train,Y_train)
print("Train R2-score: ",r2_score(Y_train,lin_svr.predict(X_train)))

print("Test R2-score:",r2_score(Y_test,lin_svr.predict(X_test)))
print("Top 50 documents with highest positive weights:")

keywords_weights = create_keyword_weight_df(lin_svr)

keywords_weights[:50]
print("Top 50 documents with lowest negative weights:")

keywords_weights[-50:]
from textblob import TextBlob

stock_news_AM[['polarity', 'subjectivity']] = stock_news_AM['summary'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
stock_news_AM