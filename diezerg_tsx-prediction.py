import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from xgboost.sklearn import XGBClassifier
import warnings

from keras.layers import Dense, Dropout, LSTM, Bidirectional,GRU
from keras import Sequential, regularizers
from keras.optimizers import Adam
warnings.filterwarnings("ignore")
cbc = pd.read_csv('../input/cbcnews/Cnews_clean.csv')
cbc = cbc.iloc[:,1:-1]
cbc.head()
gb = pd.read_csv('../input/cbcnews/Gnews_clean.csv')
gb = gb.iloc[:,1:-1]
gb.head()
news = pd.concat([cbc,gb],axis=0, sort=False)
news.drop_duplicates(['title'],keep='first',inplace=True)
news.shape
# text cluster
def clean_txt(txt):
    tokens = []
    self_stopwords = ['cbc', 'year', 'say', 'day', 'canadian','canada','new','newscbc','news']
    for text in txt:
        token = word_tokenize(text)
        token = [w.lower() for w in token]
        token = [w for w in token if w.isalpha()]
        token = [WordNetLemmatizer().lemmatize(w) for w in token]
        token = [w for w in token if w not in stopwords.words('english')]
        token = [w for w in token if w not in self_stopwords]
        token = ' '.join(token)
        tokens.append(token)
    return tokens

def news_clusters(df, tsx, c=4, clean=False, svd=50, keeptotal=False):
    df.dropna(inplace=True)
    if clean:
        tt = clean_txt(df['title'])
        df['title'] = tt
    else:
        tt = df['title']

    token = CountVectorizer().fit_transform(tt)
    token = TfidfTransformer().fit_transform(token)
    svd = TruncatedSVD(n_components=100).fit_transform(token)
    print('clustering----------')
    clt = KMeans(c,init='k-means++', max_iter=200).fit(svd)
    df['cluster'] = clt.labels_
    if keeptotal == True:
        table0 = pd.pivot_table(df, index=['date'], values=['sentiment'], aggfunc=[np.mean]).dropna(how='all')
        table = pd.pivot_table(df, index=['date'], columns=['cluster'], values=['sentiment'], aggfunc=[np.mean]).dropna(how='all')
        table0 = table0.fillna(0)
        table = table.fillna(0)
        df = tsx.merge(table0, left_on='Date',right_on='date')
        df = df.merge(table, left_on='Date',right_on='date')
        df.columns = [str(w).replace("'","").replace("(mean,","").replace(')','').replace(' ','') for w in df.columns]
    else:     
        #table0 = pd.pivot_table(df, index=['date'], values=['sentiment'], aggfunc=[np.mean]).dropna(how='all')
        table = pd.pivot_table(df, index=['date'], columns=['cluster'], values=['sentiment'], aggfunc=[np.mean]).dropna(how='all')
        #table0 = table0.fillna(0)
        table = table.fillna(0)
        #df = tsx.merge(table0, left_on='Date',right_on='date')
        df = tsx.merge(table, left_on='Date',right_on='date')
        df.columns = [str(w).replace("'","").replace("(mean,","").replace(')','').replace(' ','') for w in df.columns]
    
    return df


def news_topics(df, tsx):
    #table0 = pd.pivot_table(df, index=['date'], values=['sentiment'], aggfunc=[np.mean]).dropna(how='all')
    table = pd.pivot_table(df, index=['date'], columns=['type'], values=['sentiment'], aggfunc=[np.mean]).dropna(how='all')
    #table0 = table0.fillna(0)
    table = table.fillna(0)
    #df = tsx.merge(table0, left_on='Date',right_on='date')
    df = tsx.merge(table, left_on='Date',right_on='date')
    df.columns = [str(w).replace("'","").replace("(mean,","").replace(')','').replace(' ','') for w in df.columns]
    return df


tsx = pd.read_csv('../input/tsx-index/tsxPrices.csv')
data = news_clusters(news, tsx,7, clean=False)
#data.columns = [str(w).replace("'","").replace("(mean,","").replace(')','').replace(' ','') for w in data.columns]
data.head()
# pure sentiment predictor


tsx = pd.read_csv('../input/tsx-index/tsxPrices.csv')
data = news_clusters(news, tsx,7, clean=False, keeptotal=True)

change = [data['AdjClose'][n]/data['AdjClose'][n+1] - 1 for n in range(data.shape[0]-1)]
change.append(change[-1])
data['change'] = change

def signal(tsx, days=15, cost_rate=0, required_rate=0.01):
    high_price = [max(tsx['High'][i + 1:i + days + 1]) * (1 - cost_rate) for i in range(tsx.shape[0] - days)]
    high_price = high_price + list(tsx['High'][-days:])
    # tsx['possible_price'] = high_price
    # tsx['gain'] = high_price - tsx['Adj Close']
    # tsx['gain_rate'] = high_price/tsx['Adj Close'] -1
    tsx['signal'] = (high_price / tsx['AdjClose'] > 1 + required_rate) * 1
    return tsx

data = signal(data)

tsx = data.loc[:,['sentiment','change']]
st = MinMaxScaler()
tsx['change'] = st.fit_transform(tsx['change'].values.reshape(-1,1))
tsx['change'] = (tsx['change']-0.4)*2

plt.figure(figsize=(15,10))
plt.plot(data['Date'],tsx['sentiment'], label='sentiments')
plt.plot(data['Date'],tsx['change'], label='Price change')
plt.xticks([data['Date'][t] for t in range(1,data.shape[0],data.shape[0]//10)])
plt.legend()
plt.title('Relationship between sentiments and Price changes')
# XGBoost modeling
#y = data['change']>0
#x = tsx['sentiment'].values.reshape(-1,1)
tsx = pd.read_csv('../input/tsx-index/tsxPrices.csv')
data = news_topics(news, tsx)
data = signal(data, days=15, cost_rate=0, required_rate=0.01)

y = data['signal']
senti = list(data.columns[data.columns.str.contains('sentiment')])
x = data[senti]

cut = int(len(x)*0.7)
x_train, x_test = x[:cut], x[cut:]
y_train, y_test = y[:cut], y[cut:]
x_train.shape
y_train.shape

sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


xgb = XGBClassifier()
xgb.fit(x_train, y_train)
print('simple model accuracy:',xgb.score(x_test, y_test))

param = {
 'max_depth':[2,3,4,5],
 'min_child_weight':[6,7,8],
#  'gamma':[i/10.0 for i in range(0,5)],
#  'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gch = GridSearchCV(xgb,param,scoring='accuracy',cv=None)
gch.fit(x_train, y_train)
print('adjusted hyperparameter accuracy:',gch.score(x_test,y_test))
print(gch.best_params_)
y_pred=gch.predict(x_test)
print(metrics.confusion_matrix(y_pred, y_test))



# moving average
data_ma = data.__deepcopy__()
price = data['AdjClose']
for day in [5,10,20,50]:
    data_ma['MA%i'% day] = [np.mean(price[i:i+day]) for i in range(data_ma.shape[0]-day)] + [0]*day


data['Date']

plt.figure(figsize=(15,10))
plt.plot(data['Date'],data_ma['AdjClose'], 'b', label='Price')
plt.plot(data['Date'],data_ma['MA5'], 'y' ,label='MA5')
plt.plot(data['Date'],data_ma['MA10'],'purple' ,label='MA10')
plt.plot(data['Date'],data_ma['MA20'], 'c' ,label='MA20')
plt.plot(data['Date'],data_ma['MA50'],'r', label='MA50')
plt.xticks([data['Date'][t] for t in range(1,data.shape[0],data.shape[0]//10)])
plt.axis([0,1400,10000,18000])
plt.legend()
plt.title('Moving Average Forecasting')
plt.show()
# LSMT
data_lsmt = data.__deepcopy__()
data.columns
data_lsmt = data_lsmt[['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']]
cut = int(data_lsmt.shape[0] * 0.7)
train, test = data_lsmt[:cut], data_lsmt[cut:]



sc = MinMaxScaler()
train = sc.fit_transform(train)
test = sc.transform(test)

def to_hist(inputs, hist=20):
    r = inputs.shape[0]
    x = np.array([inputs[i:i + hist] for i in range(r - hist - 1)])
    y = np.array([list(inputs)[i + hist] for i in range(r - hist - 1)])
    x = x.reshape(x.shape[0], 1, x.shape[1])
    return x, y


x_train, y_train = to_hist(train[:,-2],  20)
x_test, y_test = to_hist(test[:,-2], 20)

model = Sequential()
model.add(LSTM(40,input_shape=(x_train.shape[1],x_train.shape[2]), dropout=0.1,return_sequences=True))
model.add(LSTM(20,dropout=0.2))
model.add(Dense(10))
model.add(Dense(1,activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.summary()

hit = model.fit(x_train,y_train,batch_size=100, epochs=50, validation_data=([x_test,y_test]), verbose=0)



plt.figure(figsize=(15,10))
plt.subplot(221)
t_acc = hit.history['loss']
v_acc = hit.history['val_loss']
epochs = np.arange(len(t_acc))
plt.plot(epochs + 1, t_acc, 'b--', label='Train_loss')
plt.plot(epochs + 1, v_acc, 'r', label='Validate_loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.title('Loss plot')
plt.legend()

plt.subplot(222)
t_acc = hit.history['mean_squared_error']
v_acc = hit.history['val_mean_squared_error']
epochs = np.arange(len(t_acc))
plt.plot(epochs + 1, t_acc, 'b--', label='Train_mse')
plt.plot(epochs + 1, v_acc, 'r', label='Validate_mse')
plt.ylabel('MSE')
plt.xlabel('epochs')
plt.title('MSE plot')
plt.legend()


x_test.shape
y_test.shape

date_train, date_test = data['Date'][:len(y_train)],data['Date'][-len(y_test):]
date_test.shape == y_test.shape
date_train.shape == y_train.shape


plt.subplot(223)
y_get = model.predict(x_train)
plt.plot(date_train, y_train, 'b', label='ACTUAL')
plt.plot(date_train, y_get, 'r--', label='predict')
plt.xticks([date_train[i] for i in range(1,len(date_train), len(date_train)//6)], rotation=25)
plt.legend()
plt.title('Prediction on Train')

plt.subplot(224)
y_pred = model.predict(x_test)
plt.plot(date_test, y_test, 'b', label='ACTUAL')
plt.plot(date_test, y_pred, 'r--', label='predict')
plt.xticks([list(date_test)[j] for j in range(1,len(date_test), len(date_test)//6)], rotation=25)
plt.legend()
plt.title('Prediction on Validation')
# Hybird model(LSTM + text mining)
hold_days = 15
expect_return = 0.0195
cost_rate = 0.001
lookback = 20 #how many day for history looking
options = [] + ['High', 'AdjClose']
# ['Open', 'Low', 'Close', 'Volume']
use_sentiment = True
use_clusters = True
cluster_num = 5
cluster_cut = 3
validation_rate = 0.4
epochs = 150
learning_rate = 0.0005

tsx0 = pd.read_csv('../input/tsx-index/tsxPrices.csv')
if use_clusters:
    tsx = news_clusters(news, tsx0,cluster_num, clean=False)
else:
    tsx = news_topics(news, tsx0)

# organize data
tsx = tsx.loc[:, 'Date':]
tsx = tsx.sort_values(by='Date', ascending=True)
tsx.reset_index(drop=True, inplace=True)
tsx0 = tsx.__deepcopy__()

senti = list(tsx.columns[tsx.columns.str.contains('sentiment')][:cluster_cut+1])

if use_sentiment:
    tsx = tsx[options + senti]
else:
    tsx = tsx[options]

tsx = signal(tsx, hold_days, cost_rate, expect_return)


cut = int(tsx.shape[0] * validation_rate)
train, test = tsx[:-cut], tsx[-cut:]

sc = MinMaxScaler()
train = sc.fit_transform(train)
test = sc.transform(test)


def to_hist(inputs, outputs, hist=20):
    r = inputs.shape[0]
    x = np.array([inputs[i:i + hist] for i in range(r - hist - 1)])
    y = np.array([list(outputs)[i + hist] for i in range(r - hist - 1)])
    # x = x.reshape(x.shape[0], x, x.shape[1])
    return x, y


x_train, y_train = to_hist(train[:, :-1], train[:, -1], lookback)
x_test, y_test = to_hist(test[:, :-1], test[:, -1], lookback)

model = Sequential()
model.add(LSTM(32,input_shape=(x_train.shape[1], x_train.shape[2]),dropout=0.1,
              kernel_regularizer=regularizers.l2(0.002),return_sequences=True))
model.add(LSTM(64,dropout=0.1,kernel_regularizer=regularizers.l2(0.002),return_sequences=False))
#model.add(GRU(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['acc'])

model.summary()

records = model.fit(x_train, y_train, batch_size=512, epochs=epochs, validation_data=[x_test, y_test], verbose=0)

plt.figure(figsize=(15,10))
t_acc = records.history['acc']
v_acc = records.history['val_acc']
epochs = np.arange(len(t_acc))
plt.plot(epochs + 1, t_acc, 'b--', label='Train_acc')
plt.plot(epochs + 1, v_acc, 'r', label='Validate_acc')
plt.ylabel('Accuracy')
plt.xlabel('epochs')
title = 'Signal accuracy (' +'with sentiment'*use_sentiment+', with cluster%i-%i' % (cluster_num, cluster_cut) * use_clusters+')' \
           '\n' + 'with {}'.format(options) + '\n %.1f percent holding %i days' % (expect_return*100, hold_days)
plt.title(title)
plt.legend()
plt.show()

print('signal_num: ', sum(model.predict(x_test)>0.5))
print('non-signal_num: ', sum(model.predict(x_test)<=0.5))
