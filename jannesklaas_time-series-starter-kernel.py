import pandas as pd
import numpy as np
%matplotlib inline
train = pd.read_csv('../input/Train.csv').fillna(0)
train = train.drop(train.columns[0], axis=1)
train = train.loc[:10000,:] # Subsample training set
train.head()
def parse_page(page):
    x = page.split('_')
    return ' '.join(x[:-3]), x[-3], x[-2], x[-1]
l = list(train.Page.apply(parse_page))
df = pd.DataFrame(l)
del l
df.columns = ['Subject','Sub_Page','Access','Agent']
df.head()
train = pd.concat([train,df],axis=1)
del train['Page']
del df
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
def lag_arr(arr, lag,fill):
    filler = np.full((arr.shape[0],lag,1),-1)
    comb = np.concatenate((filler,arr),axis=1)
    result = comb[:,:arr.shape[1]]
    return result
def single_autocorr(series, lag):
    """
    Autocorrelation for single data series
    :param series: traffic series
    :param lag: lag, days
    :return:
    """
    s1 = series[lag:]
    s2 = series[:-lag]
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2
    divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
    return np.sum(ds1 * ds2) / divider if divider != 0 else 0
def batc_autocorr(data,lag,series_length):
    corrs = []
    for i in range(data.shape[0]):
        c = single_autocorr(data, lag)
        corrs.append(c)
    corr = np.array(corrs)
    corr = corr.reshape(-1,1)
    corr = np.expand_dims(corr,-1)
    corr = np.repeat(corr,series_length,axis=1)
    return corr
train.head()
weekdays = [datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%a') 
            for date in train.columns.values[:-4]]

day_one_hot = LabelEncoder().fit_transform(weekdays)
day_one_hot = day_one_hot.reshape(-1, 1)
day_one_hot = OneHotEncoder(sparse=False).fit_transform(day_one_hot)
day_one_hot = np.expand_dims(day_one_hot,0)
agent_int = LabelEncoder().fit(train['Agent'])
agent_enc = agent_int.transform(train['Agent'])
agent_enc = agent_enc.reshape(-1, 1)
agent_one_hot = OneHotEncoder(sparse=False).fit(agent_enc)

del agent_enc
page_int = LabelEncoder().fit(train['Sub_Page'])
page_enc = page_int.transform(train['Sub_Page'])
page_enc = page_enc.reshape(-1, 1)
page_one_hot = OneHotEncoder(sparse=False).fit(page_enc)

del page_enc
acc_int = LabelEncoder().fit(train['Access'])
acc_enc = acc_int.transform(train['Access'])
acc_enc = acc_enc.reshape(-1, 1)
acc_one_hot = OneHotEncoder(sparse=False).fit(acc_enc)

del acc_enc
def get_batch(train,start=0,lookback = 100,return_target = True):
    #assert((start + lookback) <= (train.shape[1] - 5)) , 'End of lookback would be out of bounds'
    
    data = train.iloc[:,start:start + lookback].values
    target = np.zeros(data.shape[1])
    if return_target:
        target = train.iloc[:,start + lookback+1].values
    target = np.log1p(target)
    
    log_view = np.log1p(data)
    log_view = np.expand_dims(log_view,axis=-1)
    
    days = day_one_hot[:,start:start + lookback]
    days = np.repeat(days,repeats=train.shape[0],axis=0)
    
    year_lag = lag_arr(log_view,365,-1)
    halfyear_lag = lag_arr(log_view,182,-1)
    quarter_lag = lag_arr(log_view,91,-1)
    
    agent_enc = agent_int.transform(train['Agent'])
    agent_enc = agent_enc.reshape(-1, 1)
    agent_enc = agent_one_hot.transform(agent_enc)
    agent_enc = np.expand_dims(agent_enc,1)
    agent_enc = np.repeat(agent_enc,lookback,axis=1)
    
    page_enc = page_int.transform(train['Sub_Page'])
    page_enc = page_enc.reshape(-1, 1)
    page_enc = page_one_hot.transform(page_enc)
    page_enc = np.expand_dims(page_enc, 1)
    page_enc = np.repeat(page_enc,lookback,axis=1)
    
    acc_enc = acc_int.transform(train['Access'])
    acc_enc = acc_enc.reshape(-1, 1)
    acc_enc = acc_one_hot.transform(acc_enc)
    acc_enc = np.expand_dims(acc_enc,1)
    acc_enc = np.repeat(acc_enc,lookback,axis=1)
    
    year_autocorr = batc_autocorr(data,lag=365,series_length=lookback)
    halfyr_autocorr = batc_autocorr(data,lag=182,series_length=lookback)
    quarter_autocorr = batc_autocorr(data,lag=91,series_length=lookback)
    
    medians = np.median(data,axis=1)
    medians = np.expand_dims(medians,-1)
    medians = np.expand_dims(medians,-1)
    medians = np.repeat(medians,lookback,axis=1)
    
    batch = np.concatenate((log_view,
                            days, 
                            year_lag, 
                            halfyear_lag, 
                            quarter_lag,
                            page_enc,
                            agent_enc,
                            acc_enc, 
                            year_autocorr, 
                            halfyr_autocorr,
                            quarter_autocorr, 
                            medians),axis=2)
    
    return batch, target
def generate_batches(train,batch_size = 32, lookback = 100):
    num_samples = train.shape[0]
    num_steps = train.shape[1] - 5
    while True:
        for i in range(num_samples // batch_size):
            batch_start = i * batch_size
            batch_end = batch_start + batch_size

            seq_start = np.random.randint(num_steps - lookback)
            X,y = get_batch(train.iloc[batch_start:batch_end],start=seq_start)
            yield X,y
from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, Dense, Activation, GlobalMaxPool1D, Flatten, BatchNormalization
max_len = 100
n_features = 29
model = Sequential()

model.add(BatchNormalization(input_shape=(max_len,n_features)))
model.add(Conv1D(16,5))
model.add(Activation('relu'))
model.add(MaxPool1D(5))

model.add(Conv1D(16,5))
model.add(Activation('relu'))
model.add(MaxPool1D(5))

model.add(Flatten())
model.add(Dense(1))
model.add(Activation('relu'))
model.compile(optimizer='adam',loss='mean_squared_error')
from sklearn.model_selection import train_test_split
batch_size = 500
train_df, val_df = train_test_split(train, test_size=0.1)
train_gen = generate_batches(train_df,batch_size=batch_size)
val_gen = generate_batches(val_df, batch_size=batch_size)

n_train_samples = train_df.shape[0]
n_val_samples = val_df.shape[0]
np.warnings.filterwarnings('ignore')
a,b = next(train_gen)
model.fit_generator(train_gen, 
                    epochs=2,
                    steps_per_epoch=n_train_samples // batch_size, 
                    validation_data= val_gen, 
                    validation_steps=n_val_samples // batch_size)
test = pd.read_csv('../input/Test.csv').fillna(0)
test = test.drop(test.columns[0], axis=1)
test.head()

l = list(test.Page.apply(parse_page))
df = pd.DataFrame(l)
del l
df.columns = ['Subject','Sub_Page','Access','Agent']
df.head()
test = pd.concat([test,df],axis=1)
pge = test['Page']
del test['Page']
del df
def split(id_string):
    d = id_string[:10]
    name = id_string[11:]
    return d,name
sample_sub = pd.read_csv('../input/Sample_Sub.csv')
sample_sub.head()

l = list(sample_sub.Id.apply(split))
df = pd.DataFrame(l)
del l
df.columns = ['Date','Page']
df.head()
assert sorted(df.Page.unique()) == sorted(test.Page.values)
test[test.columns[-4:]].head()
fp = pd.DataFrame(np.zeros((1000,len(df.Date.unique()))),columns=df.Date.unique())
tst = pd.concat([test[test.columns[:-4]],fp,test[test.columns[-4:]]],axis=1)
del test
del fp
for i in range(30):
    pos = 30-i
    date = tst.columns.values[len(tst.columns)-4-pos]
    print(date)
    trn,_ = get_batch(tst,start=len(tst.columns)-100-4-pos,lookback=100,return_target=False)
    pred = model.predict(trn)
    tst[date] = pred
tst.head()
tst.columns[-34:-4]
ids = []
data = []
for i in range(30):
    i = -34+i
    for j in range(len(pge)):
        id_str = str(tst.columns[i])+'_'+str(pge.values[j])
        val = tst.iloc[j,i]
        ids.append(id_str)
        data.append(val)
sub = pd.DataFrame({'Id':ids,'Expected':data})
sub.head()
sub.to_csv('Submission.csv',index=False)
sample_sub.head()
