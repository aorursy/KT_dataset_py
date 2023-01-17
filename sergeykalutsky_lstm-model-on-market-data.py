import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from kaggle.competitions import twosigmanews
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import pad_sequences
from itertools import chain
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
limit = 2000000
market_train = market_train_df.tail(limit)
del market_train_df, news_train_df
gc.collect()
def market_preprocessing(market_df):
    
    market_df['time'] = market_df['time'].dt.floor('1D')
    
    num_cols = [col for col in market_df.columns if col not in ['universe', 'time', 'assetCode', 'assetName']]
    market_df.loc[:, num_cols] = market_df.loc[:, num_cols].fillna(0)
    
    for col in num_cols:
        market_df = market_df[np.abs(market_df[col]-market_df[col].mean()) \
                                <= (3*market_df[col].std())]
        
#     https://www.kaggle.com/rspadim/parse-ric-stock-code-exchange-asset
    market_df['assetCode_exchange'] = market_df['assetCode']
    map_a = {}
    for i in market_df['assetCode'].unique():
        _, a = i.split('.')
        map_a[i] = a
    market_df['assetCode_exchange'] = market_df['assetCode_exchange'].map(map_a)
    
    market_df = one_hot_encode(market_df, ['assetCode_exchange'])
    
    try:
        market_df.returnsOpenNextMktres10 = market_df.returnsOpenNextMktres10.clip(-1, 1)
        market_df['label'] = market_df.returnsOpenNextMktres10.map(lambda x: 0 if x < 0 else 1)
    except:
        pass
    
    return market_df

def one_hot_encode(df, columns):
    
    categorical_df = pd.get_dummies(df[columns].astype(str))
    df.drop(columns=columns, inplace=True)
    df = pd.concat([df, categorical_df], axis=1)  
    
    del categorical_df
    
    return df

def split(*arg, test_size=0.25):
    sets = []
    for i in range(len(arg)):
        data = arg[i]
        limit = int(len(data) * (1 - test_size))
        sets.append(data[:limit].copy())
        sets.append(data[limit+1000:].copy())
    return sets
%%time
print(market_train.shape)
market_train = market_preprocessing(market_train)
print(market_train.shape)
market_train.sort_values(by=['time'], inplace=True)
num_cols = [col for col in market_train.columns if col not in ['time','assetCode', 'universe', 'label', 'assetName',
                                                               'assetCode_exchange_A', 'assetCode_exchange_N', 'assetCode_exchange_O',
                                                               'assetCode_exchange_OB', 'assetCode_exchange_UNKNOWN', 'returnsOpenNextMktres10']]
scaler = StandardScaler(copy=False)
market_train[num_cols] = market_train[num_cols].fillna(0)
market_train.loc[:, num_cols] = scaler.fit_transform(market_train.loc[:, num_cols])
class SequenceGenerator:
    def __init__(self, df, cols, window=10, batch_size=64, train=True):
        self.groupby_obj = df.groupby(['assetCode'], sort=False)
        self.cols = cols
        self.batch_size = batch_size
        self.train = train
        self.window = window

    def generate(self):
        
        while True:
            
            X, y, d, r, u = [], [], [], [], []
            
            for _, data in self.groupby_obj:
                
                data = data.sort_values(by=['time'])
                num_sequences = data.shape[0] - self.window 
                
                for seq in range(num_sequences):
                    X.append(data[self.cols].iloc[seq:seq+self.window].values)
                    y.append(data.label.iloc[seq+self.window-1])
                    d.append(data.time.iloc[seq+self.window-1])
                    r.append(data.returnsOpenNextMktres10.iloc[seq+self.window-1])
                    u.append(data.universe.iloc[seq+self.window-1])
                    
                    if len(X) == self.batch_size:
                        X_, y_, = np.array(X), np.array(y) 
                        r_, u_, d_ = np.array(r),np.array(u), np.array(d)
                        X, y, d, r, u = [], [], [], [], []
                        if self.train:
                            yield X_, y_
                        else:
                            yield X_, y_, r_, u_, d_
                            
    def steps(self):
        # get number of steps per epoch
        steps = 0
        for _, data in self.groupby_obj:
            num_sequences = data.shape[0] - self.window 
            steps += num_sequences//self.batch_size
        return steps
train_df, val_df = split(market_train)
cols = [col for col in market_train.columns if col not in ['time','assetCode', 'universe', 'label', 'assetName', 'returnsOpenNextMktres10']]
train_gen = SequenceGenerator(train_df, cols, batch_size=64)
test_gen = SequenceGenerator(val_df, cols, batch_size=64)
train_steps = train_gen.steps()
test_steps = test_gen.steps()
#test on a fixed validation set
test = SequenceGenerator(val_df, cols, batch_size=100000, train=False)
X_val, y_val, r_val, u_val, d_val =  next(test.generate())
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, LSTM, GRU
from keras.losses import binary_crossentropy, mse


model = Sequential()
model.add(LSTM(16, input_shape=(10, len(cols)), return_sequences=False))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam',loss=binary_crossentropy, metrics=['accuracy'])
from keras.callbacks import EarlyStopping, ModelCheckpoint

check_point = ModelCheckpoint('model.hdf5',verbose=True, save_best_only=True)
early_stop = EarlyStopping(patience=5,verbose=True)
model.fit_generator(train_gen.generate(),
          validation_data=test_gen.generate(),
          epochs=1,
          steps_per_epoch=train_steps, 
          validation_steps=test_steps,
          callbacks=[early_stop,check_point]) 
# distribution of confidence that will be used as submission
model.load_weights('model.hdf5')
confidence_valid = np.array([model.predict(X.reshape(1,10,16))[0][0] for X in X_val]) * 2 - 1
print(accuracy_score(confidence_valid>0, r_val>0))
fig, ax = plt.subplots(1,2, figsize=(10, 5))
ax[0].hist(confidence_valid, bins='auto')
ax[0].set_title("Predicted Confidence")
ax[1].hist(r_val, bins='auto')
ax[1].set_title("returnsOpenNextMktres10")
plt.show()
# https://www.kaggle.com/davero/market-data-only-baseline
# calculation of actual metric that is used to calculate final score
x_t_i = confidence_valid * r_val * u_val
data = {'day' : d_val, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)