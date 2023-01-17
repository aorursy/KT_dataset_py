import pandas as pd
import os
df = pd.read_csv('../input/BCH-USD.csv')
df.head()
main_df = pd.DataFrame()
ratios = ['BTC-USD', 'LTC-USD', 'ETH-USD', 'BCH-USD']

for ratio in ratios:
    # note "f" is the new replacement for .format
    dataset = f'../input/{ratio}.csv' # set dataset path
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])
    df.rename(columns={"close":f"{ratio}_close", "volume":f"{ratio}_volume"}, inplace=True)
    
    # set time as index column
    df.set_index("time", inplace=True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]
    
    # merge all df's 
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

print(main_df.head())
for c in main_df.columns:
    print(c)
# define parameters

SEQ_LEN = 60 # number of min. data we use to predict
FUTURE_PERIOD_PREDICT = 3 # future period to predict
RATION_TO_PREDICT = "ETH-USD" # currency we will predict preice for

def classify(current, future):
    if float(future) > float(current):
        return 1 # price will rise
    else:
        return 0  # price will go down
main_df['future'] = main_df[f"{RATION_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
print(main_df[[f"{RATION_TO_PREDICT}_close", "future"]].head())
# create trarget column
 
main_df['target'] = list(map(classify, main_df[f"{RATION_TO_PREDICT}_close"], main_df['future']))
print(main_df[[f"{RATION_TO_PREDICT}_close", "future", 'target']].head())
times = sorted(main_df.index.values)
last_5pct = times[-int(0.05 * len(times))]
print(last_5pct)
validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

# this fun. for scaling , normaize and balance data
from sklearn import preprocessing
from collections import deque
import numpy as np
import random
import time
def preprocess_df(df):
    df = df.drop('future', 1) # this is only for testing and generate target
    
    for col in df.columns:
        if col != 'target':
            # pct_change compute precent of change between prev. and immediate value
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values) # scaling values 0-1
    df.dropna(inplace=True)
    
    
    # this part create a stack or queue of elements and in each iter. remove
    # one time series and add another one in the end, append everystck when len be 60
    sequencial_data = []
    prev_days = deque(maxlen=SEQ_LEN) # make queue or stack of data
    
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN :
            sequencial_data.append([np.array(prev_days) , i[-1]])
    
    random.shuffle(sequencial_data)
    
    buys = []
    sells = []
    
    for seq, target in sequencial_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
        
    random.shuffle(buys)
    random.shuffle(sells)
    
    lower = min(len(buys), len(sells))
    
    buys = buys[:lower]
    sells = sells[:lower]
    # print(f"sq data len:{len(sequencial_data)}, balanced len: {len(buys) + len(sells)}")
    sequencial_data = buys + sells
    
    random.shuffle(sequencial_data)
    
    # split data into x, y
    x = []
    y = []
    
    for seq, target in sequencial_data:
        x.append(seq)
        y.append(target)
        
    return np.array(x), y
train_x , train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)
print(f"training data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"validation dont buys: {validation_y.count(0)} buys: {validation_y.count(1)}")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

# define model parameter
EPOCHS = 10
BATCH_SIZE = 32
NAME = f"{RATION_TO_PREDICT}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"
model = Sequential()
# add layers to model
# if you using cpu version use LSTM not CuDNNLSTM
model.add(CuDNNLSTM(64, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(64, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(64, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(6,activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',
             optimizer=opt,
             metrics=['accuracy'])

# tensorbord = TensorBoard(log_dir=f'log/{NAME}')

# filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}" # uniqe file name that will incluse the epoch and the validation acc for that ecpoch
# checkpoint = ModelCheckpoint("../input/{}.model".format(filepath, monitor='val_acc', verbos=1, save_best_only=True, mod='max'))

history = model.fit(
        train_x, train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(validation_x, validation_y))
# model.save("")