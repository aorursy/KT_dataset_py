# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data1 = pd.read_csv('/kaggle/input/LTC-USD.csv', names = ["time","LTC-low","LTC-high","LTC-open","LTC-close","LTC-volume"])





data2 = pd.read_csv('/kaggle/input/BTC-USD.csv', names = ["time","BTC-low","BTC-high","BTC-open","BTC-close","BTC-volume"])





data3= pd.read_csv('/kaggle/input/ETH-USD.csv', names = ["time","ETH-low","ETH-high","ETH-open","ETH-close","ETH-volume"])





data4 = pd.read_csv('/kaggle/input/BCH-USD.csv', names = ["time","BCH-low","BCH-high","BCH-open","BCH-close","BCH-volume"])



data2.head()
main_df = pd.DataFrame()
ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]  # the 4 ratios we want to consider

for ratio in ratios:  # begin iteration

    print(ratio)

    dataset = f'/kaggle/input/{ratio}.csv'  # get the full path to the file.

    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])  # read in specific file



    # rename volume and close to include the ticker so we can still which close/volume is which:

    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)



    df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time

    df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # ignore the other columns besides price and volume



    if len(main_df)==0:  # if the dataframe is empty

        main_df = df  # then it's just the current df

    else:  # otherwise, join this data to the main one

        main_df = main_df.join(df)



main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values

main_df.dropna(inplace=True)

print(main_df.head())  # how did we do??
SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN

FUTURE_PERIOD_PREDICT = 3  # how far into the future are we trying to predict?

RATIO_TO_PREDICT = "LTC-USD"



def classify(current, future):

    if float(future) > float(current):

        return 1

    else:

        return 0

    

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)



main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))



#print(main_df[[f"{RATIO_TO_PREDICT}_close","future"]])



main_df['target'] = list(map(classify,main_df[f"{RATIO_TO_PREDICT}_close"],main_df['future']))



print(main_df[[f"{RATIO_TO_PREDICT}_close","future","target"]])



main_df.dropna(inplace = True)

print(main_df[[f"{RATIO_TO_PREDICT}_close","future","target"]])

times = sorted(main_df.index.values)



last_5pct = times[-int(0.5*len(times))]

print(last_5pct)
validation_main_df =  main_df[(main_df.index)>=last_5pct]

main_df =  main_df[(main_df.index)<last_5pct]

from sklearn import preprocessing

from collections import deque

import random



def preprocess_df(df):

    df = df.drop('future',axis = 1)

    

    for col in df.columns:

        if col != "target":

            df[col] =df[col].pct_change()

            df = df.dropna()

            df[col] = preprocessing.scale(df[col].values)

            

    df.dropna(inplace = True)

    

    sequential_data =[]

    prev_days = deque(maxlen = SEQ_LEN)

    

    for i in df.values:

        prev_days.append([n for n in i[:-1]])

        

        if(len(prev_days)==SEQ_LEN):

            sequential_data.append([np.array(prev_days), i[-1]])

            

    random.shuffle(sequential_data)

    

    buys = []

    sell = []



    for seq, target in sequential_data:

        if target ==0:

            sell.append([seq,target])

        else:

            buys.append([seq,target])



    random.shuffle(buys)

    random.shuffle(sell)



    lower = min(len(buys),len(sell))



    buys = buys[:lower]

    sell = sell[:lower]



    sequential_data = buys+sell



    random.shuffle(sequential_data)



    X= []

    y= []



    for seq,target in sequential_data:

        X.append(seq)

        y.append(target)

        

    return np.array(X),np.array(y)

    





train_x,train_y = preprocess_df(main_df)

validation_x,validation_y = preprocess_df(validation_main_df)



print(f"train data: {len(train_x)} validation: {len(validation_x)}")

#print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")

#print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")
import time

EPOCHS = 10

BATCH_SIZE = 64

NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"  # a unique name for the model

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, BatchNormalization

from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint



model = Sequential()

model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))

model.add(Dropout(0.2))

model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.



model.add(LSTM(128, return_sequences=True))

model.add(Dropout(0.1))

model.add(BatchNormalization())



model.add(LSTM(128))

model.add(Dropout(0.2))

model.add(BatchNormalization())



model.add(Dense(32, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(2, activation='softmax'))



import tensorflow as tf

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)



# Compile model

model.compile(

    loss='sparse_categorical_crossentropy',

    optimizer=opt,

    metrics=['accuracy']

)

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))



filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch

#checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

history = model.fit(

    train_x, train_y,

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    validation_data=(validation_x, validation_y),

)

score = model.evaluate(validation_x, validation_y, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])

# Save model

model.save("models/{}".format(NAME))