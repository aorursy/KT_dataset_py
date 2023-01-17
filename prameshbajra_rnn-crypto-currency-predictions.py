import pandas as pd

import numpy as np

import random

import time

import tensorflow as tf



from sklearn import preprocessing

from collections import deque

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import CuDNNLSTM, BatchNormalization, Dense, Dropout

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
cryptos = ["BCH-USD", "BTC-USD", "ETH-USD", "LTC-USD"]

main_dataframe = pd.DataFrame()

for crypto in cryptos:

    data_frame = pd.read_csv(f"../input/{crypto}.csv", 

                             names = ["time", "low", "high", "open", "close", "volume"])

    data_frame.rename(columns = {"close": f"{crypto}_close",

                                 "volume": f"{crypto}_volumn"}, inplace = True)

    data_frame.set_index("time", inplace = True)

    data_frame = data_frame[[f"{crypto}_close", f"{crypto}_volumn"]]

    

    if len(main_dataframe) == 0:

        main_dataframe = data_frame

    else:

        main_dataframe = main_dataframe.join(data_frame)

        

main_dataframe.head()
SEQ_LEN = 60 # Seconds to look back to

FUTURE_PERIOD_PREDICT = 3 #minutes to predict into future

CRYPTO_TO_PREDICT = "BTC-USD"

BATCH_SIZE = 64

EPOCHS = 10

NAME = f"{CRYPTO_TO_PREDICT}_{SEQ_LEN}_{int(time.time())}"
def classify(current, future):

    if float(future) > float(current):

        return 1

    else:

        return 0
main_dataframe["future"] = main_dataframe[f"{CRYPTO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)

main_dataframe.head()
main_dataframe["target"] = list(map(classify, 

                                    main_dataframe[f"{CRYPTO_TO_PREDICT}_close"],

                                   main_dataframe["future"]))

main_dataframe.head()
main_dataframe_timestamp = sorted(main_dataframe.index.values)

five_percent_timestamp = main_dataframe_timestamp[-int(0.05 * len(main_dataframe_timestamp))]



print(f"Length before separation {main_dataframe.shape}")



validation_dataframe = main_dataframe[(main_dataframe.index >= five_percent_timestamp)]

main_dataframe = main_dataframe[(main_dataframe.index < five_percent_timestamp)]



print(f"Length after separation {main_dataframe.shape}")
validation_dataframe.head()
main_dataframe.head()
def seggregate_values(data_frame):

    data_frame = data_frame.drop("future", 1)

    for column in data_frame.columns:

        if column != "target":

            data_frame[column] = data_frame[column].pct_change()

            data_frame.dropna(inplace = True)

            data_frame[column] = preprocessing.scale(data_frame[column].values)

    data_frame.dropna(inplace = True)

    sequential_data = []

    prev_days = deque(maxlen = SEQ_LEN)

    for value in data_frame.values:

        prev_days.append([n for n in value[:-1]])

        if len(prev_days) == SEQ_LEN:

            sequential_data.append([np.array(prev_days), value[-1]])

    

    random.shuffle(sequential_data)

    buys = []

    sells = []

    for sequence, target in sequential_data:

        if target == 0:

            sells.append([sequence, target])

        elif target == 1:

            buys.append([sequence, target])

    

    lower_number = min(len(buys), len(sells))

    

    buys = buys[:lower_number]

    sells = sells[:lower_number]

    

    random.shuffle(buys)

    random.shuffle(sells)

    

    sequential_data = buys + sells

    random.shuffle(sequential_data)

    

    sequences = []

    labels = []

    

    for sequence, target in sequential_data:

        sequences.append(sequence)

        labels.append(target)

        

    return np.array(sequences), labels
train_sequence, train_label = seggregate_values(main_dataframe)

validation_sequence, validation_label = seggregate_values(validation_dataframe)



print(f"Train {train_sequence[0]} => {train_label[0]}")

print(f"Train {validation_sequence[0]} => {validation_label[0]}")
model = Sequential()

model.add(CuDNNLSTM(128, input_shape = (train_sequence.shape[1:]), return_sequences = True))

model.add(Dropout(0.2))

model.add(BatchNormalization())



model.add(CuDNNLSTM(128, input_shape = (train_sequence.shape[1:]), return_sequences = True))

model.add(Dropout(0.1))

model.add(BatchNormalization())



model.add(CuDNNLSTM(128, input_shape = (train_sequence.shape[1:])))

model.add(Dropout(0.2))

model.add(BatchNormalization())



model.add(Dense(32, activation = "relu"))

model.add(Dropout(0.2))



model.add(Dense(2, activation = "softmax"))



model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.001, decay = 1e-6),

             loss = "sparse_categorical_crossentropy", metrics = ['accuracy'])



tensorboard = TensorBoard(log_dir = f"logs/{NAME}")



filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  

checkpoint = ModelCheckpoint("{}.model".format(filepath,

                                                      monitor='val_acc',

                                                      verbose=1,

                                                      save_best_only=True,

                                                      mode='max')) 



history = model.fit(train_sequence, train_label, batch_size = BATCH_SIZE,

                    epochs = EPOCHS, validation_data=(validation_sequence, validation_label),

                    callbacks=[tensorboard, checkpoint])          