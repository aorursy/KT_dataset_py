import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.layers import *

from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras import Sequential

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



train = pd.read_json("../input/train.json")
def plot_history(history):

    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]

    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]

    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]

    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    

    if len(loss_list) == 0:

        print('Loss is missing in history')

        return 

    

    ## As loss always exists

    epochs = range(1,len(history.history[loss_list[0]]) + 1)

    

    ## Loss

    plt.figure(1)

    for l in loss_list:

        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    for l in val_loss_list:

        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    

    plt.title('Loss')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()

    

    ## Accuracy

    plt.figure(2)

    for l in acc_list:

        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    for l in val_acc_list:    

        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')



    plt.title('Accuracy')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.legend()

    plt.show()
train.head()
for i in range(10):

    print(len(train.audio_embedding[i]), len(train.audio_embedding[i][0]), train.end_time_seconds_youtube_clip[i] - train.start_time_seconds_youtube_clip[i])
train.size
train = pd.read_json('../input/train.json')

test = pd.read_json('../input/test.json')

sample_submission = pd.read_csv('../input/sample_submission.csv')



train_train, train_val = train_test_split(train, random_state = 42)

xtrain = [k for k in train_train['audio_embedding']]

ytrain = train_train['is_turkey'].values



xval = [k for k in train_val['audio_embedding']]

yval = train_val['is_turkey'].values



# Pad the audio features so that all are "10 seconds" long

x_train = pad_sequences(xtrain, maxlen=10)

x_val = pad_sequences(xval, maxlen=10)



y_train = np.asarray(ytrain)

y_val = np.asarray(yval)



test_data = test['audio_embedding'].tolist()

x_test = pad_sequences(test_data, maxlen=10)
embed_size = 128

maxlen = 10



inp = Input(shape=(maxlen,embed_size))

x = BatchNormalization()(inp)



x = Bidirectional(GRU(128, return_sequences=True))(x)

x = Bidirectional(GRU(64, return_sequences=True))(x)



x = Flatten()(x)

x = Dense(128, activation='relu')(x)

x = Dense(64, activation='relu')(x)

# x = Dropout(0.1)(x)

outp = Dense(1, activation='sigmoid')(x)



model_1 = Model(inputs=inp, outputs=outp)



model_1.summary()
model_1.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')
# history = model_1.fit(x_train, y_train,

#           batch_size=10,

#           epochs=50,

#           validation_data=(x_val, y_val))
# plot_history(history)
embed_size = 128

maxlen = 10



inp = Input(shape=(maxlen,embed_size))

x = BatchNormalization()(inp)



x = Bidirectional(GRU(128, return_sequences=True, dropout=0.1))(x)

x = Bidirectional(GRU(64, return_sequences=True, dropout=0.1))(x)



x = Flatten()(x)

x = Dense(128, activation='relu')(x)

x = Dense(64, activation='relu')(x)

x = Dropout(0.1)(x)

outp = Dense(1, activation='sigmoid')(x)



model_2 = Model(inputs=inp, outputs=outp)



model_2.summary()
# model_2.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

# history_2 = model_2.fit(x_train, y_train,

#           batch_size=10,

#           epochs=10,

#           validation_data=(x_val, y_val))
# plot_history(history_2)
# history_3 = model_2.fit(x_train, y_train,

#           batch_size=10,

#           epochs=50,

#           validation_data=(x_val, y_val))

# plot_history(history_3)
embed_size = 128

maxlen = 10



inp = Input(shape=(maxlen,embed_size))

x = BatchNormalization()(inp)

x = Reshape((maxlen,embed_size,1))(x)



cnn_1 = Conv2D(42, kernel_size=(3, 16), kernel_initializer='he_normal', activation='tanh')(x)



cnn_2 = Conv2D(42, kernel_size=(2, 8), kernel_initializer='he_normal', activation='tanh')(cnn_1)



max_pool = GlobalMaxPooling2D()(cnn_2)

avg_pool = GlobalAveragePooling2D()(cnn_2)



concat = concatenate([max_pool, avg_pool])



x = Flatten()(concat)

x = Dense(128, activation='relu')(x)

x = Dense(64, activation='relu')(x)

x = Dropout(0.1)(x)

outp = Dense(1, activation='sigmoid')(x)



model_4 = Model(inputs=inp, outputs=outp)



model_4.summary()
# model_4.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

# #fit on a portion of the training data, and validate on the rest

# history_4 = model_4.fit(x_train, y_train,

#           batch_size=10,

#           epochs=20,

#           validation_data=(x_val, y_val))
# plot_history(history_4)
embed_size = 128

maxlen = 10



inp = Input(shape=(maxlen,embed_size))

x = BatchNormalization()(inp)

x = Bidirectional(GRU(128, return_sequences=True, dropout=0.1))(x)

x = Bidirectional(GRU(64, return_sequences=True, dropout=0.1))(x)

x = Reshape((maxlen,embed_size,1))(x)



cnn_1 = Conv2D(42, kernel_size=(3, 16), kernel_initializer='he_normal', activation='tanh')(x)

cnn_2 = Conv2D(42, kernel_size=(2, 8), kernel_initializer='he_normal', activation='tanh')(cnn_1)



max_pool = GlobalMaxPooling2D()(cnn_2)

avg_pool = GlobalAveragePooling2D()(cnn_2)



concat = concatenate([max_pool, avg_pool])



x = Flatten()(concat)

x = Dense(128, activation='relu')(x)

x = Dense(64, activation='relu')(x)

x = Dropout(0.5)(x)

outp = Dense(1, activation='sigmoid')(x)



model_5 = Model(inputs=inp, outputs=outp)



model_5.summary()
model_5.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

#fit on a portion of the training data, and validate on the rest

history_5 = model_5.fit(x_train, y_train,

          batch_size=10,

          epochs=20,

          validation_data=(x_val, y_val))
plot_history(history_5)
embed_size = 128

maxlen = 10



inp = Input(shape=(maxlen,embed_size))

x = BatchNormalization()(inp)

x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)

x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)

x = Reshape((maxlen,embed_size,1))(x)



cnn_1 = Conv2D(42, kernel_size=(3, 16), kernel_initializer='he_normal', activation='tanh')(x)

cnn_2 = Conv2D(42, kernel_size=(2, 8), kernel_initializer='he_normal', activation='tanh')(cnn_1)



max_pool = GlobalMaxPooling2D()(cnn_2)

avg_pool = GlobalAveragePooling2D()(cnn_2)



concat = concatenate([max_pool, avg_pool])



x = Flatten()(concat)

x = Dense(128, activation='relu')(x)

x = Dense(64, activation='relu')(x)

x = Dropout(0.5)(x)

outp = Dense(1, activation='sigmoid')(x)



model_6 = Model(inputs=inp, outputs=outp)



model_6.summary()
# model_6.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

# #fit on a portion of the training data, and validate on the rest

# history_6 = model_6.fit(x_train, y_train,

#           batch_size=10,

#           epochs=20,

#           validation_data=(x_val, y_val))
# plot_history(history_6)
y_test = model_5.predict(x_test)
submission = pd.DataFrame({'vid_id': test['vid_id'].values, 'is_turkey': list(y_test.flatten())})

submission.head(20)
submission.to_csv("submission.csv", index=False)