# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print(os.listdir("../input/cnews/cnews/Data"))
data_path="../input/cnews/cnews/Data"

column_name=['label', 'news']

df_train=pd.read_csv(os.path.join(data_path, 'cnews.train.txt'), sep='\t', header=None, names = column_name)

df_val=pd.read_csv(os.path.join(data_path, 'cnews.val.txt'), sep='\t', header=None, names = column_name)

df_test=pd.read_csv(os.path.join(data_path, 'cnews.test.txt'), sep='\t', header=None, names = column_name)
from sklearn import preprocessing

onehot_encoder=preprocessing.OneHotEncoder()

onehot_encoder=onehot_encoder.fit(df_test[['label']])

y_train = onehot_encoder.transform(df_train[['label']])

y_val = onehot_encoder.transform(df_val[['label']])

y_test = onehot_encoder.transform(df_test[['label']])
df_train['lengh'] = df_train.news.apply(len)

df_train.describe()
embed_size = 400

max_features = 10000

maxlen = 1500



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



all_news=df_train.news.append(df_val.news).append(df_test.news)

tokenizer = Tokenizer(max_features, char_level=True)

tokenizer.fit_on_texts(list(all_news.values))



def text_to_sequences(textlist):

    sequences = tokenizer.texts_to_sequences(textlist)

    return pad_sequences(sequences, maxlen = maxlen) 



X_train = text_to_sequences(list(df_train.news.values))

X_val = text_to_sequences(list(df_val.news.values))

X_test = text_to_sequences(list(df_test.news.values))
from keras.models import Sequential 

from keras.optimizers import RMSprop 

from keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, BatchNormalization



model = Sequential() 

model.add(Embedding(max_features, embed_size, input_length = maxlen)) 

model.add(Conv1D(256, 8, activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling1D(3)) 

model.add(Conv1D(128, 4, activation='relu'))

model.add(BatchNormalization())

model.add(GlobalMaxPooling1D())

model.add(Dense(10, activation='softmax'))

model.compile(optimizer=RMSprop(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

callback_list = [

    EarlyStopping(monitor='acc',patience=10 ),

    ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True),

    ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=3)

]
history = model.fit(X_train, y_train, epochs=500, 

                    batch_size=1024, callbacks = callback_list, 

                    validation_data = (X_val, y_val))
history_dict = history.history



loss_train = history_dict['loss']

loss_val = history_dict['val_loss']

acc_train = history_dict['acc']

acc_val = history_dict['val_acc']

epochs = range(1, len(loss_val) + 1)
import matplotlib.pyplot as plt



plt.plot(epochs, loss_train, 'bo', label='Training loss')

plt.plot(epochs, loss_val, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
plt.plot(epochs, acc_train, 'bo', label='Training acc')

plt.plot(epochs, acc_val, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
model.metrics_names
model.evaluate(X_test, y_test)
from keras.layers import Bidirectional, CuDNNGRU, Dense, GlobalMaxPooling1D



model = Sequential()

model.add(Embedding(max_features, embed_size, input_length = maxlen))

model.add(Bidirectional(CuDNNGRU(256, return_sequences=True)))

model.add(GlobalMaxPooling1D())

model.add(Dense(64, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics = ['acc'])