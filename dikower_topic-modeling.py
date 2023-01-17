# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/intenttask/intent_train.csv')

data = data.dropna()

check = pd.read_csv('/kaggle/input/intenttask/intent_check.csv')
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential, load_model

from keras.layers.core import Dense

from keras.layers import Conv1D, Input, Embedding

from keras.layers.pooling import GlobalMaxPooling1D

from keras.utils import to_categorical

from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping

from keras.models import load_model

from keras.initializers import lecun_uniform

from sklearn.model_selection import train_test_split

import pandas as pd





label_lst = sorted(data['label'].value_counts().index.to_list())

decoder = dict(enumerate(label_lst))

encoder = dict((j, i) for i, j in enumerate(label_lst))



X, Y = data['text'], to_categorical(data['label'].replace(encoder))

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)



max_words = 1000

keras_tokenizer = Tokenizer(num_words=max_words, char_level=True)

keras_tokenizer.fit_on_texts(X_train.tolist())



X_train = keras_tokenizer.texts_to_matrix(X_train)

X_val = keras_tokenizer.texts_to_matrix(X_val)



init = lecun_uniform(seed=42)

model = Sequential()

model.add(Embedding(10150, 100, input_length=1000))

model.add(Conv1D(128, 5, activation='relu', init=init))

model.add(GlobalMaxPooling1D())

model.add(Dense(80, activation='relu'))

model.add(Dense(30, activation='relu'))

model.add(Dense(14, activation='sigmoid'))

model.compile(

#     optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy'])



history = model.fit(X_train, Y_train,

                    epochs=4,

                    validation_data=(X_val, Y_val),

                    batch_size=64,

                    shuffle=True,

                    callbacks=[

                        ModelCheckpoint('model.hd5', save_best_only=True),

                        EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10),

                    ])

model.save_weights('weights.h5')
from keras.models import load_model

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential, load_model

from keras.layers.core import Dense

from keras.layers import Conv1D, Input, Embedding

from keras.layers.pooling import GlobalMaxPooling1D

from keras.utils import to_categorical

from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping

from keras.models import load_model

from keras.initializers import lecun_uniform

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np





def get_model():

    model = Sequential()

    model.add(Embedding(10150, 100, input_length=1000))

    model.add(Conv1D(128, 5, activation='relu', init=init))

    model.add(GlobalMaxPooling1D())

    model.add(Dense(80, activation='relu'))

    model.add(Dense(30, activation='relu'))

    model.add(Dense(14, activation='sigmoid'))

    return model



decoder = {

    0: 'FAQ - интернет',

    1: 'FAQ - тарифы и услуги',

    2: 'SIM-карта и номер',

    3: 'Баланс',

    4: 'Личный кабинет',

    5: 'Мобильные услуги',

    6: 'Мобильный интернет',

    7: 'Оплата',

    8: 'Роуминг',

    9: 'Устройства',

    10: 'запрос обратной связи',

    11: 'мобильная связь - зона обслуживания',

    12: 'мобильная связь - тарифы',

    13: 'тарифы - подбор'

}





model = get_model()

model.load_weights('weights.h5')

model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy']

)

# check = pd.read_csv('/kaggle/input/intent_check.csv')

# keras_tokenizer = Tokenizer(num_words=1000, char_level=False, mode='tfidf')

tokenized_check = keras_tokenizer.texts_to_matrix(check['text'].tolist())

check['label'] = pd.Series(np.argmax(model.predict(tokenized_check), axis=1)).replace(decoder)
pd.concat([pd.Series(np.argmax(model.predict(X_val), axis=1)).replace(decoder), pd.Series(np.argmax(Y_val, axis=1)).replace(decoder)], axis=1)