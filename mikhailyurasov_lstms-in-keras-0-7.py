import re

import pandas as pd

import numpy as np

from sklearn.feature_extraction import DictVectorizer
np.random.seed(777)
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
all_df = train_df.append(test_df)

del all_df['type']

del all_df['id']
# 1h-vectorize input categories

vec = DictVectorizer()

X = vec.fit_transform(all_df.to_dict('records')).toarray()
X_train = X[:len(train_df)]

X_test = X[len(train_df):]
y_train = vec.fit_transform(train_df['type'].to_frame().to_dict('records')).toarray()
import keras

from keras.models import *

from keras.layers import *
model = Sequential([

        InputLayer(input_shape=(X_train.shape[1],)),

        Reshape(target_shape=(X_train.shape[1], 1)),

        LSTM(64, return_sequences=False, dropout_U=0.0, dropout_W=0.0),

        Dropout(0.5),

        Dense(128, activation='relu'),

        Dropout(0.5),

        Dense(y_train.shape[1], activation='softmax')

    ])
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(

    X_train, y_train,

#     validation_split=.15,

    shuffle=True,

    batch_size=32,

    nb_epoch=10, # set to 1000...2000

    verbose=True, # set to False

    callbacks=[

        # keras.callbacks.TensorBoard(log_dir='/tmp/goblins/A', histogram_freq=0)

    ]

)
# create map of type index to type name



train_df['type_ix'] = [np.argmax(x) for x in y_train]



ix_to_type = {}



for r in train_df.iterrows():

    t = r[1]['type']

    tix = r[1]['type_ix']

    ix_to_type[tix] = t

    

ix_to_type
y_p = model.predict(X_test)

y_p = np.argmax(y_p, axis=1)

y_p = [ix_to_type[x] for x in y_p]
subm_df = pd.DataFrame({'id': test_df['id'], 'type': y_p}).set_index('id')

subm_df.to_csv('subm.csv')