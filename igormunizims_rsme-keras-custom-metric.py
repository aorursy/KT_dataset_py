# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Métrica da competição

from keras import backend as K



def rmse(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
train_df = pd.read_csv('/kaggle/input/e-commerce-reviews/train.csv')

test_df = pd.read_csv('/kaggle/input/e-commerce-reviews/test.csv')

train_df = train_df.dropna()
from keras.preprocessing import sequence

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers import Dense, LSTM, Flatten

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight

from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
seq_size     = 50

max_tokens   = 10000

embed_dim    = 100

tokenizer = Tokenizer(num_words=max_tokens, split=' ')



text = train_df['review_comment_message'].values

tokenizer.fit_on_texts(text)



X = tokenizer.texts_to_sequences(text)  



X = pad_sequences(X, maxlen=seq_size)



Y = train_df['review_score'].values
Y
from keras.layers import Dense, concatenate, SpatialDropout1D, Input,LSTM,Bidirectional, CuDNNLSTM, Activation,Conv1D,GRU, Dropout, GlobalMaxPooling1D,Embedding

from keras.models import Model

from keras.optimizers import SGD

def get_model():

    

    sequence_input = Input(shape=(seq_size,))

    x = Embedding(max_tokens,embed_dim, trainable=True, input_length = seq_size)(sequence_input)

    x = SpatialDropout1D(0.3)(x)

    x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)

    x2 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x1)

    max_pool1 = GlobalMaxPooling1D()(x1)

    max_pool2 = GlobalMaxPooling1D()(x2)

    x = concatenate([max_pool1, max_pool2])

    preds = Dense(1, activation='relu')(x)

    model = Model(sequence_input, preds)

   

    model.compile(loss = 'mse', optimizer='adam', metrics=[rmse])

    #model.summary()

    

    return model
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.15, random_state = 42)
Y_train
base_model = get_model()



early = EarlyStopping(monitor="val_loss", mode="min", patience=3)



reduce_lr = ReduceLROnPlateau(

                monitor  = 'val_loss',

                factor   = 0.3,

                patience = 1,

                verbose  = 1,

                mode     = 'auto',

                epsilon  = 0.0001,

                cooldown = 0,

                min_lr   = 0

            )



callbacks_list = [reduce_lr, early]



# Treina o modelo

hist = base_model.fit(X_train, Y_train, 

              validation_data =(X_valid, Y_valid),

              batch_size=512, nb_epoch = 30,  verbose = 1, callbacks=callbacks_list)
def predict(text):

    new_text = tokenizer.texts_to_sequences(text)

    new_text = pad_sequences(new_text, maxlen=seq_size)

    pred     = base_model.predict(new_text)

    

    return pred
pred     = predict(test_df.review_comment_message)

pred     = pred[:,0]

pred[:5] 
test_df['review_score'] = pred

test_df.head()
test_df[['review_id', 'review_score']].to_csv('submission_test.csv', index=False)