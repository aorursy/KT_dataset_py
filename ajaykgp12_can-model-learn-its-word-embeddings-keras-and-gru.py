# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os,gc

print(os.listdir("../input"))







from keras.optimizers import RMSprop, Adam

from keras.models import Model, Sequential

from keras.layers import Flatten, Dense, Embedding, Dropout

from keras.callbacks import EarlyStopping

from keras import losses, metrics, optimizers

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split



pd.options.display.max_colwidth = 500

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')

print('Train Shape{} Test Shape{}'.format(train.shape, test.shape))

train.head()
def clean_special_chars(text, punct, mapping):

        for p in mapping:

            text = text.replace(p, mapping[p])    

        for p in punct:

            text = text.replace(p, f' {p} ')     

        return text



def clean_text(df):

    df['comment_text'] = df['comment_text'].apply(lambda x: x.lower())

    punct_mapping = {"_":" ", "`":" "}

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    df['comment_text'] = df['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

    return df
train = clean_text(train)

test = clean_text(test)

train.head() 


max_words = 20000

maxlen = 220





tokenizer = Tokenizer(num_words = max_words, lower = True)

tokenizer.fit_on_texts(list(train['comment_text']) + list(test['comment_text']))

word_index = tokenizer.word_index



y = train.target.apply(lambda x: 0 if x < 0.5 else 1)



X = tokenizer.texts_to_sequences(list(train['comment_text']))

X_test =  tokenizer.texts_to_sequences(list(test['comment_text']) )  

                                       

X = pad_sequences(X, maxlen= maxlen)   

X_test =  pad_sequences(X_test, maxlen= maxlen)      



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2,  random_state = 42,stratify = y )

del(train, test, X,y)

gc.collect()
from keras.optimizers import RMSprop, Adam

from keras.layers import Embedding, Input, Dense, CuDNNGRU,concatenate, Bidirectional, SpatialDropout1D, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D

#Model Architecture from https://www.kaggle.com/taindow/simple-cudnngru-python-keras

embedding_dim = 30



def get_model():

    sequence_input = Input(shape=(maxlen,), dtype='int32')

    embedding_layer = Embedding(len(tokenizer.word_index) + 1,

                                embedding_dim,                          

                                input_length=maxlen)



    x = embedding_layer(sequence_input)

    x = SpatialDropout1D(0.2)(x)

    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)   

    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)



    avg_pool1 = GlobalAveragePooling1D()(x)

    max_pool1 = GlobalMaxPooling1D()(x)     



    x = concatenate([avg_pool1, max_pool1])



    preds = Dense(1, activation='sigmoid')(x)





    model = Model(sequence_input, preds)



    model.compile(loss='binary_crossentropy',

                  optimizer=Adam(),

                  metrics=['acc'])

    return model

model = get_model()

print(model.summary())

EPOCHS = 15

BATCH_SIZE = 512

print('Train Size{}, Validation Size {}, Test Size {}'.format(X_train.shape, X_valid.shape, X_test.shape))

print('Max Words {}, Max Length {}, Embedding Dimesions {}'.format(max_words, maxlen, embedding_dim))

history = model.fit( X_train,

                     y_train,

                    epochs = EPOCHS,

                    batch_size = BATCH_SIZE,

                    callbacks = [EarlyStopping(monitor = 'val_acc', patience = 3)],

                    validation_data = (X_valid, y_valid)

                   )


y_pred = model.predict(X_valid)

print('Validation ROC AUC Score', roc_auc_score(y_valid, y_pred))
history_dict = history.history

valid_acc = history_dict['val_acc'] 

best_epoch = valid_acc.index(max(valid_acc)) + 1

best_acc =  max(valid_acc)

print('Best Accuracy Score {}, is for epoch {}'.format( best_acc, best_epoch))



model = get_model()

history = model.fit( X_train,

                     y_train,

                    epochs = best_epoch,

                    batch_size = BATCH_SIZE,

                    validation_data = (X_valid, y_valid)

                   )


y_pred = model.predict(X_valid)

print('Validation ROC AUC Score', roc_auc_score(y_valid, y_pred))
y_pred = model.predict(X_test)

sub = pd.read_csv('../input/sample_submission.csv')

sub['prediction'] = y_pred

sub.to_csv('submission.csv', index = False)



sub.head()