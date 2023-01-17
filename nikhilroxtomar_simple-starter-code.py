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
train_df = pd.read_csv('../input/train.csv')
X_train = train_df["question_text"].fillna("dieter").values
test_df = pd.read_csv('../input/test.csv')
X_test = test_df["question_text"].fillna("dieter").values
y = train_df["target"]
from keras.models import Model
from keras.layers import Input, Dense, Embedding, concatenate
from keras.layers import CuDNNGRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D
from keras.layers import Add, BatchNormalization, Activation, CuDNNLSTM, Dropout
from keras.layers import *
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
maxlen = 100
max_features = 30000
embed_size=100

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

def model1(init):
    x = init
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(64))(x)
    out = Dense(64, activation="relu")(x)
    return out

def m2_block(init, filter, kernel, pool):
    x = init
    x = Conv1D(filter, kernel, padding='same')(x)
    skip = x
    x = BatchNormalization()(x)
    x = Conv1D(filter, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, skip])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool)(x)
    
    x = Flatten()(x)
    
    return x

def model2(init):
    #init = Reshape((maxlen, embed_size, 1))(init)
    
    x0 = m2_block(init, 32, 2, 2)
    x1 = m2_block(init, 32, 3, 2)
    x2 = m2_block(init, 32, 5, 2)
    x3 = m2_block(init, 32, 7, 8)
    
    x = concatenate([x0, x1, x2, x3])
    out = Dense(64, activation="relu")(x)
    return out

def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    
    out1 = model1(x)
    out2 = model2(x)
    
    conc = concatenate([out1, out2])
    x = Dropout(0.5)(conc)
    x = Dense(64, activation='relu')(x)
    x = Reshape((x.shape[1].value, 1))(x)
    x = CuDNNLSTM(32)(x)
    outp = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])    

    return model
model = get_model()
model.summary()
batch_size = 256
epochs = 20
from sklearn.model_selection import train_test_split
X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, test_size = 0.1, random_state=42)
early_stopping = EarlyStopping(patience=3, verbose=1)
model_checkpoint = ModelCheckpoint('./quora.model', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.0001, verbose=1)

hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 verbose=True, callbacks = [early_stopping, model_checkpoint, reduce_lr])
y_pred = model.predict(x_test, batch_size=1024, verbose=True)
y_te = (y_pred[:,0] > 0.5).astype(np.int)

submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})
submit_df.to_csv("submission.csv", index=False)
from IPython.display import HTML
import base64  
import pandas as pd  

def create_download_link( df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index =False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(submit_df)

