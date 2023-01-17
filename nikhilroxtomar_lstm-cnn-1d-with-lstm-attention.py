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
from keras.models import *
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
maxlen = 60
max_features = 30000
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
def attention_3d_block(inputs, name):
    # inputs.shape = (batch_size, time_steps, input_dim)
    TIME_STEPS = inputs.shape[1].value
    SINGLE_ATTENTION_VECTOR = False
    
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name=name)(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul
embedding_index = dict()
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt',encoding='utf8')

for line in f:
    
    values = line.split(" ")
    words = values[0]
   
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[words]= coefs
    
f.close()
print('Loaded %s word vectors.' % len(embedding_index))

embedding_matrix = np.zeros((max_features, embed_size))
for word, index in tokenizer.word_index.items():
    if index > max_features - 1:
        break
    else:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
def model1(init):
    x = init
    x = Conv1D(64, 3,strides=2,padding='same',activation='relu')(x)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = attention_3d_block(x, 'attention_vec_1')
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = attention_3d_block(x, 'attention_vec_2')
    x = GlobalMaxPool1D()(x)
    out = Dense(64, activation="relu")(x)
    return out

def m2_block(init, filter, kernel, pool):
    x = init
    
    x = Conv1D(filter, kernel, padding='same', kernel_initializer='he_normal', activation='elu')(x)
    skip = x
    x = Conv1D(filter, kernel, padding='same', kernel_initializer='he_normal', activation='elu')(x)
    x = Conv1D(filter, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, skip])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool)(x)
    
    x = Flatten()(x)
    x = BatchNormalization()(x)
    
    return x

def model2(init):
    #init = Reshape((maxlen, embed_size, 1))(init)
    
    # pool = maxlen - filter + 1
    x0 = m2_block(init, 32, 1, maxlen - 1 + 1)
    x1 = m2_block(init, 32, 2, maxlen - 2 + 1)
    x2 = m2_block(init, 32, 3, maxlen - 3 + 1)
    x3 = m2_block(init, 32, 5, maxlen - 5 + 1)
    
    x = concatenate([x0, x1, x2, x3])
    x = Dropout(0.5)(x)
    out = Dense(64, activation="relu")(x)
    return out


def get_model():
    inp = Input(shape=(maxlen, ))
    #x = Embedding(max_features, embed_size)(inp)
    x = Embedding(input_dim=max_features, output_dim= embed_size , input_length=maxlen,weights=[embedding_matrix], trainable=False)(inp)
    
    out1 = model1(x)
    out2 = model2(x)
    
    conc = concatenate([out1, out2])
    
    #conc = out1
    x = Dropout(0.4)(conc)
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
epochs = 5
from sklearn.model_selection import train_test_split
X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, test_size = 0.07, random_state=42)
early_stopping = EarlyStopping(patience=3, verbose=1)
model_checkpoint = ModelCheckpoint('./quora.model', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.0001, verbose=1)

#model = load_model('./quora.model')

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
