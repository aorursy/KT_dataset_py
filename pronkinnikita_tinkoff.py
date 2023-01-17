import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
%matplotlib inline


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
q_train = pd.read_csv('../input/train.csv')
q_test = pd.read_csv('../input/test.csv')
### delete punctuaction
table = str.maketrans({key: None for key in string.punctuation})

q_train.question1 = q_train.question1.str.translate(table).str.lower().map(str)
q_train.question2 = q_train.question2.str.translate(table).str.lower().map(str)

q_test.question1 = q_test.question1.str.translate(table).str.lower().map(str)
q_test.question2 = q_test.question2.str.translate(table).str.lower().map(str)
full_text = list(q_train.question2.values) + list(q_train.question1.values)
full_text += list(q_test.question1.values) + list(q_test.question1.values)
tk = Tokenizer(num_words=50000) 
tk.fit_on_texts(full_text)
len(tk.word_index)
train_tokenized1 = tk.texts_to_sequences(q_train.question1)
train_tokenized2 = tk.texts_to_sequences(q_train.question2)

test_tokenized1 = tk.texts_to_sequences(q_test.question1)
test_tokenized2 = tk.texts_to_sequences(q_test.question2)

max_len = 50

X_train1 = pad_sequences(train_tokenized1, maxlen = max_len)
X_train2 = pad_sequences(train_tokenized2, maxlen = max_len)

X_test1 = pad_sequences(test_tokenized1, maxlen = max_len)
X_test2 = pad_sequences(test_tokenized2, maxlen = max_len)
y = q_train.is_duplicate
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(y.values.reshape(-1, 1))
embed_size = 100
max_features = 50000
def LSTM_CNN(lr=0.0, lr_d=0.0, units=0, 
                 spatial_dr=0.0, dense_units=128, 
                 dr=0.1, conv_size=32):
    file_path = "best_model.hdf5"
    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                  save_best_only = True, mode = "min")
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)
    
    inp1 = Input(shape = (max_len,))
    inp2 = Input(shape = (max_len,))
    
    Embedding_layer = Embedding(min(len(tk.word_index), max_features),
                                embed_size, trainable = True)
    SpatialDropout1D_layer = SpatialDropout1D(spatial_dr)
    LSTM_layer = Bidirectional(CuDNNLSTM(units, return_sequences = True))
    Conv1_layer = Conv1D(conv_size, kernel_size=2, 
                         padding='valid', kernel_initializer='he_uniform')
    Conv2_layer = Conv1D(conv_size, kernel_size=3, 
                         padding='valid', kernel_initializer='he_uniform')
    Conv3_layer = Conv1D(conv_size, kernel_size=4, 
                         padding='valid', kernel_initializer='he_uniform')
    GlobalMaxPooling1D_layer = GlobalMaxPooling1D()
    
    def head_block(inp):
        x = Embedding_layer(inp)
        x = SpatialDropout1D_layer(x)
        x_lstm = LSTM_layer(x)
        x = concatenate([Conv1_layer(x_lstm),
                         Conv2_layer(x_lstm),
                         Conv3_layer(x_lstm),], axis=1)
        x = GlobalMaxPooling1D_layer(x)
        return x
    
    x1 = head_block(inp1)
    x2 = head_block(inp2)
    x = concatenate([x1, x2])
    x = BatchNormalization()(x)
    x = Dropout(dr)(Dense(dense_units, activation='relu')(x))
    x = BatchNormalization()(x)
    x = Dropout(dr)(Dense(dense_units, activation='relu')(x))
    x = Dense(2, activation = "softmax")(x)
    model = Model(inputs = [inp1, inp2], outputs = x)
    model.compile(loss = "categorical_crossentropy", optimizer = Adam(lr = lr, decay = lr_d),
                  metrics = ["categorical_accuracy"])
    model.summary()
    history = model.fit([X_train1, X_train2], 
                        y_ohe, batch_size = 128,
                        epochs = 10,
                        validation_split=0.1, 
                        verbose = 1, callbacks = [check_point, early_stop])
    model = load_model(file_path)
    return model
nn = LSTM_CNN(lr=1e-3, lr_d=1e-9, 
                      units=32, spatial_dr=0.3, 
                      dense_units=64, dr=0.5, conv_size=64)
res = nn.predict([X_test1, X_test2], batch_size=128, verbose=1)
pd.DataFrame(res).to_csv('res')
answ = pd.DataFrame(res).reset_index().iloc[:, 0, 2]
answ.columns = ['test_id', 'is_duplicate']
answ.to_csv('answ.csv', index=False)