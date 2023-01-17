import pandas as pd
import numpy as np
import os
df_train = pd.read_json('../input/train.json')
df_test = pd.read_json('../input/test.json')
df_train.head()
'''
NxTxD
T = 10 #rows, seq. length
D = 128 #columns, input dim.
N = 1195 (train) #samples
k = 2 {0,1} #classes
M = ?, LSTM latent dim.
'''
T = 10
D = 128
N = None
k = 1
M = 15
from keras.layers import LSTM, Bidirectional, Input, Dense, Flatten, BatchNormalization
from keras.models import Model
main_input = Input(shape=(T,D,))
lstm = Bidirectional(LSTM(M, return_sequences=True))(main_input)
flat = Flatten()(lstm)
bn = BatchNormalization()(flat)
main_output = Dense(1, activation='sigmoid')(bn)
model = Model(inputs=[main_input], outputs=[main_output])
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
from keras.preprocessing.sequence import pad_sequences

X_train_list = []
y_train_list = []
for i in range(len(df_train)):
    X_train = np.array(df_train.loc[i,'audio_embedding'])
    #If the sample doesn't have 10 seconds, we instert zeros at the end of the sequence
    X_train = pad_sequences(X_train.T, maxlen=10, padding='post').T
    X_train = X_train.reshape(10,128)
    X_train_list.append(X_train)
    
    y_train = np.array(df_train.loc[i,'is_turkey'])
    y_train = y_train.reshape(1,)
    y_train_list.append(y_train)
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True)

X_train_list = np.array(X_train_list)
y_train_list = np.array(y_train_list)

for train_index, test_index in kf.split(X_train_list):
    Xtrain = X_train_list[train_index]
    Xtest = X_train_list[test_index]
    ytrain = y_train_list[train_index]
    ytest = y_train_list[test_index]
    
    model.fit(np.array(Xtrain), np.array(ytrain), epochs=10, batch_size=128, validation_data=(Xtest, ytest))

df_test.head()
pred_list = []
for i in range(len(df_test)):
    pred = np.array(df_test.loc[i,'audio_embedding'])
    pred = pad_sequences(pred.T, maxlen=10, padding='post').T
    pred = pred.reshape(10,128)
    pred_list.append(pred)
preds = model.predict(np.array(pred_list))
df_test['is_turkey'] = preds
df_test[['vid_id', 'is_turkey']].to_csv('my_dinner.csv', index=False)
