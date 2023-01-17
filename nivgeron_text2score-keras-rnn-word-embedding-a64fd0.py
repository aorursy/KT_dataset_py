import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from keras.preprocessing import sequence, text
from keras.models import Sequential
import keras.layers as layer 
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras import backend as K

from sklearn.model_selection import train_test_split
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_all = pd.concat([df_train, df_test])
del(df_train, df_test)
text_features = ['title', 'description']
df_all = df_all[text_features + ['deal_probability']]
df_all['text'] = ""
for text_col in text_features:
    df_all['text'] += " " + df_all[text_col].fillna("")
    
pattern = re.compile('[^(?u)\w\s]+')
df_all['text'] =df_all['text'].apply(lambda x: re.sub(pattern, "", x).lower())
max_len = 30
tk = text.Tokenizer(num_words=50000)
tk.fit_on_texts(df_all['text'].str.lower().tolist())
X = tk.texts_to_sequences(df_all['text'].str.lower().values)
X = sequence.pad_sequences(X, maxlen=max_len)
df_all.drop(text_features, axis=1, inplace=True)
word_index = tk.word_index
df_train = df_all[df_all['deal_probability'].notnull()]
X_train, X_val, y_train, y_val  = train_test_split(X[:len(df_train)], df_train['deal_probability'].values, test_size=0.01)
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
def get_model():

    model = Sequential()
    model.add(layer.Embedding(len(word_index) + 1, 30, input_shape=(max_len,)))
    model.add(layer.LSTM(30, recurrent_dropout=0.2, dropout=0.2, kernel_regularizer=regularizers.l2(2e-5),
                activity_regularizer=regularizers.l1(2e-5)))
    
    model.add(layer.Dense(32,  kernel_regularizer=regularizers.l2(2e-5),
                activity_regularizer=regularizers.l1(2e-5)))
    model.add(layer.PReLU())
    model.add(layer.Dropout(0.2))
    model.add(layer.BatchNormalization())
    
    model.add(layer.Dense(32, kernel_regularizer=regularizers.l2(2e-5),
                activity_regularizer=regularizers.l1(2e-5)))
    model.add(layer.PReLU())
    model.add(layer.Dropout(0.2))
    model.add(layer.BatchNormalization())

    
    model.add(layer.Dense(1))
    model.add(layer.Activation('sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='nadam')
    
    
    
    return model  
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4, mode='auto')
model = get_model()
model.fit(X_train, y=y_train, 
                     validation_data = (X_val, y_val),
                     batch_size=4096, epochs=10000,
                     verbose=1, shuffle=True, callbacks=[early_stopping])
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
y_pred = model.predict(X[len(df_train):])
df_test = pd.read_csv('../input/test.csv')
df_test['deal_probability'] = y_pred.T[0]
df_test = df_test[['item_id','deal_probability']]
df_test.to_csv('word2score.csv', index=None)