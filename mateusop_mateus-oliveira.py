import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Model

from keras.layers import LSTM, Activation, Dense, SpatialDropout1D, Input, Embedding

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from sklearn.metrics import roc_auc_score

import os

print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')

df_valid = pd.read_csv('../input/valid.csv')
submission = pd.DataFrame()

submission['ID'] = df_valid['ID']
df = pd.concat([df_train,df_valid],sort=True)



print(len(df_valid))

print(len(df_train))

print(len(df))

df.sample(5)
df.isnull().sum()
df = df[['headline','is_sarcastic']]

df.head()
tokenizer = Tokenizer(num_words=2000, split=' ')

tokenizer.fit_on_texts(df['headline'].values)

X = tokenizer.texts_to_sequences(df['headline'].values)

X = pad_sequences(X)
df_train = df[df['is_sarcastic'] >= 0]



y = pd.get_dummies(df_train['is_sarcastic']).values



X_train = X[:18696]

X_sub = X[18696:]



x_train, x_test, y_train, y_test = train_test_split(X_train, y, random_state=42, test_size=0.2)
model = Sequential([

    Embedding(2000, 200, input_length = X.shape[1]),

    SpatialDropout1D(0.2),

    LSTM(200, dropout=0.2, recurrent_dropout=0.2),

    Dense(2,activation='softmax')

])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 25, verbose = 2)
valid_pred = model.predict(x_test)



print("AUC: %.2f" % roc_auc_score(y_test, valid_pred))
model.fit(X_train, y, epochs = 200, verbose = 2)
pred = model.predict(X_sub,batch_size=1,verbose = 0)



submission['is_sarcastic'] = np.argmax(pred, axis= 1)



submission.head()
submission.to_csv('submission.csv', index = False)