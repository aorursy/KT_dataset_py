# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, Dropout, CuDNNGRU
from keras.layers import Bidirectional, GlobalMaxPool1D, Concatenate
from keras.models import Model, Sequential

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

train_df, val_df  = train_test_split(df, test_size=0.1, random_state=2018)
test_df = pd.read_csv('../input/test.csv')

del df
import gc; gc.collect()
train_df.shape
train_df.columns
for i, row in train_df.iterrows():
    print (row)
    break
label_col = 'is_duplicate'
## some config values 
embed_size = 300 # how big should each word vector be
max_features = 50_000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use
## fill up the missing values
train_A = train_df["question1"].fillna("_na_").values
val_A = val_df["question1"].fillna("_na_").values
test_A = test_df["question1"].fillna("_na_").values

train_B = train_df["question2"].fillna("_na_").values
val_B = val_df["question2"].fillna("_na_").values
test_B = test_df["question2"].fillna("_na_").values


## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_A) + list(train_B))
train_A = tokenizer.texts_to_sequences(train_A)
val_A = tokenizer.texts_to_sequences(val_A)
test_A = tokenizer.texts_to_sequences(test_A)

train_B = tokenizer.texts_to_sequences(train_B)
val_B = tokenizer.texts_to_sequences(val_B)
test_B = tokenizer.texts_to_sequences(test_B)

## Pad the sentences 
train_A = pad_sequences(train_A, maxlen=maxlen)
val_A = pad_sequences(val_A, maxlen=maxlen)
test_A = pad_sequences(test_A, maxlen=maxlen)

train_B = pad_sequences(train_B, maxlen=maxlen)
val_B = pad_sequences(val_B, maxlen=maxlen)
test_B = pad_sequences(test_B, maxlen=maxlen)

## Get the target values
train_y = train_df[label_col].values
val_y = val_df[label_col].values

def create_gru_nn():
    seq = Sequential()
    # embedding layer
    seq.add(Embedding(max_features, embed_size, trainable=True))
    # encode via bidirectional GRU
    seq.add(Bidirectional(CuDNNGRU(64, return_sequences=True)))
    seq.add(GlobalMaxPool1D())
    # some dropout for regularization
    seq.add(Dropout(0.1))
    return seq

gru_nn = create_gru_nn()

input_a = Input(shape=(maxlen,))
input_b = Input(shape=(maxlen,))

processed_a = gru_nn(input_a)
processed_b = gru_nn(input_b)

merged = Concatenate()([processed_a, processed_b])
merged = Dense(64, activation='elu')(merged)
merged = Dropout(0.1)(merged)
out = Dense(1, activation='sigmoid')(merged)

model = Model(input=[input_a, input_b], output=out)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse', 'mae'])

print(model.summary())
model.fit(x=[train_A, train_B], y=train_y, validation_data=([val_A, val_B], val_y), batch_size=512, epochs=2)
val_y_pred = model.predict([val_A, val_B], batch_size=1024)[:, 0]
f_scores = []
for thresh in np.arange(0.1, 0.9, 0.01):
    thresh = np.round(thresh, 2)
    f_score = metrics.f1_score(val_y, (val_y_pred>thresh).astype(int))
    f_scores.append((f_score, thresh))
    
fmax, opt_thresh = max(f_scores)
print("F1 score at threshold {0} is {1}".format(opt_thresh, fmax))
pred_test_y = model.predict([test_A, test_B], batch_size=1024)[:, 0]
# pred_test_y = (pred_test_y>opt_thresh).astype(int)
out_df = pd.DataFrame({"test_id":test_df["test_id"].values})
out_df['is_duplicate'] = pred_test_y
out_df.to_csv("submission.csv", index=False)
