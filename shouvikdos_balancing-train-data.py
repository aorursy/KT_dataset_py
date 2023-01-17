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
import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras import backend as K
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)
print ("train data class 0 count is %d, and class 1 count is %d" %(list(train_df["target"]).count(0), list(train_df["target"]).count(1)))
frac_0 = np.float(list(train_df["target"]).count(0))/(list(train_df["target"]).count(0) + list(train_df["target"]).count(1))
frac_1 = np.float(list(train_df["target"]).count(1))/(list(train_df["target"]).count(0) + list(train_df["target"]).count(1))
print ("train data class 0 count fraction is %f, and class 1 count fraction is %f" %(frac_0, frac_1))
train_indices = np.where(train_df["target"] == 1)[0]
print ("\n".join(list(train_df.iloc[train_indices].question_text)[:10]))
positive_indices = list(np.where(train_df["target"] == 1)[0])
negative_indices = list(np.where(train_df["target"] == 0)[0])
validation_fraction = 0.2
val_pos_index = list(np.random.choice(positive_indices, int(len(positive_indices) * validation_fraction), replace = False))
train_pos_index = list(set(positive_indices) - set(val_pos_index))
val_neg_index = list(np.random.choice(negative_indices, int(len(negative_indices) * validation_fraction), replace = False))
train_neg_index = list(set(negative_indices) - set(val_neg_index))
total_negative = train_neg_index + val_neg_index
total_positive = train_pos_index + val_pos_index

if len(total_negative) != len(negative_indices):
    raise Exception("class 0 length mismatch, please check..")
if len(total_positive) != len(positive_indices):
    raise Exception("class 1 length mismatch, please check..")
train_pos_index_resample = train_pos_index
while len(train_pos_index_resample) < int(0.8 * len(train_neg_index)):
    sample_indices = list(np.random.choice(train_pos_index, 10000, replace = False))
    train_pos_index_resample += sample_indices
train_indices = train_pos_index + train_neg_index
np.random.shuffle(train_indices)
train = train_df.iloc[train_indices]

val_indices = val_pos_index + val_neg_index
np.random.shuffle(val_indices)
val = train_df.iloc[val_indices]
embed_size = 300 # how big is each word vector
max_features = 100000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

## fill up the missing values
train_X = train["question_text"].fillna("_na_").values
val_X = val["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = train['target'].values
val_y = val['target'].values

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])

print(model.summary())
model.fit(train_X, train_y, batch_size=512, epochs=3, validation_data=(val_X, val_y))
pred_val_y = model.predict([val_X], batch_size=1024, verbose=1)
thres_list = []; result = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))))
    thres_list.append(thresh); result.append(metrics.f1_score(val_y, (pred_val_y>thresh).astype(int)))

indices = np.argsort(result)[::-1]
print ("best threhold is : {0}".format(thres_list[indices[0]]))                                             
                            
pred_noemb_test_y = model.predict([test_X], batch_size=1024, verbose=1)
pred_test_y = (pred_noemb_test_y>0.33).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)
