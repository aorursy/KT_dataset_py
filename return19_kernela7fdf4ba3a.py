# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from tensorflow.keras.layers import Reshape, Dropout, Input, Embedding, Dense, Flatten, Conv2D, MaxPool2D, Concatenate

from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.optimizers import Adam



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

train_input_path = "../input/train.csv"

test_input_path = "../input/test.csv"

print(train_input_path)



print(os.listdir("../input/embeddings"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv(train_input_path)



test = pd.read_csv(test_input_path)

test
max_features = 95000

maxlen = 70

X = train.question_text

y = train.target

test_X = test.question_text

# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

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
print( "Length of train is {}", len(train_X))

print( "Length of validation is {}", len(val_X))

print("Ratio of train is {}, val is {}", len(train_X)/len(X), len(val_X)/len(X) )

print("Length of test: ", len(test_X))
EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

def get_coefs(word,*arr): 

    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]

print("Embeddings size: ", embed_size)
word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)

x = Reshape((maxlen, embed_size, 1))(x)



cnn_1 = Conv2D(42, kernel_size=(7, embed_size), kernel_initializer='he_normal', activation='relu')(x)

pool_1 = MaxPool2D(pool_size=(7,1))(cnn_1)



cnn_2 = Conv2D(42, kernel_size=(5, embed_size), kernel_initializer='he_normal', activation='relu')(x)

pool_2 = MaxPool2D(pool_size=(5,1))(cnn_2)



cnn_3 = Conv2D(42, kernel_size=(3, embed_size), kernel_initializer='he_normal', activation='relu')(x)

pool_3 = MaxPool2D(pool_size=(3,1))(cnn_3)



cnn_4 = Conv2D(42, kernel_size=(1, embed_size), kernel_initializer='he_normal', activation='relu')(x)

pool_4 = MaxPool2D(pool_size=(1,1))(cnn_4)



x = Concatenate(axis=1)([pool_1,pool_2,pool_3,pool_4])



x = Dense(64, activation='relu')(x)

x = Flatten()(x)

x = Dropout(0.1)(x)

outp = Dense(1, activation='softmax')(x)



model_1 = Model(inputs=inp, outputs=outp)



model_1.summary()
model_1.compile(optimizer=Adam(lr=5e-3), loss='binary_crossentropy', metrics=['accuracy'])
model_1.fit(train_X, train_y, batch_size=1024, epochs=3, validation_data=(val_X, val_y))
# import numpy as np

# from sklearn import metrics



# pred_val_y = model_1.predict([val_X], batch_size=1024, verbose=1)



# def get_best_threshold():

#     best_tresh = 0.33 # default

#     best_f1_score = 0

#     for thresh in np.arange(0.05, 0.501, 0.01):

#         f1_score = metrics.f1_score(val_y, (pred_val_y >= thresh))

#         print('F1 score at ',thresh,' is ', f1_score)

#         if best_f1_score < f1_score :

#             best_f1_score = f1_score

#             best_tresh = thresh

#     return best_tresh

    

# best_threshold = get_best_threshold()
pred_test_y = model_1.predict([test_X], batch_size=1024, verbose=1)
best_threshold = 0.33

pred_test_y = (pred_test_y > best_threshold).astype(int)

out_df = pd.DataFrame({"qid":test["qid"].values})

out_df['prediction'] = pred_test_y

out_df.to_csv("submission.csv", index=False)
print(os.listdir("."))
pred_test_y