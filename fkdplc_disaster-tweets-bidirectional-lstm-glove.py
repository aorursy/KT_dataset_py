import numpy as np

import pandas as pd

import os



CSV_TRAIN = "/kaggle/input/nlp-getting-started/train.csv"

CSV_TEST = "/kaggle/input/nlp-getting-started/test.csv"



train = pd.read_csv(CSV_TRAIN)

test = pd.read_csv(CSV_TEST)

train.head()
from matplotlib import pyplot as plt

import seaborn as sns



def upper_rate(t):

    return len([i for i in t if i.isupper()]) / len(t)



def digits_rate(t):

    return len([i for i in t if i.isdigit()]) / len(t)



def how_many(t, c):

    return len(t) - len(t.replace(c, ''))



def word_len(t):

    for sign in ' .:!?,/|':

        t = t.replace(sign, ' ')

    prepared = t.replace('  ', ' ')

    cnt = len(prepared.split(' '))

    total = len(prepared.replace(' ', ''))

    return cnt / total



def extract_feature(name, func, col_from='text'):

    train[name] = train[col_from].apply(func)

    test[name] = test[col_from].apply(func)



extract_feature('text_len', lambda t: len(t))

extract_feature('urls', lambda t: how_many(t, 'http'))

extract_feature('upper_rate', upper_rate)

extract_feature('digits', digits_rate)

extract_feature('has_location', lambda l: 1 if l == l else 0, col_from='location')

extract_feature('has_keyword', lambda l: 1 if l == l else 0, col_from='keyword')

signs = '.,-|/:;+=#[](){}<>?!@$%^&*_"~' + "'"

for sign in signs:

    extract_feature('sign_' + sign, lambda t: how_many(t, sign))

    

def clean_text(t):

    for s in sign:

        t = t.replace(s, ' ')

    for i in '1234567890':

        t = t.replace(i, '')

    t = t.replace('https', '')

    t = t.replace('http', '')

    t = t.replace('  ', ' ')

    t = t.lower()

    return t



extract_feature('text', clean_text)



def plot_feature(feature):

    plt.xlabel(feature)

    plt.hist(train[train['target'] == 1][feature], alpha=0.4, label='real', color='r')

    plt.hist(train[train['target'] == 0][feature], alpha=0.4, label='fake', color='g')

    plt.legend()



EX_FEAT = ['text_len', 'urls', 'upper_rate', 'digits', 'has_location', 'has_keyword'] + ['sign_' + i for i in signs]

W, H = 6, 6

corr = train[EX_FEAT + ['target']].corr()



plt.figure(1, figsize=(16, 16))



for i, feat in enumerate(EX_FEAT):

    plt.subplot(H, W, i+1)

    plot_feature(feat)



plt.show()
plt.figure(1, figsize=(16, 16))

sns.heatmap(corr, annot=True, fmt='.2f', cbar=False)

plt.show()
disaster = train['target'].sum() / train.shape[0]

print('imbalance:\nreal vs fake\n%.2f\t%.2f' % (disaster, 1 - disaster))
from sklearn.decomposition import PCA



# extract targets and ids of test set

target = train['target'].to_numpy()

test_ids = test['id'].to_numpy()



tweet_max_length = max([len(i) for i in train['text']])

print('Max length =', tweet_max_length)



train_ex = train[EX_FEAT]

test_ex = test[EX_FEAT]



pca = PCA(0.999)

pca.fit(train_ex)



train_ex = pca.transform(train_ex)

test_ex = pca.transform(test_ex)



#remove unused columns

train.drop(columns=['id', 'keyword', 'location', 'target'] + EX_FEAT, inplace=True)

test.drop(columns=['id', 'keyword', 'location'] + EX_FEAT, inplace=True)



train.shape, train_ex.shape, test.shape, test_ex.shape, target.shape
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



MAX_WORDS = 20000

MAX_LEN = tweet_max_length



# create words vector

tk = Tokenizer(num_words=MAX_WORDS)

corpus = [text for text in train['text']]

tk.fit_on_texts(corpus)



# convert texts into vectors 

train_sequences = tk.texts_to_sequences(train['text'])

test_sequences = tk.texts_to_sequences(test['text'])

word_indexes = tk.word_index



# convert tweets into vectors

x_train = pad_sequences(train_sequences, maxlen=MAX_LEN)

x_test = pad_sequences(test_sequences, maxlen=MAX_LEN)



# extend with extras

x_train = np.concatenate([x_train, train_ex], axis=1)

x_test = np.concatenate([x_test, test_ex], axis=1)



x_train.shape, x_test.shape
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split



# split train into train and validation data

x_train, x_train_val, target, target_val = train_test_split(x_train, target, test_size=0.2, random_state=42)



# x_train, target = rus.fit_resample(x_train, target)

x_train.shape, target.shape, x_train_val.shape, target_val.shape
from tqdm.notebook import tqdm



# GloVe dimension

EMBEDDING_DIM = 100



embeddings = {}

f = open('/kaggle/input/glove6b100dtxt/glove.6B.100d.txt', 'r')

for line in tqdm(f, total=400000):

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings[word] = coefs

f.close()



# weights matrix for embedded layer

embedding_matrix = np.zeros((MAX_WORDS, EMBEDDING_DIM))

for word, i in word_indexes.items():

    if i < MAX_WORDS:

        emb_vec = embeddings.get(word)

        if emb_vec is not None:

            embedding_matrix[i] = emb_vec

embedding_matrix
from keras.layers import Dense, Dropout, Embedding, LSTM, Flatten, Bidirectional, concatenate, LeakyReLU

from keras.layers.convolutional import Conv1D, MaxPooling1D

from keras.models import Model

from keras import Input

from keras.utils import plot_model

from keras.regularizers import l2

from keras import backend as K



def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def f1_score(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def create_model():

    

    # glove branch

    emb_input = Input(shape=(MAX_LEN, ), name='glove')

    emb_x = Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN, weights=[embedding_matrix], trainable=False)(emb_input)

    emb_x = Bidirectional(LSTM(32, activation='elu', dropout=0.2, recurrent_dropout=0.2))(emb_x)

    

    # extra features branch

    ex_input = Input(shape=(train_ex.shape[1], ), name='extra')

    ex_x = Dense(8)(ex_input)

    ex_x = LeakyReLU()(ex_x)

    ex_x = Dropout(0.5)(ex_x)

    

    out = Dense(1, activation='sigmoid')(concatenate([emb_x, ex_x]))

    

    model = Model(inputs=[emb_input, ex_input], outputs=out)



    model.compile(

        loss='binary_crossentropy',

        optimizer='adam',

        metrics=[f1_score]

    )

    return model



m = create_model()

m.summary()

plot_model(m, show_shapes=True, to_file='model.png')
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler().fit(train_ex)



def create_x_dict(x_train):

    return {

        'glove': x_train[:, :-train_ex.shape[1]],

        'extra': scaler.transform(x_train[:, -train_ex.shape[1]:])

    }

    

x_train_local, target_local = create_x_dict(x_train), target

x_val_local, target_val_local = create_x_dict(x_train_val), target_val



es = EarlyStopping(patience=17, restore_best_weights=True, verbose=1)

rlr = ReduceLROnPlateau(patience=5, verbose=1)



model = create_model()

history = model.fit(

    x_train_local,

    target_local,

    epochs=100,

    batch_size=32,

    verbose=1,

    callbacks=[es, rlr],

    validation_data=(x_val_local, target_val_local)

)



plt.figure(1, figsize=(12, 6))



plt.subplot(121)

plt.plot(history.history['val_loss'], label='val loss')

plt.plot(history.history['loss'], label='train loss')

plt.legend()



plt.subplot(122)

plt.plot(history.history['val_f1_score'], label='val f1')

plt.plot(history.history['f1_score'], label='train f1')

plt.legend()



plt.show()
from sklearn.metrics import f1_score, precision_score, recall_score



y_true = target_val

y_pred = model.predict(create_x_dict(x_train_val))



steps = 1000

y_min = int(y_pred.min() * steps)

y_max = int(y_pred.max() * steps)

diff = y_max - y_min

thresholds = [i / steps for i in range(y_min, y_max)]

precisions = np.zeros((diff, ))

recalls = np.zeros((diff, ))

f1s = np.zeros((diff, ))

for i, thres in enumerate(thresholds):

    y = y_pred.copy()

    y[y > thres] = 1

    y[y != 1] = 0

    prec = precision_score(y_true, y)

    rec = recall_score(y_true, y)

    f1 = 2 * prec * rec / (prec + rec)

    precisions[i] = prec

    recalls[i] = rec

    f1s[i] = f1



index_of_best = np.argsort(f1s, axis=None)[-1]

best_threshold = thresholds[index_of_best]

best_f1 = f1s[index_of_best]



plt.figure(1, figsize=(8, 8))

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.xlabel('threshold')

plt.plot(thresholds, recalls, label='recall')

plt.plot(thresholds, precisions, label='precision')

plt.plot(thresholds, f1s, label='f1 (best: %.3f)' % best_f1)

plt.plot([best_threshold] * 2, [0, 1], '--', label='best threshold (%.3f)' % best_threshold)

plt.legend()

plt.show()
predict = model.predict(create_x_dict(x_test)).flatten()

predict[predict > best_threshold] = 1

predict[predict != 1] = 0

predict = predict.round().astype(int)

disaster = predict.sum() / predict.shape[0]

disaster
submission = pd.DataFrame()

submission['id'] = test_ids

submission['target'] = predict

submission.to_csv('submission.csv', index=False)

submission.head(12)