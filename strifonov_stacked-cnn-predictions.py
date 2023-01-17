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
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Dense, Dropout, Concatenate, Lambda, Flatten
from keras.layers import GlobalMaxPool1D
from keras.models import Model


import tqdm

MAX_SEQUENCE_LENGTH = 70
MAX_WORDS = 95000
EMBEDDINGS_TRAINED_DIMENSIONS = 100
EMBEDDINGS_LOADED_DIMENSIONS = 300
def load_embeddings(file):
    embeddings = {}
    with open(file) as f:
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        embeddings = dict(get_coefs(*line.split(" ")) for line in f)
        
    print('Found %s word vectors.' % len(embeddings))
    return embeddings
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
BATCH_SIZE = 512
Q_FRACTION = 1
questions = df_train.sample(frac=Q_FRACTION)
question_texts = questions["question_text"].values
question_targets = questions["target"].values
test_texts = df_test["question_text"].fillna("_na_").values

print(f"Working on {len(questions)} questions")
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(list(df_train["question_text"].values))
# custom_embeddings = train_w2v(question_texts, epochs=5)
pretrained_embeddings = load_embeddings("../input/embeddings/glove.840B.300d/glove.840B.300d.txt")

from collections import defaultdict

def create_embedding_weights(tokenizer, embeddings, dimensions):
    not_embedded = defaultdict(int)
    
    word_index = tokenizer.word_index
    words_count = min(len(word_index), MAX_WORDS)
    embeddings_matrix = np.zeros((words_count, dimensions))
    for word, i in word_index.items():
        if i >= MAX_WORDS:
            continue
        if word not in embeddings:
            not_embedded[word] = not_embedded[word] + 1
            continue
        embedding_vector = embeddings[word]
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
            
    print(sorted(not_embedded, key=not_embedded.get)[:10])
    return embeddings_matrix
# custom_emb_weights = create_embedding_weights(tokenizer, custom_embeddings, EMBEDDINGS_TRAINED_DIMENSIONS)
pretrained_emb_weights = create_embedding_weights(tokenizer, pretrained_embeddings, EMBEDDINGS_LOADED_DIMENSIONS)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

THRESHOLD = 0.35

class EpochMetricsCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.f1s = []
        self.precisions = []
        self.recalls = []
        
    def on_epoch_end(self, epoch, logs={}):
        predictions = self.model.predict(self.validation_data[0])
        predictions = (predictions > THRESHOLD).astype(int)
        predictions = np.asarray(predictions)
        targets = self.validation_data[1]
        f1 = metrics.f1_score(targets, predictions)
        precision = metrics.precision_score(targets, predictions)
        recall = metrics.recall_score(targets, predictions)

        print(" - F1 score: {0:.4f}, Precision: {1:.4f}, Recall: {2:.4f}"
              .format(f1, precision, recall))
        self.f1s.append(f1)
        self.precisions.append(precision)
        self.recalls.append(recall)
        return
    
def display_model_history(history):
    data = pd.DataFrame(data={'Train': history.history['loss'], 'Test': history.history['val_loss']})
    ax = sns.lineplot(data=data, palette="pastel", linewidth=2.5, dashes=False)
    ax.set(xlabel='Epoch', ylabel='Loss', title='Loss')
    plt.show()

def display_model_epoch_metrics(epoch_callback):
    fig, axes = plt.subplots(1, 3, figsize = (15, 5), sharey=False)
    a1, a2, a3 = axes
    
    a1.set_title('F1')
    a1.set(xlabel='Epoch', title='F1')
    sns.lineplot(data=pd.DataFrame(data={'F1': epoch_callback.f1s}),
                 palette="pastel", linewidth=2.5, dashes=False, ax=a1, legend=False)

    a2.set_title('Precision')
    a2.set(xlabel='Epoch', title='Precision')
    sns.lineplot(data=pd.DataFrame(data={'Precision': epoch_callback.precisions}),
                 palette="pastel", linewidth=2.5, dashes=False, ax=a2, legend=False)

    a3.set_title('Recall')
    a3.set(xlabel='Epoch', title='Recall')
    sns.lineplot(data=pd.DataFrame(data={'Recall': epoch_callback.recalls}),
                 palette="pastel", linewidth=2.5, dashes=False, ax=a3, legend=False)

    plt.show()
X = pad_sequences(tokenizer.texts_to_sequences(question_texts),
                        maxlen=MAX_SEQUENCE_LENGTH)
Y = question_targets

test_word_tokens = pad_sequences(tokenizer.texts_to_sequences(test_texts),
                       maxlen=MAX_SEQUENCE_LENGTH)
from keras.layers import Conv1D, Conv2D, Reshape, MaxPool1D, MaxPool2D, BatchNormalization

def make_model(filter_size, num_filters):
    tokenized_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name="tokenized_input")
    
    pretrained = Embedding(MAX_WORDS,
                           EMBEDDINGS_LOADED_DIMENSIONS,
                           weights=[pretrained_emb_weights],
                           trainable=False)(tokenized_input)

    pretrained = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDINGS_LOADED_DIMENSIONS, 1))(pretrained)
    conv_0 = Conv2D(num_filters, kernel_size=(filter_size, EMBEDDINGS_LOADED_DIMENSIONS), kernel_initializer='he_normal', activation='tanh')(pretrained)
    maxpool_0 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_size + 1, 1))(conv_0)

    d0 = Dense(4)(maxpool_0)

    x = Flatten()(d0)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[tokenized_input], outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model
import random
from sklearn.model_selection import train_test_split

# filter_sizes = range(1, 11)
filter_sizes = [1, 2, 3, 5]
num_filters = 45

train_predictions = []
test_predictions = []
kaggle_predictions = []

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.15)

for f in filter_sizes:
    print("CNN MODEL WITH FILTER OF SIZE {0}".format(f))
    epoch_callback = EpochMetricsCallback()
    model = make_model(f, num_filters)
    
    # use a lot of validation data on purpose so that the models would be trained on a noticeably less than the whole dataset
    x, val_x, y, val_y = train_test_split(train_X, train_Y, test_size=0.015)
    history = model.fit(
        x=x, y=y, validation_data=(val_x, val_y),
        batch_size=512, epochs=7, callbacks=[epoch_callback], verbose=2)
    display_model_history(history)
    display_model_epoch_metrics(epoch_callback)
    
    train_predictions.append(model.predict([train_X], batch_size=1024, verbose=2))
    test_predictions.append(model.predict([test_X], batch_size=1024, verbose=2))
    kaggle_predictions.append(model.predict([test_word_tokens], batch_size=1024, verbose=2))
    

def stack_models(predictions, targets):
    layer_size = len(predictions)
    inp = Input(shape=(layer_size,))
    d0 = Dropout(0.2)(inp)
    d0 = Dense(pow(layer_size, 2))(d0)
    d1 = Dropout(0.2)(d0)
    d1 = Dense(2 * layer_size)(d1)
    b = BatchNormalization()(d1)
    out = Dropout(0.2)(b)
    out = Dense(1, activation='sigmoid')(out)

    model = Model(inputs=inp, outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    epoch_callback = EpochMetricsCallback()
    # TODO find a more idiomatic way to transform...
    x = np.array(list(zip(*np.squeeze(predictions))))
    y = targets
    print(np.shape(x))
    print(np.shape(y))
    
    history = model.fit(x=x, y=y, epochs=5, callbacks=[epoch_callback], validation_split=0.02, verbose=2)
    display_model_history(history)
    display_model_epoch_metrics(epoch_callback)
    
    return model
    
model = stack_models(test_predictions, test_Y)

stacked_kaggle_predictions = np.array(list(zip(*np.squeeze(kaggle_predictions))))
stacked_kaggle_predictions = model.predict(stacked_kaggle_predictions, batch_size=1024)

stacked_test_predictions = np.array(list(zip(*np.squeeze(test_predictions))))
stacked_test_predictions = model.predict(stacked_test_predictions, batch_size=1024)
# df_out = pd.DataFrame({"qid":df_test["qid"].values})
# df_out['prediction'] = (kaggle_predictions > THRESHOLD).astype(int) 
# df_out.to_csv("submission.csv", index=False)
# Adjust the threshold
print(np.shape(stacked_test_predictions))

f1s = []
precisions = []
recalls = []

Ts = [x * 0.01 for x in range(0, 50)]
for t in Ts:
    pred = (stacked_test_predictions > t).astype(int)
    f1s.append(metrics.f1_score(test_Y, pred))
    precisions.append(metrics.precision_score(test_Y, pred))
    recalls.append(metrics.recall_score(test_Y, pred))


plt.plot(Ts, f1s)
plt.plot(Ts, precisions)
plt.plot(Ts, recalls)
plt.title('Threshold levels')
plt.ylabel('Value')
plt.xlabel('Threshold')
plt.legend(['F1', 'Precision', 'Recall'])
plt.show()

thresh = Ts[np.argmax(f1s)]
pred = (stacked_test_predictions > thresh).astype(int)
f1 = metrics.f1_score(test_Y, pred)
print("Test F1 {0:.4f} at threshold {1:.3f}".format(f1, thresh))
df_out = pd.DataFrame({"qid":df_test["qid"].values})
df_out['prediction'] = (stacked_kaggle_predictions > thresh).astype(int)
df_out.to_csv("submission.csv", index=False)

