import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Dense, Dropout, Concatenate, Lambda, Flatten
from keras.layers import GlobalMaxPool1D
from keras.models import Model


import tqdm

MAX_SEQUENCE_LENGTH = 60
MAX_WORDS = 45000
EMBEDDINGS_TRAINED_DIMENSIONS = 100
EMBEDDINGS_LOADED_DIMENSIONS = 300
def load_embeddings(file):
    embeddings = {}
    with open(file) as f:
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        embeddings = dict(get_coefs(*line.split(" ")) for line in f)
        
    print('Found %s word vectors.' % len(embeddings))
    return embeddings
pretrained_embeddings = load_embeddings("../input/embeddings/glove.840B.300d/glove.840B.300d.txt")
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
    sns.despine()
    plt.show()

def display_model_epoch_metrics(epoch_callback):
    data = pd.DataFrame(data = {
        'F1': epoch_callback.f1s,
        'Precision': epoch_callback.precisions,
        'Recall': epoch_callback.recalls})
    sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)
    sns.despine()
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
    pretrained = Dropout(0.1)(pretrained)
    conv_0 = Conv2D(num_filters, kernel_size=(filter_size, EMBEDDINGS_LOADED_DIMENSIONS),
                    kernel_initializer='he_normal', activation='tanh')(pretrained)
    maxpool_0 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_size + 1, 1))(conv_0)

    d0 = Dropout(0.15)(maxpool_0)
    d0 = Dense(10)(d0)

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

filter_sizes = [1, 2, 3, 5]
num_filters = 45

test_predictions = []
kaggle_predictions = []

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.025)

for f in filter_sizes:
    print("CNN MODEL WITH FILTER OF SIZE {0}".format(f))
    epoch_callback = EpochMetricsCallback()
    model = make_model(f, num_filters)
    
    x, val_x, y, val_y = train_test_split(train_X, train_Y, test_size=0.01)
    history = model.fit(
        x=x, y=y, validation_data=(val_x, val_y),
        batch_size=512, epochs=23, callbacks=[epoch_callback], verbose=2)
    display_model_history(history)
    display_model_epoch_metrics(epoch_callback)
    
    kaggle_predictions.append(model.predict([test_word_tokens], batch_size=1024, verbose=2))
    test_predictions.append(model.predict([test_X]))
    

avg = np.average(kaggle_predictions, axis=0)

df_out = pd.DataFrame({"qid":df_test["qid"].values})
df_out['prediction'] = (avg > THRESHOLD).astype(int) 
df_out.to_csv("submission.csv", index=False)
# Adjust the threshold

avg = np.average(test_predictions, axis=0)
f1s = []
precisions = []
recalls = []

Ts = [x * 0.01 for x in range(0, 50)]
for t in Ts:
    pred = (avg > t).astype(int)
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
pred = (avg > thresh).astype(int)
f1 = metrics.f1_score(test_Y, pred)
print("Test F1 {0:.4f} at threshold {1:.3f}".format(f1, thresh))

thresh = THRESHOLD
pred = (avg > thresh).astype(int)
f1 = metrics.f1_score(test_Y, pred)
print("Test F1 {0:.4f} at threshold {1:.3f}".format(f1, thresh))


# avg = np.average(kaggle_predictions, axis=0)
# df_out = pd.DataFrame({"qid":df_test["qid"].values})
# df_out['prediction'] = (avg > thresh).astype(int)
# df_out.to_csv("submission.csv", index=False)

