import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn import metrics


print(os.listdir("../input"))
MAX_SEQUENCE_LENGTH = 60
MAX_WORDS = 45000
EMBEDDINGS_LOADED_DIMENSIONS = 300
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
BATCH_SIZE = 512
Q_FRACTION = 1
questions = df_train.sample(frac=Q_FRACTION)
question_texts = questions["question_text"].values
question_targets = questions["target"].values
test_texts = df_test["question_text"].fillna("_na_").values

print(f"Working on {len(questions)} questions")
def load_embeddings(file):
    embeddings = {}
    with open(file) as f:
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        embeddings = dict(get_coefs(*line.split(" ")) for line in f)
        
    print('Found %s word vectors.' % len(embeddings))
    return embeddings

%time pretrained_embeddings = load_embeddings("../input/embeddings/glove.840B.300d/glove.840B.300d.txt")
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=MAX_WORDS)

%time tokenizer.fit_on_texts(list(df_train["question_text"].values))
from collections import defaultdict

def create_embedding_weights(tokenizer, embeddings, dimensions):
    not_embedded = defaultdict(int)
    
    word_index = tokenizer.word_index
    words_count = min(len(word_index), MAX_WORDS)
    embeddings_matrix = np.zeros((words_count, dimensions))
    for word, i in word_index.items():
        if i >= MAX_WORDS:
            continue
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
            
    return embeddings_matrix

pretrained_emb_weights = create_embedding_weights(tokenizer, pretrained_embeddings, EMBEDDINGS_LOADED_DIMENSIONS)
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
    
from keras.preprocessing.sequence import pad_sequences

%time X = pad_sequences(tokenizer.texts_to_sequences(question_texts), maxlen=MAX_SEQUENCE_LENGTH)
%time Y = question_targets

%time test_word_tokens = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=MAX_SEQUENCE_LENGTH)
from keras.layers import Input, Embedding, Dense, Dropout, SpatialDropout1D, Flatten
from keras.layers import LSTM, Bidirectional, GRU, BatchNormalization
from keras.models import Model

def make_model():
    tokenized_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name="tokenized_input")
    embedding = Embedding(MAX_WORDS, EMBEDDINGS_LOADED_DIMENSIONS,
                          weights=[pretrained_emb_weights],
                          trainable=False)(tokenized_input)
    
    d0 = SpatialDropout1D(0.1)(embedding)
    lstm = Bidirectional(LSTM(128, return_sequences=True))(d0)
    lstm = Bidirectional(LSTM(64, return_sequences=False))(lstm)
    d1 = Dropout(0.15)(lstm)
    d1 = Dense(64)(d1)
    b = BatchNormalization()(d1)
    out = Dense(1, activation='sigmoid')(b)
    
    model = Model(inputs=[tokenized_input], outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    return model
from sklearn.model_selection import train_test_split

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.01)

epoch_callback = EpochMetricsCallback()
model = make_model()
history = model.fit(x=train_X, y=train_Y, validation_split=0.015,
                    batch_size=BATCH_SIZE, epochs=4, verbose=2,
                    callbacks=[epoch_callback])
display_model_history(history)
display_model_epoch_metrics(epoch_callback)
test_word_tokens = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=MAX_SEQUENCE_LENGTH)
kaggle_predictions = (model.predict([test_word_tokens], batch_size=1024, verbose=2))

df_out = pd.DataFrame({"qid":df_test["qid"].values})
df_out['prediction'] = (kaggle_predictions > THRESHOLD).astype(int) 
df_out.to_csv("submission.csv", index=False)