import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
plt.style.use("ggplot")

from tensorflow.keras.layers import *
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow.keras.metrics as metrics

from nltk.tokenize import WordPunctTokenizer
from string import punctuation, ascii_lowercase
import regex as re
from tqdm import tqdm
path = '../input/hsemath2020reviews/'

TRAIN_DATA_FILE = path + 'train.csv'
TEST_DATA_FILE = path + 'test.csv'
train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)
embeddings_dict = {}
with open("../input/glovetwitter27b100dtxt/glove.twitter.27B.200d.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float")
        embeddings_dict[word] = vector
lens = sorted([len(x) for x in train_df['positive']])
lens_test = sorted([len(x) for x in test_df['positive']])
plt.hist(lens)
plt.hist(lens_test)
plt.show()
REVIEW_LEN = 250
WORD_VEC_LEN = 200
BATCH = 32
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

docs = list(train_df['negative'].values) +  list(train_df['positive'].values) + list(test_df['negative'].values) +  list(test_df['positive'].values)
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1

embedding_matrix = np.zeros((vocab_size, WORD_VEC_LEN))
for word, i in t.word_index.items():
    embedding_vector = embeddings_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=BATCH, dim=(REVIEW_LEN,WORD_VEC_LEN), with_y=True, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.with_y = with_y
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X_positive = []
        X_negative = []

        y = np.empty((self.batch_size), dtype=float)
        
        for i, r in enumerate(indexes):
            row = self.df.iloc[r]
            X_positive.append(row['positive'])
            X_negative.append(row['negative'])

            if self.with_y:
                y[i] = float(row['score']) / 10.0
                
        X_positive1 = pad_sequences(t.texts_to_sequences(X_positive), maxlen=REVIEW_LEN)
        X_negative1 = pad_sequences(t.texts_to_sequences(X_negative), maxlen=REVIEW_LEN)

        if self.with_y:
            return [X_positive1, X_negative1], y
        else:
            return [X_positive1, X_negative1]
from sklearn.model_selection import train_test_split
train, validation = train_test_split(train_df, test_size=0.01)

train_generator = DataGenerator(train)
validation_generator = DataGenerator(validation)
positive = Input(shape=(REVIEW_LEN,), name='positive') 
negative = Input(shape=(REVIEW_LEN,), name='negative')

embedding = Embedding(vocab_size, WORD_VEC_LEN, weights=[embedding_matrix], trainable=False)

positives = SpatialDropout1D(0.2)(embedding(positive))
negatives = SpatialDropout1D(0.2)(embedding(negative))
lstm = Bidirectional(CuDNNLSTM(200, return_sequences=True))
positives = lstm((positives))
negatives = lstm((negatives))

x = Concatenate()([positives, negatives])

x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(30, activation='relu')(x)
# x = BatchNormalization()(x)
# x = Dropout(0.25)(x)
# x = Dense(10, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(
    inputs=[positive, negative],
    outputs=[output]
)
model.compile(
    optimizer=Adam(),
    loss=tf.keras.losses.MAE,
    metrics=[tf.keras.losses.MSE]
)
model.summary()
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    workers=16
)
history = model.fit(
    train_generator,
    epochs=1,
    validation_data=validation_generator,
    workers=16
)
model.save('tw200+double200+2(do+bn+dence)+sigmoid-3.h5')
history = model.fit(
    train_generator,
    epochs=13,
    validation_data=validation_generator,
    workers=16,
    initial_epoch=12
)
# model = tf.keras.models.load_model('tw200+double200+2(do+bn+dence)+sigmoid-2.h5')
preds = model.predict(DataGenerator(test_df, with_y=False, shuffle=False))
test_pred_dict = []
test_pred_dict.extend([{
    'review_id': review_id,
    'score': score[0][0] * 10} for review_id, score in zip(test_df['review_id'], preds)])
submission_df = pd.DataFrame(test_pred_dict)
submission_df = pd.DataFrame(test_pred_dict)
submission_df.to_csv('submission-9.csv', index=None)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
  ('vect', CountVectorizer()),
  ('tfidf', TfidfTransformer()),
  ('regressor', Ridge()),
])
text_clf.fit(X_train, y_train)
text_clf.score(X_validation, y_validation)
