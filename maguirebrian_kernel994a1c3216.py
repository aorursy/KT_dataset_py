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
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
df = pd.read_csv('../input/train.csv')
df.describe()
df['question_length'] = df['question_text'].str.split().apply(len)
df.describe()
print(df['question_text'][0])
print(df['question_length'][0])
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2)
print(os.listdir("../input/embeddings/GoogleNews-vectors-negative300"))
from gensim.models import KeyedVectors
EMBEDDING_FILE = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
MAX_SEQUENCE_LENGTH=30
WORD_EMBEDDING_LENGTH=300
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout, Conv2D, Flatten, MaxPool2D, Reshape, BatchNormalization, CuDNNLSTM, Bidirectional, MaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import relu, sigmoid
from tensorflow import keras as keras
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['question_text'].values)
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index)+1, WORD_EMBEDDING_LENGTH))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
#del word2vec

embedding_layer = Embedding(len(word_index) + 1,
                            WORD_EMBEDDING_LENGTH,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False, 
                            mask_zero=True,
                           name="embedding")
# https://github.com/aravindsiv/dan_qa/blob/master/custom_layers.py

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf

class AverageWords(Layer):
    def __init__(self):
        super(AverageWords,self).__init__()
        self.supports_masking = True

    def call(self, x, mask=None):
        axis = K.ndim(x) - 2
        if mask is not None:
            summed = K.sum(x, axis=axis)
            n_words = K.expand_dims(K.sum(K.cast(mask, 'float32'), axis=axis), axis)
            return summed / n_words
        else:
            return K.mean(x, axis=axis)

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        dimensions = list(input_shape)
        n_dimensions = len(input_shape)
        del dimensions[n_dimensions - 2]
        return tuple(dimensions)

class WordDropout(Layer):
    def __init__(self, rate):
        super(WordDropout,self).__init__()
        self.rate = min(1., max(0., rate))
        self.supports_masking = True

    def call(self, inputs, training=None, input_mask=None):
        if 0. < self.rate < 1.0:
            def dropped_inputs():
                input_shape = K.shape(inputs)
                batch_size = input_shape[0]
                n_time_steps = input_shape[1]
                mask = tf.random_uniform((batch_size, n_time_steps, 1)) >= self.rate
                w_drop = K.cast(mask, 'float32') * inputs
                if input_mask is not None:
                    w_drop = input_mask*inputs
                return w_drop
            return K.in_train_phase(dropped_inputs, inputs, training=training)
        return inputs

    def get_config(self):
        config = {'rate': self.rate}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

class ConsumeMask(Layer):

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x
# https://www.aclweb.org/anthology/D14-1181
drop_out = 0.5
word_drop = 0.3
filters = [3,4,5]

input1 = keras.Input(shape=(MAX_SEQUENCE_LENGTH,), name="input")
embedding = embedding_layer(input1)
word_drop = WordDropout(word_drop)(embedding)
remove_mask = ConsumeMask()(word_drop)
reshape = Reshape((MAX_SEQUENCE_LENGTH, WORD_EMBEDDING_LENGTH, 1), name='reshape')(remove_mask)
cnn_layers = []
for f in filters:
    conv = Conv2D(
        100,
        (f, WORD_EMBEDDING_LENGTH),
        name="%s_conv" % f,
        input_shape=(None, MAX_SEQUENCE_LENGTH, WORD_EMBEDDING_LENGTH, 1))(reshape)
    
    cnn_layers.append(MaxPool2D((MAX_SEQUENCE_LENGTH + 1 - f, 1), name='%s_max' % f)(conv))
concat1 = keras.layers.concatenate(cnn_layers, axis=1, name="concat1")
flatten = Flatten(name="flatten")(concat1)
avg = AverageWords()(word_drop)
concat2 = keras.layers.concatenate([flatten, avg], axis=1, name="concat2")
#drop1 = Dropout(drop_out, name="drop1")(concat2)
dense1 = Dense(512, activation=relu, name="dense1")(concat2)
drop2 = Dropout(drop_out, name="drop2")(dense1)
output = Dense(1, activation='sigmoid', name="output")(drop2)

model = keras.Model(inputs=input1, outputs=output)
model.summary()
from tensorflow.keras import backend as K

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


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc', precision, recall, f1])
def process_text(array, tokenizer):
    seq = tokenizer.texts_to_sequences(array)
    return pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
import math
def batch_gen(train_df, batch_size=128):
    while True:
        n_batches = math.ceil(len(train_df) / batch_size)
        for i in range(n_batches):
            x = np.array(
                process_text(
                    train_df['question_text'][i*batch_size:(i+1)*batch_size].values,
                    tokenizer)
            )
            yield x, np.array(train_df["target"][i*batch_size:(i+1)*batch_size].values)
import math
def batch_gen_lang_model(train_df, batch_size=128):
    while True:
        n_batches = math.ceil(len(train_df) / batch_size)
        for i in range(n_batches):
            x = np.array(
                process_text(
                    train_df['question_text'][i*batch_size:(i+1)*batch_size].values,
                    tokenizer)
            )
            y = []
            for sample in train_df['question_text'][i*batch_size:(i+1)*batch_size].values:
                y_sentence = []
                for word in sample.split():
                    y_sentence.append(word2vec.word_vec(word))
                y_sentence+= [0*WORD_EMBEDDING_LENGTH]*(MAX_SEQUENCE_LENGTH - len(sample))
                y.append(y_sentence)
            y = np.array(y)
            yield x, y
batch_size = 512

x_v, y_v = np.array(process_text(test_df['question_text'].values,tokenizer)), np.array(test_df["target"].values)

loss = model.fit_generator(
    batch_gen(train_df, batch_size),
    epochs=3,
    steps_per_epoch=2000,
    validation_data=(x_v, y_v),#batch_gen(train_df, batch_size),
    #validation_steps=2000,
    shuffle=False,
    use_multiprocessing=False
).history
import matplotlib.pyplot as plt

metric = 'f1'

plt.plot(
    range(len(loss[metric])),loss[metric],
    range(len(loss['val_' + metric])), loss['val_' + metric]
)
plt.show()

pred = model.predict(
    x=process_text(np.array(train_df['question_text'].head(100).values), tokenizer)
)
pred
result = model.evaluate(
    x=process_text(np.array(test_df['question_text'].values), tokenizer),
    y=np.array(test_df['target'].values),
    batch_size=128)
result
df_test = pd.read_csv('../input/test.csv')
df_test.head(2)
test_x = process_text(np.array(df_test['question_text'].values), tokenizer)
predictions_list = model.predict(test_x)
df_sub = df_test.drop('question_text', axis=1)
df_sub.head(2)
df_sub['prediction'] = np.array(predictions_list)
df_sub.head()
df_sub.describe()
df_sub['prediction'] = df_sub['prediction'].round(0).astype(int)
df_sub.describe()
df_sub.loc[df_sub['prediction'] == 1].count()
df_sub.to_csv('submission.csv', index=False)