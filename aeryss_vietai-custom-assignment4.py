!pip install vncorenlp
!mkdir -p vncorenlp/models/wordsegmenter
!wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
!wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
!wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
!mv VnCoreNLP-1.1.1.jar vncorenlp/ 
!mv vi-vocab vncorenlp/models/wordsegmenter/
!mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/
import os
import sys
import time
import datetime
import math

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf
import tensorflow_addons as tfa
import keras.backend as K

import warnings
from gensim.models import FastText
warnings.filterwarnings("ignore")
pd.options.display.max_colwidth=1000
train_df = pd.read_csv("../input/vietai-dataset/assignment4-data/Assignment4/train.csv")
train_df.head()
print('Number of train samples in total:', len(train_df))
print('Number of positives:', np.sum(train_df['class']==1))
print('Number of negatives:', np.sum(train_df['class']==0))
sample_positive = train_df[train_df['class'] == 1].sample(5)
sample_positive
sample_negative = train_df[train_df['class'] == 0].sample(5)
sample_negative
test_df = pd.read_csv("../input/vietai-dataset/assignment4-data/Assignment4/test.csv")
print('Number of test samples in total:', len(test_df))
test_df.head()
words_list = np.load('../input/vietai-dataset/assignment4-data/Assignment4/words_list.npy')
print('Prunned vocabulary loaded!')
words_list = words_list.tolist()
word_vectors = np.load('../input/vietai-dataset/assignment4-data/Assignment4/word_vectors.npy')
word_vectors = np.float32(word_vectors)
print ('Word embedding matrix loaded!')
print('Size of the vocabulary: ', len(words_list))
print('Size of the word embedding matrix: ', word_vectors.shape)
word2idx = {w:i for i,w in enumerate(words_list)}
print(list(word2idx.items())[:10])
word2idx['UNK']
# Loại bỏ các dấu câu, dấu ngoặc, chấm than chấm hỏi, vân vân..., chỉ chừa lại các kí tự chữ và số
import re
# re = regular expressions
strip_special_chars = re.compile("[^\w0-9 ]+")

def clean_sentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())
def get_sentence_indices(sentence, max_seq_length, _words_list):
    """
    Get index of each word in the sentence. Only letters, can be uppercase.
    
    Parameters
    ----------
    sentence: string
        Sentence that needs to process
    max_seq_length: int
        Max number of words in the sentence
    _words_list: list
        a copy of words_list
    """
    indices = np.zeros((max_seq_length), dtype='int32')
    
    # Lowercase and split the sentence into words
    words = [word.lower() for word in sentence.split()]
    
    # Get "unk" index
    unk_idx = word2idx['UNK']
    ### TODO 1 ###
    # Write code that fills the i-th index in "indices" with the i-th index in the "words"
    # NOTE: len(indices) can be shorter than len(words)
    ### START CODE HERE ###
    for idx, word in enumerate(words):
        if idx >= max_seq_length:
            break
        try:
            indices[idx] = word2idx[word]
        except:
            indices[idx] = unk_idx
    ### END CODE HERE ###
    return indices
# Ví dụ:
sentence = "Quán này bé_tí, nhưng siêu cute hạt_me."

# Tiền xử lý câu
sentence = clean_sentences(sentence)
print(sentence)
sentence_indices = get_sentence_indices(sentence, max_seq_length=10, _words_list=words_list)
print(sentence_indices)
print('Vector representation of sentence: {}'.format(sentence))
print(tf.nn.embedding_lookup(word_vectors,sentence_indices))
num_words = [len(clean_sentences(x).split()) for x in list(train_df['text'])]
print('The total number of samples is', len(train_df))
print('The total number of words in the files is', sum(num_words))
print('The average number of words in the files is', sum(num_words)/len(num_words))
import matplotlib.pyplot as plt
%matplotlib inline
plt.hist(num_words, 100)
plt.xlabel('Number of words in a sentence')
plt.ylabel('Frequency')
plt.axis([0, 600, 0, 5000])
plt.show()
MAX_SEQ_LENGTH = 200
def text2ids(df, max_length, _word_list):
    """
    Transform text in the dataframe into matrix index
    NOTE: this is the train_ids.npy in the dataset
    
    Parameters
    ----------
    df: DataFrame
        dataframe that stores the text
    max_length: int
        max length of a text
    _word_list: numpy.array
        array that stores the words in word vectors
    
    Returns
    -------
    ids: numpy.array
        len(df) * max_length, contains indices of text
    """
    ids = np.zeros((len(df), max_length), dtype='int32')
    for idx, text in enumerate(tqdm(df['text'])):
        ids[idx,:] = get_sentence_indices(clean_sentences(text), max_length, _word_list)
    return ids
print("Converting train_df to train_ids...")
train_ids = text2ids(train_df, MAX_SEQ_LENGTH, words_list)
np.save('train_ids.npy', train_ids)
print('Word indices of the first review: ')
print(train_ids[0])
# train_x, test_validation_x, train_y, test_validation_y  = train_test_split(train_ids, 
#                                                                            train_df['class'], test_size=0.2, random_state=2019)

train_x, validation_x, train_y, validation_y  = train_test_split(train_ids, train_df['class'], test_size=0.2)

# validation_x, test_x, validation_y, test_y = train_test_split(test_validation_x, 
#                                                               test_validation_y, test_size=0.5, random_state=2018)
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_dataset = train_dataset.batch(BATCH_SIZE)

validation_dataset = tf.data.Dataset.from_tensor_slices((validation_x, validation_y))
validation_dataset = validation_dataset.batch(BATCH_SIZE)

for idx, (x,y) in enumerate(train_dataset):
    if idx == 0:
        print('X =',x)
        print('y =',y)
print("Total: ", idx)
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision + recall + K.epsilon())
    return f1_val
num_classes = 2 # Binary output
lstm_output_dim = 64
epochs=100
# F1 score and Acc callback, when it reaches a threshold
class F1ScoreCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("get_f1") is not None:
            max_allowed = 0.98
            if(logs.get("get_f1") > max_allowed):
                print("\nReached {}% f1_score so cancelling training!".format(max_allowed))
                self.model.stop_training = True
class AccScoreCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("acc") is not None:
            max_allowed = 0.98
            if(logs.get("acc") > max_allowed):
                print("\nReached {}% acc so cancelling training!".format(max_allowed))
                self.model.stop_training = True
                
f1_score_callback = F1ScoreCallback()
acc_callback = AccScoreCallback()
model_cp = tf.keras.callbacks.ModelCheckpoint("weight_model.h5", 
                                              monitor="val_get_f1", save_best_only=True, save_weights_only=True)
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_get_f1", 
                                              patience = epochs/10, restore_best_weights=True)
word_vectors.shape
# Test with simple FNN
def create_test_model_1():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(word_vectors.shape[0], word_vectors.shape[1], name="embedding", 
                                  embeddings_initializer=tf.keras.initializers.Constant(word_vectors), trainable=False),
        tf.keras.layers.GRU(lstm_output_dim, dropout=0.42),
        tf.keras.layers.Dense(word_vectors.shape[1]*3, activation="tanh"),
        tf.keras.layers.Dense(word_vectors.shape[1]*2, activation="tanh"),
        tf.keras.layers.Dense(word_vectors.shape[1], activation="tanh"),
        tf.keras.layers.Dense(word_vectors.shape[1]*0.5, activation="tanh"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", 
              metrics=[get_f1, "accuracy"])
    return model

def create_test_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(word_vectors.shape[0], word_vectors.shape[1], name="embedding", 
                                  embeddings_initializer=tf.keras.initializers.Constant(word_vectors), trainable=False),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_output_dim, return_sequences=True), name="bidi_lstm_1"),
        tf.keras.layers.Dropout(0.42, name="dropout_1"),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_output_dim//2), name="bidi_lstm_2"),
        tf.keras.layers.Dropout(0.42, name="dropout_2"),
        tf.keras.layers.Dense(lstm_output_dim, activation='tanh', kernel_regularizer=tf.keras.regularizers.l1_l2(), name="dense_1"),
        tf.keras.layers.Dropout(0.42, name="dropout_3"),
        tf.keras.layers.Dense(128, activation='tanh',kernel_regularizer=tf.keras.regularizers.l1_l2(), name="dense_2"),
        tf.keras.layers.Dropout(0.42, name="dropout_4"),
        tf.keras.layers.Dense(64, activation='tanh', name="dense_3"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="dense_output")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", 
              metrics=[get_f1])
    return model
test_1=False

if test_1:
    model = create_test_model_1()
else:
    model = create_test_model()
    
model.summary()

history = model.fit(train_dataset, epochs=100, validation_data=validation_dataset, callbacks=[f1_score_callback,
                                                                                              acc_callback,
                                                                                              model_cp])
plt.plot(history.history['get_f1'])
plt.plot(history.history['val_get_f1'])
plt.title('model f1 score')
plt.ylabel('f1_score')

plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
model.save_weights("weight_model.h5")
from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
tokenized = []
result = []
threshold = 0.5

for idx, row in test_df.iterrows():
    row = clean_sentences(row.text)
    segmented_list = rdrsegmenter.tokenize(row)
    segmented_sentence = " ".join(item for innerlist in segmented_list for item in innerlist)
    indices = get_sentence_indices(segmented_sentence, MAX_SEQ_LENGTH, words_list)
    input_data = indices.reshape(1, *indices.shape)
    tokenized.append(input_data)
for ele in tokenized:
    prediction = model.predict(ele)
    result.append(1 if prediction >= threshold else 0)
print(result[:20])
print(tokenized[:20])
test_df
submission_df = pd.read_csv("../input/vietai-dataset/assignment4-data/Assignment4/sample_submission.csv")
submission_df["class"] = result
submission_df.head(20)
submission_df.to_csv("rnn_submission.csv", index=False)
