import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
import gc

from tensorflow.python.keras.layers import LSTM, CuDNNLSTM,CuDNNGRU,Conv1D, MaxPooling1D

from nltk.stem import SnowballStemmer
from string import punctuation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.optimizers import Adam

import sys



from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve,log_loss
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df
train_df = import_data('../input/toxic-classification-trainset/train.csv')
test_df = import_data('../input/toxic-classification-testset/test.csv')
MAX_NB_WORDS = 100000    # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 200 # max length of each entry (sentence), including padding
VALIDATION_SPLIT = 0.1   # data for validation (not used in training)
EMBEDDING_DIM = 300      # embedding dimensions for word vectors (word2vec/GloVe) 
GLOVE_DIR = "../input/glove840b300dtxt/glove.840B.300d.txt"
def clean_text(text):    
    text = text.lower().split()
    text = " ".join(text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+\-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text

special_character_removal=re.compile(r'[^a-z\d ]',re.IGNORECASE)
replace_numbers=re.compile(r'\d+',re.IGNORECASE)

def text_to_wordlist(text):
    text = text.lower().split()

    
    text = " ".join(text)
    
    #Remove Special Characters
    text=special_character_removal.sub('',text)
    #Replace Numbers
    text=replace_numbers.sub('',text)
    
    return(text)
print('Processing text dataset')

train_df['comment_text'] = train_df['comment_text'].map(lambda x: clean_text(x))
test_df['comment_text'] = test_df['comment_text'].map(lambda x: clean_text(x))
print('Train shape: ', train_df.shape)
print('Test shape: ', test_df.shape)
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train_df[labels].values


comments = []
for text in train_df["comment_text"].fillna("NA").values:
    comments.append(text_to_wordlist(text))
    
test_comments=[]
for text in test_df["comment_text"].fillna("NA").values:
    test_comments.append(text_to_wordlist(text))
tokenizer = Tokenizer(num_words=MAX_NB_WORDS) # define Tokenize text function
tokenizer.fit_on_texts(comments)#fit the function on the text
sequences = tokenizer.texts_to_sequences(comments)# convert  to sequence
test_sequences = tokenizer.texts_to_sequences(test_comments)
word_index = tokenizer.word_index #num of unique tokens
print('Vocabulary size:', len(word_index))
#Limit size  to 200 and pad the sequence
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of pre train data tensor:', data.shape)

test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of pre test data tensor:', test_data.shape)

data_post = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,padding='post', truncating='post')
print('Shape of post train data tensor:', data_post.shape)

test_data_post = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
print('Shape of post test data tensor:', test_data_post.shape)
embeddings_index = {}
f = open(GLOVE_DIR)
print('Loading GloVe from:', GLOVE_DIR,'...', end='')

for line in f:
    values = line.rstrip().rsplit(' ')
    word = values[0]
    embeddings_index[word] = np.asarray(values[1:], dtype='float32')
f.close()
print("Done.\n Proceeding with Embedding Matrix...")
print(f'Found {len(embeddings_index)} word vectors', end="")

nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print(" Completed!")
del(train_df)
del(test_df)
del(embedding_vector)
del(tokenizer)
del(sequences)
del(test_sequences)
del(word_index)
del(comments)
del(test_comments)
del(embeddings_index)
del(values)
del(special_character_removal)
del(replace_numbers)
gc.collect() 
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))
def build_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):
    inp = Input(shape = (MAX_SEQUENCE_LENGTH,))
    x1 = Embedding(nb_words, EMBEDDING_DIM, weights = [embedding_matrix], trainable = False)(inp)
    x1 = SpatialDropout1D(0.4)(x1)
    x1 = Bidirectional(CuDNNGRU(64, return_sequences = True))(x1)
    x1 = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x1)
    avg_pool1 = GlobalAveragePooling1D()(x1)
    max_pool1 = GlobalMaxPooling1D()(x1)
    
    inp_post = Input(shape = (MAX_SEQUENCE_LENGTH,))
    x2 = Embedding(nb_words, EMBEDDING_DIM, weights = [embedding_matrix], trainable = False)(inp_post)
    x2 = SpatialDropout1D(0.4)(x2)
    x2 = Bidirectional(CuDNNGRU(64, return_sequences = True))(x2)
    x2 = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x2)
    avg_pool2 = GlobalAveragePooling1D()(x2)
    max_pool2 = GlobalMaxPooling1D()(x2)
    
    x = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])

    pred = Dense(6, activation = "sigmoid")(x)
    model = Model(inputs=[inp, inp_post], outputs=pred)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), 
                  metrics = ["accuracy"])
    return model


from sklearn.metrics import log_loss
import numpy as np

test_predicts_list = []

def train_folds(data,data_post, y, fold_count, batch_size):
    print("Starting to train models...")
    fold_size = len(data) // fold_count
    models = []
    for fold_id in range(0, fold_count):
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(data)

        print("Fold {0}".format(fold_id))
        
        train_x = np.concatenate([data[:fold_start], data[fold_end:]])
        train_xp = np.concatenate([data_post[:fold_start], data_post[fold_end:]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:]])

        val_x = data[fold_start:fold_end]
        val_xp = data_post[fold_start:fold_end]
        val_y = y[fold_start:fold_end]
        
        file_path="cnngru_fold{0}.h5".format(fold_id)
        model = build_model(lr = 1e-3, lr_d = 0, units = 128, dr = 0.2)
        check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                              save_best_only = True, mode = "min")
        ra_val = RocAucEvaluation(validation_data = ([val_x, val_xp], val_y), interval = 1)
        early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)
        hist = model.fit([train_x, train_xp], train_y, batch_size=128, epochs=8, verbose = 1, 
                         validation_data = ([val_x, val_xp], val_y), 
                         callbacks = [ra_val, check_point, early_stop])
        model.load_weights(file_path)
        best_score = min(hist.history['val_loss'])
        
        print("Fold {0} loss {1}".format(fold_id, best_score))
#         print("Predicting validation...")
#         val_predicts_path = "cnngru_val_predicts{0}.npy".format(fold_id)
#         val_predicts = model.predict([val_x, val_xp], batch_size=1024, verbose=1)
#         np.save(val_predicts_path, val_predicts)
        
        print("Predicting results...")
        test_predicts_path = "cnngru_test_predicts{0}.npy".format(fold_id)
        test_predicts = model.predict([test_data, test_data_post], batch_size=1024, verbose=1)
        test_predicts_list.append(test_predicts)
        np.save(test_predicts_path, test_predicts)
train_folds(data, data_post, y, 3, 128)
test_df = pd.read_csv('../input/toxic-classification-testset/test.csv')
test_predicts_am = np.zeros(test_predicts_list[0].shape)

for fold_predict in test_predicts_list:
    test_predicts_am += fold_predict

test_predicts_am = (test_predicts_am / len(test_predicts_list))

test_ids = test_df["id"].values
test_ids = test_ids.reshape((len(test_ids), 1))

test_predicts_am = pd.DataFrame(data=test_predicts_am, columns=labels)
test_predicts_am["id"] = test_ids
test_predicts_am = test_predicts_am[["id"] + labels]
test_predicts_am.to_csv("3fold_cnngru.csv", index=False)
