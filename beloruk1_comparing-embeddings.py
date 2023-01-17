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
from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import datetime

import os, gc

import pandas as pd

import numpy as np

import pkg_resources

import seaborn as sns

import time

import scipy.stats as stats



from sklearn import metrics

from sklearn import model_selection



from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, Embedding, Dropout, Reshape, SpatialDropout1D, Activation, LSTM, GRU, Conv1D, Conv2D, MaxPool2D, GlobalMaxPooling1D, Flatten, Concatenate, Bidirectional, GlobalAveragePooling1D, GlobalMaxPool1D, MaxPooling1D, concatenate

from keras.layers import Embedding

from keras.layers import Input

from keras.layers import Conv1D

from keras.layers import MaxPooling1D

from keras.layers import Flatten

from keras.layers import Dropout

from keras.layers import Dense

from keras.optimizers import RMSprop

from keras.models import Model

from keras.models import load_model



import gensim

from gensim.models import KeyedVectors
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

print('loaded %d records' % len(train))



# Make sure all comment_text values are strings

train['comment_text'] = train['comment_text'].astype(str) 



# List all identities

identity_columns = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']



# Convert taget and identity columns to booleans

def convert_to_bool(df, col_name):

    df[col_name] = np.where(df[col_name] >= 0.5, True, False)

    

def convert_dataframe_to_bool(df):

    bool_df = df.copy()

    for col in ['target'] + identity_columns:

        convert_to_bool(bool_df, col)

    return bool_df



train = convert_dataframe_to_bool(train)
train_df, validate_df = model_selection.train_test_split(train, test_size=0.2)

print('%d train comments, %d validate comments' % (len(train_df), len(validate_df)))

MAX_NUM_WORDS = 95000 #10000

TOXICITY_COLUMN = 'target'

TEXT_COLUMN = 'comment_text'



# Create a text tokenizer.

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

tokenizer.fit_on_texts(train_df[TEXT_COLUMN])



# All comments must be truncated or padded to be the same length.

MAX_SEQUENCE_LENGTH = 250

def pad_text(texts, tokenizer):

    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)
#EMBEDDINGS_PATH = '../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt'

EMBEDDINGS_DIMENSION = 300 #100

DROPOUT_RATE = 0.3

LEARNING_RATE = 0.00005

NUM_EPOCHS = 10

BATCH_SIZE = 128
## some config values 

# EMBED_SIZE = 300 # how big is each word vector EMBEDDINGS_DIMENSION

# MAX_FEATURES = 95000 # how many unique words to use (i.e num rows in embedding vector) MAX_NUM_WORDS

# MAX_NUMWORDS = 70 # max number of words in a question to use MAX_SEQUENCE_LENGTH



def load_glove():

    print('Loading glove embeddings...')

    EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore'))

    return make_matrix(embeddings_index)

    

def load_fasttext():  

    print('Loading wiki-news embeddings...')

    EMBEDDING_FILE = '../input/wikinews300d1mvec/wiki-news-300d-1M.vec'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    return make_matrix(embeddings_index)



def load_para():

    print('Loading paragram embeddings...')

    EMBEDDING_FILE = '../input/paragram-300-sl999/paragram_300_sl999/paragram_300_sl999/paragram_300_sl999.txt'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    return make_matrix(embeddings_index)



def load_wordvec():

    print('Loading word2vec embeddings...')

    EMBEDDING_FILE = '../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin'

    embeddings_index = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

    word_index = tokenizer.word_index

    nb_words = min(MAX_NUM_WORDS, len(word_index))

    embedding_matrix = np.zeros((nb_words, EMBEDDINGS_DIMENSION))

    for word, i in word_index.items():

        if i >= MAX_NUM_WORDS: continue

        if word in embeddings_index.vocab:

            embedding_matrix[i] = embeddings_index.word_vec(word)

    return embedding_matrix



def make_matrix(embeddings_index):

    #print('Making matrix...')

    #embedding_matrix = np.zeros((len(tokenizer.word_index) + 1,EMBEDDINGS_DIMENSION))

    #num_words_in_embedding = 0

    #for word, i in tokenizer.word_index.items():

    #    embedding_vector = embeddings_index.get(word)

    #    if embedding_vector is not None:

    #        num_words_in_embedding += 1

    #        # words not found in embedding index will be all-zeros.

    #        embedding_matrix[i] = embedding_vector

    #return embedding_matrix



    print('Making matrix...')

    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]

    word_index = tokenizer.word_index

    nb_words = min(MAX_NUM_WORDS, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= MAX_NUM_WORDS: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix



print('Loading Glove...')

matrix_glove = load_glove()



#print('Loading Fasttext...')

#matrix_fasttext = load_fasttext()



print('Loading Paragram...')

matrix_para = load_para()



print('Loading Wordvec...')

matrix_wordvec = load_wordvec()



embedding_matrix = np.mean((matrix_glove

                            #, matrix_fasttext

                            , matrix_para

                            , matrix_wordvec

                            ), axis = 0)

np.shape(embedding_matrix)

del matrix_glove

#del matrix_fasttext

del matrix_para

del matrix_wordvec

gc.collect()
def load_embeddings(path, dim):

    # Load embeddings

    print('loading embeddings from ', path)

    embeddings_index = {}

    with open(path) as f:

        for line in f:

            values = line.split()

            word = values[0]

            coefs = np.asarray(values[1:], dtype='float32')

            embeddings_index[word] = coefs



    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1,dim))

    num_words_in_embedding = 0

    for word, i in tokenizer.word_index.items():

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            num_words_in_embedding += 1

            # words not found in embedding index will be all-zeros.

            embedding_matrix[i] = embedding_vector

            

    return embedding_matrix



#test_emb = load_embeddings('../input/glove840b300dtxt/glove.840B.300d.txt', EMBEDDINGS_DIMENSION)

#test_emb.shape
def prepare_data(train_df, validate_df, tokenizer):    

    print('preparing data')

    train_text = pad_text(train_df[TEXT_COLUMN], tokenizer)

    train_labels = to_categorical(train_df[TOXICITY_COLUMN])

    validate_text = pad_text(validate_df[TEXT_COLUMN], tokenizer)

    validate_labels = to_categorical(validate_df[TOXICITY_COLUMN])

    

    return train_text, train_labels, validate_text, validate_labels, tokenizer



train_text, train_labels, validate_text, validate_labels, tokenizer = prepare_data(train_df, validate_df, tokenizer)

def train_model(train_text, train_labels, validate_text, validate_labels, tokenizer):

    # Prepare data



    #embedding_matrix = load_embeddings(EMBEDDINGS_PATH, EMBEDDINGS_DIMENSION)

    

    # Create model layers.

    def get_gru_neural_net_layers():

        print('Modeling...')

        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

        x = Embedding(MAX_NUM_WORDS, EMBEDDINGS_DIMENSION, weights=[embedding_matrix], trainable=False)(sequence_input)

        x = Bidirectional(GRU(128, return_sequences=True))(x)

        x = GlobalMaxPool1D()(x)

        x = Dense(16, activation="relu")(x)

        x = Dropout(0.1)(x)

        preds = Dense(2, activation='softmax')(x)

        return sequence_input, preds

    

    def get_convolutional_neural_net_layers():

        """Returns (input_layer, output_layer)"""

        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

        #embedding_layer = Embedding(len(tokenizer.word_index) + 1,

        #                            EMBEDDINGS_DIMENSION,

        #                            weights=[embedding_matrix],

        #                            input_length=MAX_SEQUENCE_LENGTH,

         #                           trainable=False)

        #x = embedding_layer(sequence_input)

        x = Embedding(MAX_NUM_WORDS, EMBEDDINGS_DIMENSION, weights=[embedding_matrix], trainable=False)(sequence_input)

        

        x = Conv1D(128, 2, activation='relu', padding='same')(x)

        x = MaxPooling1D(5, padding='same')(x)

        x = Conv1D(128, 3, activation='relu', padding='same')(x)

        x = MaxPooling1D(5, padding='same')(x)

        x = Conv1D(128, 4, activation='relu', padding='same')(x)

        x = MaxPooling1D(40, padding='same')(x)

        x = Flatten()(x)

        x = Dropout(DROPOUT_RATE)(x)

        x = Dense(128, activation='relu')(x)

        preds = Dense(2, activation='softmax')(x)

        return sequence_input, preds



    # Compile model.

    print('compiling model')

    input_layer, output_layer = get_convolutional_neural_net_layers()

    model = Model(input_layer, output_layer)

    model.compile(loss='categorical_crossentropy',

                  optimizer=RMSprop(lr=LEARNING_RATE),

                  metrics=['acc'])



    # Train model.

    print('training model')

    model.fit(train_text,

              train_labels,

              batch_size=BATCH_SIZE,

              epochs=NUM_EPOCHS,

              validation_data=(validate_text, validate_labels),

              verbose=2)



    return model



model = train_model(train_text, train_labels, validate_text, validate_labels, tokenizer)
MODEL_NAME = 'emb_model'

validate_df[MODEL_NAME] = model.predict(pad_text(validate_df[TEXT_COLUMN], tokenizer))[:, 1]
SUBGROUP_AUC = 'subgroup_auc'

BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative

BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive



def compute_auc(y_true, y_pred):

    try:

        return metrics.roc_auc_score(y_true, y_pred)

    except ValueError:

        return np.nan



def compute_subgroup_auc(df, subgroup, label, model_name):

    subgroup_examples = df[df[subgroup]]

    return compute_auc(subgroup_examples[label], subgroup_examples[model_name])



def compute_bpsn_auc(df, subgroup, label, model_name):

    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""

    subgroup_negative_examples = df[df[subgroup] & ~df[label]]

    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]

    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)

    return compute_auc(examples[label], examples[model_name])



def compute_bnsp_auc(df, subgroup, label, model_name):

    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""

    subgroup_positive_examples = df[df[subgroup] & df[label]]

    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]

    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)

    return compute_auc(examples[label], examples[model_name])



def compute_bias_metrics_for_model(dataset,

                                   subgroups,

                                   model,

                                   label_col,

                                   include_asegs=False):

    """Computes per-subgroup metrics for all subgroups and one model."""

    records = []

    for subgroup in subgroups:

        record = {

            'subgroup': subgroup,

            'subgroup_size': len(dataset[dataset[subgroup]])

        }

        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)

        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)

        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)

        records.append(record)

    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)



bias_metrics_df = compute_bias_metrics_for_model(validate_df, identity_columns, MODEL_NAME, TOXICITY_COLUMN)

bias_metrics_df
def calculate_overall_auc(df, model_name):

    true_labels = df[TOXICITY_COLUMN]

    predicted_labels = df[model_name]

    return metrics.roc_auc_score(true_labels, predicted_labels)



def power_mean(series, p):

    total = sum(np.power(series, p))

    return np.power(total / len(series), 1 / p)



def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):

    bias_score = np.average([

        power_mean(bias_df[SUBGROUP_AUC], POWER),

        power_mean(bias_df[BPSN_AUC], POWER),

        power_mean(bias_df[BNSP_AUC], POWER)

    ])

    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)

    

get_final_metric(bias_metrics_df, calculate_overall_auc(validate_df, MODEL_NAME))
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')
submission['prediction'] = model.predict(pad_text(test[TEXT_COLUMN], tokenizer))[:, 1]

submission.to_csv('submission.csv')