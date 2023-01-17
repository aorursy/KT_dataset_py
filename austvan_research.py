from __future__ import absolute_import

from __future__ import division

from __future__ import print_function





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/jigsaw-toxic-comment-classification-challenge"))

print(os.listdir("../input/nlpword2vecembeddingspretrained"))

print(os.listdir("../input/fasttext-wikinews"))

print(os.listdir("../input/fasttext-crawl-300d-2m"))

print(os.listdir("../input/glove-global-vectors-for-word-representation"))





#print(os.listdir("../input/jigsaw-unintended-bias-in-toxicity-classification"))



# Any results you write to the current directory are saved as output.
import datetime

import os, codecs

import pandas as pd

import numpy as np

import pkg_resources

import seaborn as sns

import time

import scipy.stats as stats



from sklearn import metrics

from sklearn import model_selection



# Loads word2vec.bin embeddings.

import gensim

from gensim.models import Word2Vec

from gensim.utils import simple_preprocess

from gensim.models.keyedvectors import KeyedVectors



# Loads Fastext Embeddings

from tqdm import tqdm



from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

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
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')



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

#train.head(5)
# #####################

# ADDS WEIGHTS TO LABELS

# #####################

# x_train = preprocess(train['comment_text'])

# weights = np.ones((len(x_train),)) / 4

# # Subgroup

# weights += (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4

# # Background Positive, Subgroup Negative

# weights += (( (train['target'].values>=0.5).astype(bool).astype(np.int) +

#    (train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4

# # Background Negative, Subgroup Positive

# weights += (( (train['target'].values<0.5).astype(bool).astype(np.int) +

#    (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4

# loss_weight = 1.0 / weights.mean()



# y_train = np.vstack([(train['target'].values>=0.5).astype(np.int),weights]).T

# y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values

# #####################

# END OF WEIGHTS TO LABELS

# #####################
train_df, validate_df = model_selection.train_test_split(train, test_size=0.2)

print('%d train comments, %d validate comments' % (len(train_df), len(validate_df)))

train_df.head()
# #####################

# Data Augmentation and Weights

# #####################

# Expands data set in the following ways:

#     NOTE: comment out to use one at a time.



# #######################################

# 1.) Adds previous competitions data to the training set. (done - tested)

# #######################################

# train_old = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')



# # Obtains columns

# id_col = train_old['id'].tolist()

# toxic_col = train_old['toxic'].tolist()

# comment_col = train_old['comment_text'].tolist()



# # creates frame (values already bool)

# old_lists = list(zip(id_col, toxic_col, comment_col))

# old_frame = pd.DataFrame(old_lists, columns =['id', 'target', 'comment_text']) 

# # old_frame.head(10)



# # appends frame

# train_df = train_df.append(old_frame, ignore_index=True, sort=False)

# train_df.fillna(0, inplace=True)

# #train.tail(10)



# #######################################

# end

# #######################################



# #######################################

# 2.) Removes subgroup positives. (tested)

# #######################################

# toxic = train_df.loc[train_df['target'] >= 0.5]



# # obtains toxic subgroups

# sub_pos_df = pd.DataFrame()

# for ident in identity_columns:

#     sub_pos_df = pd.concat([ sub_pos_df, toxic.loc[toxic[ident] >= 0.5] ], axis=0)

    

# # removes from train_df based on columns.

# train_df = train_df.drop(sub_pos_df.index.values)



# # # Sanity check

# print('# Toxic Identities')

# toxic = train_df[train_df['target'] >= 0.5]

# for ident in identity_columns:

#     print(ident, toxic.loc[toxic[ident] >= 0.5].shape[0])

# #######################################

# end

# #######################################



# #######################################

# 3.) Balance subgroup negatives via oversampling. (test)

# #######################################

non_toxic = train_df[train_df['target'] < 0.5]



# Finds the maximum subgroup negative group.

max_val = 0

for ident in identity_columns:

    val = non_toxic.loc[non_toxic[ident] >= 0.5].shape[0]

    if val >= max_val:

        max_val = val

print('balancing subgroup negatives to max value: ', max_val)

print('starting... ', train_df.shape[0])



# Over samples data and appends data.

a = 0

for ident in identity_columns:

    # Creates local, deep copy for replacing.

    ident_df = non_toxic[non_toxic[ident] >= 0.5].copy(deep=True)

    val = max_val - ident_df.shape[0]

    ident_oversampled = ident_df.sample(n=val, replace=True)

    train_df = pd.concat([train_df, ident_oversampled], axis=0)

    a = a + ident_oversampled.shape[0]

    print('size   : ', ident_df.shape[0])

    print('adding : ', ident_oversampled.shape[0])

    print('total  : ', train_df.shape[0], '\n')



# Shuffles data

train_df = train_df.sample(frac=1)

    

# Sanity check

print('added... ', a)

print('final... ', train_df.shape[0])

non_toxic = train_df[train_df['target'] < 0.5]

for ident in identity_columns:

    print(ident, non_toxic.loc[non_toxic[ident] >= 0.5].shape[0])

# #######################################

# end

# #######################################



# #######################################

# 4.)  Oversample subgroup negatives (extreme). (todo)

# #######################################

# non_toxic = train_df[train_df['target'] < 0.5]



# # Over samples data and appends data.

# max_val = 100000

# a = 0

# for ident in identity_columns:

#     # Creates local, deep copy for replacing.

#     ident_df = non_toxic[non_toxic[ident] >= 0.5].copy(deep=True)

#     val = max_val - ident_df.shape[0]

#     ident_oversampled = ident_df.sample(n=val, replace=True)

#     train_df = pd.concat([train_df, ident_oversampled], axis=0)

#     a = a + ident_oversampled.shape[0]

#     print('size   : ', ident_df.shape[0])

#     print('adding : ', ident_oversampled.shape[0])

#     print('total  : ', train_df.shape[0], '\n')



# # Shuffles data

# train_df = train_df.sample(frac=1)

    

# # Sanity check

# print('added... ', a)

# print('final... ', train_df.shape[0])

# non_toxic = train_df[train_df['target'] < 0.5]

# for ident in identity_columns:

#     print(ident, non_toxic.loc[non_toxic[ident] >= 0.5].shape[0])



# #######################################

# end

# #######################################



# #######################################

# 5.) Over sample subgroup positives (fine-tuned). 

# #######################################





# #######################################

# end

# #######################################
# Augment: Remove Subgroup Pos.

# subgroup pos = identity > 0.5 && target > 0.5

# = full data w/o subgroups appended to subgroups non toxic.



# obtains subgroup negatives.

# s_neg = pd.DataFrame(data=None, columns=train_df.columns, index=train_df.index)

# non_toxic = complete.loc[complete['target'] < 0.5]

# for ident in identity_columns:

#     s_neg = s_neg.append(train_df.loc[non_toxic[ident >= 0.5]])



# # obtain background pos/neg

# background = pd.DataFrame(data=None, columns=train_df.columns, index=train_df.index)

# for ident in identity_columns:

#     background = background.append(train_df.loc[train_df[ident < .5]])

MAX_NUM_WORDS = 10000

TOXICITY_COLUMN = 'target'

TEXT_COLUMN = 'comment_text'



# Create a text tokenizer.

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

tokenizer.fit_on_texts(train_df[TEXT_COLUMN])



# All comments must be truncated or padded to be the same length.

MAX_SEQUENCE_LENGTH = 250

def pad_text(texts, tokenizer):

    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)
# IMPORTANT # ###########################

# Ensure that the desired embedding is loaded and that the correct dimensions are set.

# Different embedding files require various methods to be loaded. Ensure correct loading is uncommented.



EMBEDDINGS_PATH = '../input/fasttext-wikinews/wiki-news-300d-1M.vec'

#EMBEDDINGS_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

#EMBEDDINGS_PATH = '../input/nlpword2vecembeddingspretrained/glove.6B.300d.txt'

#EMBEDDINGS_PATH = '../input/nlpword2vecembeddingspretrained/glove.6B.200d.txt'

#EMBEDDINGS_PATH = '../input/glove-global-vectors-for-word-representation/glove.twitter.27B.200d.txt'

EMBEDDINGS_DIMENSION = 300

# #######################################



DROPOUT_RATE = 0.3

LEARNING_RATE = 0.00005

NUM_EPOCHS = 10

BATCH_SIZE = 128



def train_model(train_df, validate_df, tokenizer):

    # Prepare data

    train_text = pad_text(train_df[TEXT_COLUMN], tokenizer)

    train_labels = to_categorical(train_df[TOXICITY_COLUMN])

    validate_text = pad_text(validate_df[TEXT_COLUMN], tokenizer)

    validate_labels = to_categorical(validate_df[TOXICITY_COLUMN])

    

# #####################

# LOAD EMBEDDINGS

# #####################

# Commet out undesired embeddings. Only one embedding may be loaded at a time (with this model).

    embeddings_index = {}

    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1,

                                 EMBEDDINGS_DIMENSION))



# FASTEXT EMBEDDINGS ############################ (.vec)

    print('Loading word embeddings.')

    f = codecs.open(EMBEDDINGS_PATH, encoding='utf-8')

    for line in tqdm(f):

        values = line.rstrip().rsplit(' ')

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs

    f.close()

    print('Preparing embedding matrix.')

    words_not_found = []

    for word, i in tokenizer.word_index.items():

        embedding_vector = embeddings_index.get(word)

        if (embedding_vector is not None) and len(embedding_vector) > 0:

            # words not found in embedding index will be all-zeros.

            embedding_matrix[i] = embedding_vector

        else:

            words_not_found.append(word)

    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

# #################################################



# GLOVE EMBEDDINGS # ############################## (.txt)

#     print('Loading Glove embeddings.')

#     with open(EMBEDDINGS_PATH) as f:

#         for line in f:

#             values = line.split()

#             word = values[0]

#             coefs = np.asarray(values[1:], dtype='float32')

#             embeddings_index[word] = coefs

#     print('Preparing embedding matrix.')

#     num_words_in_embedding = 0

#     for word, i in tokenizer.word_index.items():

#         embedding_vector = embeddings_index.get(word)

#         if embedding_vector is not None:

#             num_words_in_embedding += 1

#             # words not found in embedding index will be all-zeros.

#             embedding_matrix[i] = embedding_vector

# # ################################################



# WORD2VEC EMBEDDINGS ############################ (.bin)

#     word_vectors = KeyedVectors.load_word2vec_format(EMBEDDINGS_PATH, binary=True)

#     num_words_in_embedding = 0

#     for word, i in tokenizer.word_index.items():

#         if i >= MAX_NUM_WORDS:

#             continue

#         try:

#             embedding_vector = word_vectors[word]

#             embedding_matrix[i] = embedding_vector

#         except KeyError:

#             embedding_matrix[i] = np.zeros((EMBEDDINGS_DIMENSION))

# ############################################### 

# #####################

# END OF LOAD EMBEDDINGS

# #####################    

    



    # Create model layers.

    def get_convolutional_neural_net_layers():

        """Returns (input_layer, output_layer)"""

        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

                               

        embedding_layer_static = Embedding(len(tokenizer.word_index) + 1,

                                    EMBEDDINGS_DIMENSION,

                                    weights=[embedding_matrix],

                                    input_length=MAX_SEQUENCE_LENGTH,

                                    trainable=False)   

                               

#         embedding_layer_non_static = Embedding(len(tokenizer.word_index) + 1,

#                                          EMBEDDINGS_DIMENSION,

#                                          weights=[embedding_matrix],

#                                          input_length=MAX_SEQUENCE_LENGTH,

#                                          trainable=True)



        # Hybrid: Nonstatic and Static

#         xs = embedding_layer_static(sequence_input)

#         xns = embedding_layer_non_static(sequence_input)

#         xs= Conv1D(128, 2, activation='relu', padding='same')(xs)

#         xns = Conv1D(128, 2, activation='relu', padding='same')(xns)   

#         xs = MaxPooling1D(40, padding='same')(xs)

#         xns = MaxPooling1D(40, padding='same')(xns)

#         xs = Conv1D(128, 3, activation='relu', padding='same')(xs)

#         xns = Conv1D(128, 3, activation='relu', padding='same')(xns)

#         xs = MaxPooling1D(40, padding='same')(xs)

#         xns = MaxPooling1D(40, padding='same')(xns)

#         xs = Conv1D(128, 4, activation='relu', padding='same')(xs)

#         xns = Conv1D(128, 4, activation='relu', padding='same')(xns)

#         xs = MaxPooling1D(40, padding='same')(xs)

#         xns = MaxPooling1D(40, padding='same')(xns)

#         xs = Conv1D(128, 5, activation='relu', padding='same')(xs)

#         xns = Conv1D(128, 5, activation='relu', padding='same')(xns)

#         x = Maximum()([xs,xns])

#         x = MaxPooling1D(40, padding='same')(x)

#         x = Flatten()(x)

#         x = Dropout(DROPOUT_RATE)(x)

#         x = Dense(128, activation='relu')(x)

#         preds = Dense(2, activation='softmax')(x)

        # End of Hybrid: Static and Non Static



        # Static

        xs = embedding_layer_static(sequence_input)

        xs = Conv1D(128, 2, activation='relu', padding='same')(xs)   

        xs = MaxPooling1D(40, padding='same')(xs)

        xs = Conv1D(128, 3, activation='relu', padding='same')(xs)

        xs = MaxPooling1D(40, padding='same')(xs)

        xs = Conv1D(128, 4, activation='relu', padding='same')(xs)

        xs = MaxPooling1D(40, padding='same')(xs)

        xs = Conv1D(128, 5, activation='relu', padding='same')(xs)

        xs = MaxPooling1D(40, padding='same')(xs)

        x = Flatten()(xs)

        x = Dropout(DROPOUT_RATE)(x)

        x = Dense(128, activation='relu')(x)

        preds = Dense(2, activation='softmax')(x)

        # End of Static

        

        # Non Static

#         xns = embedding_layer_non_static(sequence_input)

#         xns = Conv1D(128, 2, activation='relu', padding='same')(xns)   

#         xns = MaxPooling1D(40, padding='same')(xns)

#         xns = Conv1D(128, 3, activation='relu', padding='same')(xns)

#         xns = MaxPooling1D(40, padding='same')(xns)

#         xns = Conv1D(128, 4, activation='relu', padding='same')(xns)

#         xns = MaxPooling1D(40, padding='same')(xns)

#         xns = Conv1D(128, 5, activation='relu', padding='same')(xns)

#         xns = MaxPooling1D(40, padding='same')(xns)

#         x = Flatten()(xns)

#         x = Dropout(DROPOUT_RATE)(x)

#         x = Dense(128, activation='relu')(x)

#         preds = Dense(2, activation='softmax')(x)

        # End of Nonstatic

        

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



model = train_model(train_df, validate_df, tokenizer)
MODEL_NAME = 'my_model'

validate_df[MODEL_NAME] = model.predict(pad_text(validate_df[TEXT_COLUMN], tokenizer))[:, 1]
validate_df.head()
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