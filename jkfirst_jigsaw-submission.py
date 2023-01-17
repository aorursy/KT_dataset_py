from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import datetime

import os

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
train = pd.read_csv('../input/train.csv')

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
print(os.listdir("../input"))
EMBEDDINGS_PATH = '../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt'

EMBEDDINGS_DIMENSION = 100

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



    # Create model layers.

    def get_convolutional_neural_net_layers():

        """Returns (input_layer, output_layer)"""

        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

        embedding_layer = Embedding(len(tokenizer.word_index) + 1,

                                    EMBEDDINGS_DIMENSION,

                                    #weights=[embedding_matrix],

                                    input_length=MAX_SEQUENCE_LENGTH,

                                    trainable=False)

        x = embedding_layer(sequence_input)

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



model = train_model(train_df, validate_df, tokenizer)
MODEL_NAME = 'my_model'

validate_df[MODEL_NAME] = model.predict(pad_text(validate_df[TEXT_COLUMN], tokenizer))[:,1]
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
test = pd.read_csv('../input/test.csv')

submission = pd.read_csv('../input/sample_submission.csv', index_col='id')
submission['prediction'] = model.predict(pad_text(test[TEXT_COLUMN], tokenizer))[:, 1]

submission.to_csv('submission.csv')