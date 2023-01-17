
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers
from tqdm.notebook import tqdm
from tokenizers import BertWordPieceTokenizer
import random
from sklearn.metrics import f1_score
import os
import keras.backend as K
from kaggle_datasets import KaggleDatasets
from transformers import TFAutoModel, AutoTokenizer
random.seed(0)
VAL_PROPORTION = 0
MAX_SEQ_LEN = 224
def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    #print (enc_di)
    return np.array(enc_di['input_ids'])
train_val = pd.read_csv("../input/analytics-vidhya-nlp-junta-hackathon/train.csv")
train_val_es = pd.read_csv("../input/analytics-vidhya-nlp-junta-hackathon/train-google-es.csv")
train_val_fr = pd.read_csv("../input/analytics-vidhya-nlp-junta-hackathon/train-google-fr_orig.csv")
test = pd.read_csv("../input/analytics-vidhya-nlp-junta-hackathon/test.csv")
test_es = pd.read_csv("../input/analytics-vidhya-nlp-junta-hackathon/test-google-es.csv")
test_fr = pd.read_csv("../input/analytics-vidhya-nlp-junta-hackathon/test-google-fr.csv")
imp_cols = ['review_id', 'user_review','user_suggestion']
train_val = pd.concat([train_val[imp_cols], train_val_fr[imp_cols],train_val_es[imp_cols]])
train_val['user_suggestion'].value_counts()
train_val = train_val[train_val['user_suggestion']!='[2]']
train_val['user_suggestion'].value_counts()
train_val['user_review'] = list(map(lambda x: str(x), train_val['user_review']))
train_val['user_suggestion'] = list(map(lambda x: int(x), train_val['user_suggestion']))

print (train_val.shape)
train_val['user_suggestion'].value_counts()
test_es['user_review'] = list(map(lambda x: str(x), test_es['user_review']))
test_fr['user_review'] = list(map(lambda x: str(x), test_fr['user_review']))

def reverse_review(string):
    string_splitted = string.split()
    string_reversed = reversed(string_splitted)
    return ' '.join(string_reversed)
train_val['user_review_reversed'] = list(map(lambda x: reverse_review(x), train_val['user_review']))
test['user_review_reversed'] = list(map(lambda x: reverse_review(x), test['user_review']))
test_es['user_review_reversed'] = list(map(lambda x: reverse_review(x), test_es['user_review']))
test_fr['user_review_reversed'] = list(map(lambda x: reverse_review(x), test_fr['user_review']))
## Function to split data into train and test sets 

def train_test_split(df,test_prop=0.25):
    n_rows = df.shape[0]
    list_indices = list(range(n_rows))
    random.shuffle(list_indices)
    n_rows_test = int(n_rows*test_prop)
    n_rows_train = n_rows - n_rows_test
    df_train = df.iloc[list_indices[:n_rows_train]]
    df_test = df.iloc[list_indices[n_rows_train:]]
    return df_train, df_test
train, val =train_test_split(train_val, VAL_PROPORTION)
print ("Number of rows in train dataset: ", train.shape)
print ("Number of rows in test dataset: ", val.shape)

AUTO = tf.data.experimental.AUTOTUNE

# Data access
#GCS_DS_PATH = KaggleDatasets().get_gcs_path()

# Create strategy from tpu
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)
# Configuration
EPOCHS = 4
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
MAX_LEN = 224
MODEL = 'jplu/tf-xlm-roberta-large'
review_lengths = list(map(lambda x: len(x.split()), train['user_review']))

print ("10 percentile length of training user reviews: ", np.quantile(review_lengths, 0.1))
print ("25 percentile length of training user reviews: ", np.quantile(review_lengths, 0.25))
print ("50 percentile length of training user reviews: ", np.quantile(review_lengths, 0.5))
print ("75 percentile length of training user reviews: ", np.quantile(review_lengths, 0.75))
print ("90 percentile length of training user reviews: ", np.quantile(review_lengths, 0.9))
print ("99 percentile length of training user reviews: ", np.quantile(review_lengths, 0.99))
tokenizer = AutoTokenizer.from_pretrained(MODEL)
%%time

train_x = regular_encode(train.user_review.astype(str), tokenizer, maxlen=MAX_SEQ_LEN)
#val_x = regular_encode(val.user_review_reversed.astype(str), tokenizer, maxlen=MAX_SEQ_LEN)
test_x = regular_encode(test.user_review.astype(str), tokenizer, maxlen=MAX_SEQ_LEN)
test_es_x = regular_encode(test_es.user_review.astype(str), tokenizer, maxlen=MAX_SEQ_LEN)
test_fr_x = regular_encode(test_fr.user_review.astype(str), tokenizer, maxlen=MAX_SEQ_LEN)

train_y = np.array(list(train.user_suggestion))
#val_y = np.array(list(val.user_suggestion))


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_x, train_y))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

#valid_dataset = (
#    tf.data.Dataset
#    .from_tensor_slices((val_x, val_y))
#    .batch(BATCH_SIZE)
#    .cache()
#    .prefetch(AUTO)
#)

#valid_dataset_pred = (
#    tf.data.Dataset
#    .from_tensor_slices(val_x)
#    .batch(BATCH_SIZE)
#)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_x)
    .batch(BATCH_SIZE)
)

test_es_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_es_x)
    .batch(BATCH_SIZE)
)

test_fr_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_fr_x)
    .batch(BATCH_SIZE)
)

def build_model(transformer, max_len=128):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    
    #dense1 = Dense(256, activation='relu')(cls_token)
    #dropout = tf.keras.layers.Dropout(0.25)(dense1)
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
    
    return model
%%time
with strategy.scope():
    transformer_layer = TFAutoModel.from_pretrained(MODEL)
    model = build_model(transformer_layer, max_len=MAX_SEQ_LEN)
model.summary()

N_STEPS = train_x.shape[0] // BATCH_SIZE
print (N_STEPS)
print (BATCH_SIZE)
%%time

train_history = model.fit(
    train_dataset,
    steps_per_epoch=N_STEPS,
#    validation_data=valid_dataset,
    epochs=EPOCHS
)

optimal_threshold = 0.45
test_predictions = model.predict(test_dataset, verbose=1)

test_es_predictions = model.predict(test_es_dataset, verbose=1)
test_fr_predictions = model.predict(test_fr_dataset, verbose=1)

test_predictions_df_1 = pd.DataFrame()
test_predictions_df_1['review_id'] = test['review_id']
test_predictions_df_1['prob_english'] = test_predictions

test_predictions_df_2 = pd.DataFrame()
test_predictions_df_2['review_id'] = test_es['review_id']
test_predictions_df_2['prob_es'] = test_es_predictions

test_predictions_df_3 = pd.DataFrame()
test_predictions_df_3['review_id'] = test_fr['review_id']
test_predictions_df_3['prob_fr'] = test_fr_predictions

test_predictions_df = pd.merge(test_predictions_df_1, test_predictions_df_2, on='review_id', how='left')
test_predictions_df = pd.merge(test_predictions_df, test_predictions_df_3, on='review_id', how='left')

test_predictions_df['user_suggestion_prob'] = (test_predictions_df['prob_english'] + test_predictions_df['prob_es'] + test_predictions_df['prob_fr'])/3
test_predictions_df['user_suggestion'] = list(map(lambda x: np.int(x > optimal_threshold), test_predictions_df['user_suggestion_prob'] ))

test_predictions_df['user_suggestion_english'] = list(map(lambda x: np.int(x > optimal_threshold), test_predictions_df['prob_english'] ))
test_predictions_df['user_suggestion_es'] = list(map(lambda x: np.int(x > optimal_threshold), test_predictions_df['prob_es'] ))
test_predictions_df['user_suggestion_fr'] = list(map(lambda x: np.int(x > optimal_threshold), test_predictions_df['prob_fr'] ))


test_predictions_df[['review_id','user_suggestion','user_suggestion_english','user_suggestion_es','user_suggestion_fr','user_suggestion_prob','prob_english','prob_es','prob_fr']].to_csv('test_pred_21_training_en-fr-es_testing_en-fr-es.csv', index=False)
