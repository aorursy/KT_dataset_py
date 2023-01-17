import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import re

import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from kaggle_datasets import KaggleDatasets

import transformers

from tqdm.notebook import tqdm

from tokenizers import BertWordPieceTokenizer
def clean_text(text):

    text = str(text)

    text = re.sub(r'[0-9"]', '', text) # number

    text = re.sub(r'#[\S]+\b', '', text) # hash

    text = re.sub(r'@[\S]+\b', '', text) # mention

    text = re.sub(r'https?\S+', '', text) # link

    text = re.sub(r'\s+', ' ', text) # multiple white spaces

#     text = re.sub(r'\W+', ' ', text) # non-alphanumeric

    return text.strip()
def text_process(text):

    ws = text.split(' ')

    if(len(ws)>160):

        text = ' '.join(ws[:160]) + ' ' + ' '.join(ws[-32:])

    return text
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):

 

    tokenizer.enable_truncation(max_length=maxlen)

    tokenizer.enable_padding(max_length=maxlen)

    all_ids = []

    

    for i in tqdm(range(0, len(texts), chunk_size)):

        text_chunk = texts[i:i+chunk_size].tolist()

        encs = tokenizer.encode_batch(text_chunk)

        all_ids.extend([enc.ids for enc in encs])

    

    return np.array(all_ids)
# First load the real tokenizer

tokenizer = transformers.BertTokenizer.from_pretrained('bert-large-uncased')



# Save the loaded tokenizer locally

save_path = '/kaggle/working/bert_base_uncased/'

if not os.path.exists(save_path):

    os.makedirs(save_path)

tokenizer.save_pretrained(save_path)



# Reload it with the huggingface tokenizers library

fast_tokenizer = BertWordPieceTokenizer('bert_base_uncased/vocab.txt', lowercase=True)

fast_tokenizer
# Detect hardware, return appropriate distribution strategy

try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
# Configuration

AUTO = tf.data.experimental.AUTOTUNE

EPOCHS_1 = 20

EPOCHS_2 = 2

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

MAX_LEN = 192

SHUFFLE = 2048

VERBOSE = 1
train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")

train2.toxic = train2.toxic.round().astype(int)



valid = pd.read_csv('/kaggle/input/val-en-df/validation_en.csv')



test1 = pd.read_csv('/kaggle/input/test-en-df/test_en.csv')

test2 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv')



sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

sub1 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

sub2 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
# oversample====

train = pd.concat([

    train1[['comment_text', 'toxic']],

    train2[['comment_text', 'toxic']].query('toxic==1'),

    train2[['comment_text', 'toxic']].query('toxic==0').sample(n=100000, random_state=1234)

    ])

# shufle it just to make sure

train = train.sample(frac=1, random_state=1234)
train['comment_text'] = train.apply(lambda x: clean_text(x['comment_text']), axis=1)

valid['comment_text_en'] = valid.apply(lambda x: clean_text(x['comment_text_en']), axis=1)

test1['content'] = test1.apply(lambda x: clean_text(x['content_en']), axis=1)

test2['content'] = test2.apply(lambda x: clean_text(x['translated']), axis=1)



train['comment_text'] = train['comment_text'].apply(lambda x: text_process(x))

valid['comment_text_en'] = valid['comment_text_en'].apply(lambda x: text_process(x))

test1['content'] = test1['content_en'].apply(lambda x: text_process(x))

test2['content'] = test2['translated'].apply(lambda x: text_process(x))
x_train = fast_encode(train1.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)

x_valid = fast_encode(valid.comment_text_en.astype(str), fast_tokenizer, maxlen=MAX_LEN)

x_test1 = fast_encode(test1.content_en.astype(str), fast_tokenizer, maxlen=MAX_LEN)

x_test2 = fast_encode(test2.translated.astype(str), fast_tokenizer, maxlen=MAX_LEN)



y_train = train1.toxic.values

y_valid = valid.toxic.values
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(SHUFFLE)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid, y_valid))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)



test1_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test1)

    .batch(BATCH_SIZE)

)



test2_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test2)

    .batch(BATCH_SIZE)

)
lrs = ReduceLROnPlateau(monitor='val_accuracy', mode ='max', factor = 0.7, min_lr= 1e-7, verbose = 1, patience = 2)

es1 = EarlyStopping(monitor='val_accuracy', mode='max', verbose = 1, patience = 5, restore_best_weights=True)

es2 = EarlyStopping(monitor='accuracy', mode='max', verbose = 1, patience = 1, restore_best_weights=True)

callbacks_list1 = [lrs,es1]

callbacks_list2 = [lrs,es2]
def build_model(transformer, max_len=512):



    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    x = tf.keras.layers.Dropout(0.4)(cls_token)

    out = Dense(1, activation='sigmoid')(cls_token)

    

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy'])

    

    return model
%%time

with strategy.scope():

    transformer_layer = transformers.TFBertModel.from_pretrained('bert-large-uncased')

    model = build_model(transformer_layer, max_len=MAX_LEN)

model.summary()
n_train_steps = train.shape[0] // (BATCH_SIZE*16)

model_history_1 = model.fit(

    train_dataset,

    steps_per_epoch=n_train_steps,

    validation_data=valid_dataset,

    epochs=EPOCHS_1,

    callbacks=callbacks_list1,

    verbose=VERBOSE

 )
n_valid_steps = valid.shape[0] // (BATCH_SIZE)

model_history_2 = model.fit(

    valid_dataset.repeat(),

    steps_per_epoch=n_valid_steps,

    epochs=EPOCHS_2,

    callbacks=callbacks_list2,

    verbose=VERBOSE

)
eng1 = model.predict(test1_dataset, verbose=1)

eng2 = model.predict(test2_dataset, verbose=1)
sub['toxic'] = eng1*0.5 + eng2*0.5

sub.toxic.hist(bins=100, log=False, alpha=1)
sub.to_csv('submission.csv', index=False)