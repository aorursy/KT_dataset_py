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
!wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

!wget -q https://raw.githubusercontent.com/google-research/bert/master/modeling.py 

!wget -q https://raw.githubusercontent.com/google-research/bert/master/optimization.py 

!wget -q https://raw.githubusercontent.com/google-research/bert/master/run_classifier.py 

!wget -q https://raw.githubusercontent.com/google-research/bert/master/tokenization.py 
import os

import numpy as np

import pandas as pd

import datetime

import sys

import zipfile

import modeling

import optimization

import run_classifier

import tokenization

from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
folder = 'model_folder'

with zipfile.ZipFile("uncased_L-12_H-768_A-12.zip","r") as zip_ref:

    zip_ref.extractall(folder)
BERT_MODEL = 'uncased_L-12_H-768_A-12'

BERT_PRETRAINED_DIR = f'{folder}/uncased_L-12_H-768_A-12'

OUTPUT_DIR = f'{folder}/outputs'

print(f'>> Model output directory: {OUTPUT_DIR}')

print(f'>>  BERT pretrained directory: {BERT_PRETRAINED_DIR}')
train_df = pd.read_excel('../input/Train_Data.xlsx')
train_df.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_df[['qid','question_text']], train_df['target'],test_size=0.33, random_state=42)
label_encoder = LabelEncoder().fit(pd.concat([y_train, y_test]))
def create_examples(lines, set_type, labels=None):

    # Generate data for the BERT model

    guid = f'{set_type}'

    examples = []

    if guid == 'train':

        for line, label in zip(lines, labels):

            text_a = line

            label = str(label)

            examples.append(

              run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

    else:

        for line in lines:

            text_a = line

            label = '0'

            examples.append(

              run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

    return examples
# Model Hyper Parameters

TRAIN_BATCH_SIZE = 32

EVAL_BATCH_SIZE = 8

LEARNING_RATE = 1e-5

NUM_TRAIN_EPOCHS = 3.0

WARMUP_PROPORTION = 0.1

MAX_SEQ_LENGTH = 128

# Model configs

SAVE_CHECKPOINTS_STEPS = 100000 #if you wish to finetune a model on a larger dataset, use larger interval

# each checpoint weights about 1,5gb

ITERATIONS_PER_LOOP = 100000

NUM_TPU_CORES = 8

VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')

CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')

INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')

DO_LOWER_CASE = BERT_MODEL.startswith('uncased')



label_list = [str(num) for num in range(8)]

tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)

train_examples = create_examples(X_train, 'train', labels=y_train)



tpu_cluster_resolver = None #Since training will happen on GPU, we won't need a cluster resolver

#TPUEstimator also supports training on CPU and GPU. You don't need to define a separate tf.estimator.Estimator.

run_config = tf.contrib.tpu.RunConfig(

    cluster=tpu_cluster_resolver,

    model_dir=OUTPUT_DIR,

    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,

    tpu_config=tf.contrib.tpu.TPUConfig(

        iterations_per_loop=ITERATIONS_PER_LOOP,

        num_shards=NUM_TPU_CORES,

        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))



num_train_steps = int(

    len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)

num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
model_fn = run_classifier.model_fn_builder(

    bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),

    num_labels=len(label_list),

    init_checkpoint=INIT_CHECKPOINT,

    learning_rate=LEARNING_RATE,

    num_train_steps=num_train_steps,

    num_warmup_steps=num_warmup_steps,

    use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available  

    use_one_hot_embeddings=True)



estimator = tf.contrib.tpu.TPUEstimator(

    use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available 

    model_fn=model_fn,

    config=run_config,

    train_batch_size=TRAIN_BATCH_SIZE,

    eval_batch_size=EVAL_BATCH_SIZE)
print('Please wait...')

train_features = run_classifier.convert_examples_to_features(

    train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

print('>> Started training at {} '.format(datetime.datetime.now()))

print('  Num examples = {}'.format(len(train_examples)))

print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))

tf.logging.info("  Num steps = %d", num_train_steps)

train_input_fn = run_classifier.input_fn_builder(

    features=train_features,

    seq_length=MAX_SEQ_LENGTH,

    is_training=True,

    drop_remainder=True)

estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

print('>> Finished training at {}'.format(datetime.datetime.now()))
def input_fn_builder(features, seq_length, is_training, drop_remainder):

    """Creates an `input_fn` closure to be passed to TPUEstimator."""



    all_input_ids = []

    all_input_mask = []

    all_segment_ids = []

    all_label_ids = []



    for feature in features:

        all_input_ids.append(feature.input_ids)

        all_input_mask.append(feature.input_mask)

        all_segment_ids.append(feature.segment_ids)

        all_label_ids.append(feature.label_id)



    def input_fn(params):

        """The actual input function."""

        print(params)

        batch_size = 500



        num_examples = len(features)



        d = tf.data.Dataset.from_tensor_slices({

            "input_ids":

                tf.constant(

                    all_input_ids, shape=[num_examples, seq_length],

                    dtype=tf.int32),

            "input_mask":

                tf.constant(

                    all_input_mask,

                    shape=[num_examples, seq_length],

                    dtype=tf.int32),

            "segment_ids":

                tf.constant(

                    all_segment_ids,

                    shape=[num_examples, seq_length],

                    dtype=tf.int32),

            "label_ids":

                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),

        })



        if is_training:

            d = d.repeat()

            d = d.shuffle(buffer_size=100)



        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)

        return d



    return input_fn



            
predict_examples = create_examples(X_test, 'test')



predict_features = run_classifier.convert_examples_to_features(

    predict_examples, label_list, MAX_SEQ_LENGTH, tokenizer)



predict_input_fn = input_fn_builder(

    features=predict_features,

    seq_length=MAX_SEQ_LENGTH,

    is_training=False,

    drop_remainder=False)



result = estimator.predict(input_fn=predict_input_fn)