USE_TPU = True

import os



import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

from kaggle_datasets import KaggleDatasets

import transformers

from transformers import TFAutoModel, AutoTokenizer

from tqdm.notebook import tqdm

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
def regular_encode(texts, tokenizer, maxlen=512):

    enc_di = tokenizer.batch_encode_plus(

        texts, 

        return_attention_masks=False, 

        return_token_type_ids=False,

        pad_to_max_length=True,

        max_length=maxlen

    )

    

    return np.array(enc_di['input_ids'])



def build_model(transformer, max_len=512):

    """

    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras

    """

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(cls_token)

    

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
tf.keras.metrics.Accuracy.__name__
if USE_TPU:

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

    BATCH_SIZE = 8 * strategy.num_replicas_in_sync

    

else:

    BATCH_SIZE = 8 * 8
AUTO = tf.data.experimental.AUTOTUNE



# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path()



# Configuration

EPOCHS = 2

MAX_LEN = 192

MODEL = 'jplu/tf-xlm-roberta-large'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")

train2.toxic = train2.toxic.round().astype(int)



valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv').sample(n=200, random_state=0)

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv').sample(n=200, random_state=0)

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
# Combine train1 with a subset of train2

train = pd.concat([

    train2[['comment_text', 'toxic']].query('toxic==1').sample(n=400, random_state=0),

    train2[['comment_text', 'toxic']].query('toxic==0').sample(n=400, random_state=0)

])
%%time 



x_train = regular_encode(train.comment_text.values, tokenizer, maxlen=MAX_LEN)

x_valid = regular_encode(valid.comment_text.values, tokenizer, maxlen=MAX_LEN)

x_test = regular_encode(test.content.values, tokenizer, maxlen=MAX_LEN)



y_train = train.toxic.values

y_valid = valid.toxic.values
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(2048)

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



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(BATCH_SIZE)

)
%%time

if USE_TPU:

    with strategy.scope():

        transformer_layer = TFAutoModel.from_pretrained(MODEL)

        model = build_model(transformer_layer, max_len=MAX_LEN)

        

else:

    transformer_layer = TFAutoModel.from_pretrained(MODEL)

    model = build_model(transformer_layer, max_len=MAX_LEN)

    

model.summary()
# Code from: https://stackoverflow.com/questions/47731935/using-multiple-validation-sets-with-keras

class AdditionalValidationSets(tf.keras.callbacks.Callback):

    def __init__(self, validation_sets, verbose=0, batch_size=None):

        """

        :param validation_sets:

        a list of 3-tuples (validation_data, validation_targets, validation_set_name)

        or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)

        :param verbose:

        verbosity mode, 1 or 0

        :param batch_size:

        batch size to be used when evaluating on the additional datasets

        """

        super(AdditionalValidationSets, self).__init__()

        self.validation_sets = validation_sets

        for validation_set in self.validation_sets:

            if len(validation_set) not in [2, 3]:

                raise ValueError()

        self.epoch = []

        self.history = {}

        self.verbose = verbose

        self.batch_size = batch_size



    def on_train_begin(self, logs=None):

        self.epoch = []

        self.history = {}



    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        self.epoch.append(epoch)



        # record the same values as History() as well

        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)



        # evaluate on the additional validation sets

        for validation_set in self.validation_sets:

            if len(validation_set) == 3:

                validation_data, validation_targets, validation_set_name = validation_set

                sample_weights = None

            elif len(validation_set) == 4:

                validation_data, validation_targets, sample_weights, validation_set_name = validation_set

            else:

                raise ValueError()



            results = self.model.evaluate(x=validation_data,

                                          y=validation_targets,

                                          verbose=self.verbose,

                                          sample_weight=sample_weights,

                                          batch_size=self.batch_size)



            for i, result in enumerate(results):

                if i == 0:

                    valuename = validation_set_name + '_loss'

                else:

                    valuename = validation_set_name + '_' + str(self.model.metrics[i-1].name)

                self.history.setdefault(valuename, []).append(result)

                

                print(" {}:{}".format(valuename, result), end='\t')

                

            print('\n')
n_steps = x_valid.shape[0] // BATCH_SIZE

print(n_steps)


train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=EPOCHS,

    callbacks=[AdditionalValidationSets([(x_valid, y_valid, 'val_additional')])]

)
sub['toxic'] = model.predict(test_dataset, verbose=1)

sub.to_csv('submission.csv', index=False)