# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint
import transformers



from tokenizers import BertWordPieceTokenizer



from tqdm.notebook import tqdm





from kaggle_datasets import KaggleDatasets
def regular_encode(texts, tokenizer, maxlen=512):

    enc_di = tokenizer.batch_encode_plus(

        texts, 

        return_attention_masks=False, 

        return_token_type_ids=False,

        pad_to_max_length=True,

        max_length=maxlen

    )

    

    return np.array(enc_di['input_ids'])
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
AUTO = tf.data.experimental.AUTOTUNE



# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path()
EPOCHS = 2

BATCH_SIZE = 32 * strategy.num_replicas_in_sync



MAX_LEN = 192
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

# Save the loaded tokenizer locally

tokenizer.save_pretrained('.')

# Reload it with the huggingface tokenizers library

# fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)

# fast_tokenizer
DATA_PATH = "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/"

train1 = pd.read_csv(os.path.join(DATA_PATH, "jigsaw-toxic-comment-train.csv"))

train2 = pd.read_csv(os.path.join(DATA_PATH, "jigsaw-unintended-bias-train.csv"))

train2.toxic = train2.toxic.round().astype(int)



valid = pd.read_csv(os.path.join(DATA_PATH, 'validation.csv'))

test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

sub = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
train = pd.concat([

    train1[['comment_text', 'toxic']],

    train2[['comment_text', 'toxic']].query('toxic==1'),

    train2[['comment_text', 'toxic']].query('toxic==0').sample(n=150000, random_state=0)])
train.head(5)
valid.head(5)
test.head(5)
x_train = regular_encode(train.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)

x_valid = regular_encode(valid.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)

x_test = regular_encode(test.content.astype(str), tokenizer, maxlen=MAX_LEN)



y_train = train.toxic.values

y_valid = valid.toxic.values
train_dataset = (tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(AUTO))



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
def loss(y_true, y_pred):

    return tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False,label_smoothing = 0.099)

    
def build_model(transformer, max_len=512):

    """

    Model initalization

    Source: https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras

    """

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)

    

    x1 = tf.keras.layers.Dropout(0.1)(sequence_output[0]) 

    x1 = tf.keras.layers.Conv1D(128, 2,padding='same')(x1)

    x1 = tf.keras.layers.LeakyReLU()(x1)

    #x1 = tf.keras.layers.ReLU()(x1)

    x1 = tf.keras.layers.Conv1D(64, 2,padding='same')(x1)

    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv1D(32, 2,padding='same')(x1)

    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Flatten()(x1)

    x1 = tf.keras.layers.Dense(32,activation = 'relu')(x1)

    out = tf.keras.layers.Dense(1,activation='sigmoid')(x1)

    

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=1e-5), loss=loss, metrics=['accuracy'])

    

    return model
%%time



with strategy.scope():

    transformer_layer = (

        transformers.TFDistilBertModel

        .from_pretrained('distilbert-base-multilingual-cased')

    )

    model = build_model(transformer_layer, max_len=MAX_LEN)

model.summary()
n_steps = x_train.shape[0] // BATCH_SIZE

train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=5)
import matplotlib.pyplot as plt 



plt.plot(train_history.history['accuracy'])

plt.plot(train_history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
n_steps = x_valid.shape[0] // BATCH_SIZE

train_history_2 = model.fit(

    valid_dataset.repeat(),

    steps_per_epoch=n_steps,

    epochs=10)
plt.plot(train_history_2.history['accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train'], loc='upper left')

plt.show()

pred = model.predict(test_dataset, verbose=1)

# sub.to_csv('submission.csv', index=False)
sub['toxic'] = pred

sub.to_csv('submission.csv', index=False)
sub.head(5)
model.save_weights('distilbert-base-multilingual-cased.h5')
sub.tail(10)