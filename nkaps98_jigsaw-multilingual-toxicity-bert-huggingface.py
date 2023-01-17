# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import transformers

from tokenizers import BertWordPieceTokenizer

import tensorflow as tf

from tqdm import tqdm

from tensorflow.keras.optimizers import Adam
 #Detect hardware, return appropriate distribution strategy

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
#IMP DATA FOR CONFIG



AUTO = tf.data.experimental.AUTOTUNE





# Configuration

EPOCHS = 3

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

MAX_LEN = 192
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):

    '''

    Function for fast encoding

    '''

    all_ids = []

    for i in tqdm(range(0, len(texts), chunk_size)):

        text_chunk = list(texts[i:chunk_size+i])

        encs = tokenizer.batch_encode_plus(text_chunk, max_length=maxlen, pad_to_max_length = True)

        all_ids.extend(encs['input_ids'])

        

    return np.array(all_ids)
train = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')

validation = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

test = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/test.csv')
# Refer the HuggingFace Documention

fast_tokenizer = transformers.DistilBertTokenizerFast.from_pretrained('distilbert-base-multilingual-cased')
x_train = fast_encode(train.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)

x_valid = fast_encode(validation.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)

x_test = fast_encode(test.content.astype(str), fast_tokenizer, maxlen=MAX_LEN)
y_train = train.toxic.values

y_valid = validation.toxic.values
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train,y_train))

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid,y_valid))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(BATCH_SIZE)

)
def build_model(transformer, maxlen=512):

    input_word_ids = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32, name='input_word_ids')

    sequence_output = transformer(input_word_ids)[0]

    

    clf_output = sequence_output[:,0,:]

    out = tf.keras.layers.Dense(1, activation='sigmoid')(clf_output)

    

    model = tf.keras.models.Model(inputs = input_word_ids, outputs=out)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    return model
with strategy.scope():

    transformer_layer = (

        transformers.TFDistilBertModel

        .from_pretrained('distilbert-base-multilingual-cased')

    )

    model = build_model(transformer_layer, maxlen=MAX_LEN)

    

model.summary()
n_steps = x_train.shape[0]//BATCH_SIZE

train_history = model.fit(

    train_dataset,

    steps_per_epoch = n_steps,

    validation_data = valid_dataset,

    epochs = EPOCHS

)
n_steps_valid = x_valid.shape[0]//BATCH_SIZE

valid_history = model.fit(

    valid_dataset.repeat(),

    steps_per_epoch = n_steps_valid,

    epochs = EPOCHS*2

)
sub = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

sub['toxic'] = model.predict(test_dataset, verbose=1)

sub.to_csv('submission.csv', index=False )