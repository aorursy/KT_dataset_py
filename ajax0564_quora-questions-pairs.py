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
import numpy as np # linear algebra

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/quora-question-pairs/train.csv.zip')

test = pd.read_csv('/kaggle/input/quora-question-pairs/test.csv')

submission = pd.read_csv('/kaggle/input/quora-question-pairs/sample_submission.csv.zip')
train.head(5)
test.head(5)
plt.style.use('fivethirtyeight')

%matplotlib inline



f,ax=plt.subplots(1,figsize=(8,8))

train['is_duplicate'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax,shadow=True)
from kaggle_datasets import KaggleDatasets

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import transformers

from transformers import TFAutoModel, AutoTokenizer

from tqdm.notebook import tqdm
try:    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    print('Running on TPU ', tpu.master())

except ValueError:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
model_name = 'jplu/tf-xlm-roberta-base'

n_epochs = 6

max_len = 70



# Our batch size will depend on number of replicas

batch_size = 16 * strategy.num_replicas_in_sync
tokenizer = AutoTokenizer.from_pretrained(model_name)
train.isnull().sum()
test.isnull().sum()
train = train.dropna(axis = 0)
test = test.dropna(axis = 0)
train.isnull().sum()
train.shape
train = train.sample(frac = 0.45)

train.shape
test = test.sample(frac = 0.01)

test.shape
%%time



train_text = train[['question1', 'question2']].values.tolist()

test_text = test[['question1', 'question2']].values.tolist()



# Now, we use the tokenizer we loaded to encode the text

train_encoded = tokenizer.batch_encode_plus(

    train_text,

    pad_to_max_length=True,

    max_length=max_len)



test_encoded = tokenizer.batch_encode_plus(

    test_text,

    pad_to_max_length=True,

    max_length=max_len)
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(

    train_encoded['input_ids'], train.is_duplicate.values, 

    test_size=0.15, random_state=2020)



x_test = test_encoded['input_ids']
auto = tf.data.experimental.AUTOTUNE



train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(2048)

    .batch(batch_size)

    .prefetch(auto)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid, y_valid))

    .batch(batch_size)

    .cache()

    .prefetch(auto)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(batch_size)

)
with strategy.scope():

    # First load the transformer layer

    transformer_encoder = TFAutoModel.from_pretrained(model_name)



    # This will be the input tokens 

    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")



    # Now, we encode the text using the transformers we just loaded

    sequence_output = transformer_encoder(input_ids)[0]



    # Only extract the token used for classification, which is <s>

    cls_token = sequence_output[:, 0, :]



    # Finally, pass it through a 3-way softmax, since there's 3 possible laels

    out = Dense(2, activation='softmax')(cls_token)



    # It's time to build and compile the model

    model = Model(inputs=input_ids, outputs=out)

    model.compile(

        Adam(lr=1e-5), 

        loss='sparse_categorical_crossentropy', 

        metrics=['accuracy']

    )

model.summary()
n_steps = len(x_train) // batch_size



train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=n_epochs

)
test_preds = model.predict(test_dataset, verbose=1)

test['is_duplicate'] = test_preds.argmax(axis=1)
test.sample(n = 30)