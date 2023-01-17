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
import tensorflow as tf

print("TF version: ", tf.__version__)
train = pd.read_csv("../input/jigsaw-toxic-comment-classification-cleaned-data/train_data.csv")
train
import transformers

from tokenizers import BertWordPieceTokenizer

from transformers import TFBertModel



tokenizer = transformers.AutoTokenizer.from_pretrained('roberta-large')

# Save the loaded tokenizer locally
max_seq_length = 200
from sklearn.model_selection import train_test_split



train, val = train_test_split(train, test_size = 0.1)
train_input_ids = [tokenizer.encode(str(i), max_length = max_seq_length , pad_to_max_length = True) for i in train.cleaned_text.values]

val_input_ids = [tokenizer.encode(str(i), max_length = max_seq_length , pad_to_max_length = True) for i in val.cleaned_text.values]
def create_model(): 

    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,

                                           name="input_word_ids")

    bert_layer = TFBertModel.from_pretrained('roberta-large')

    #bert_layer = TFAutoModel.from_pretrained('jplu/tf-xlm-roberta-large')

    bert_outputs = bert_layer(input_word_ids)[0]

    pred = tf.keras.layers.Dropout(0.2)(bert_outputs)

    pred = tf.keras.layers.Conv1D(128,2,padding='same')(pred)

    pred = tf.keras.layers.Dropout(0.3)(pred)

    pred = tf.keras.layers.LeakyReLU()(pred)

    pred = tf.keras.layers.Conv1D(64,2,padding='same')(pred)

    pred = tf.keras.layers.Dense(256, activation='relu')(pred)

    pred = tf.keras.layers.Dense(1, activation='sigmoid')(pred)



    model = tf.keras.models.Model(inputs=input_word_ids, outputs=pred)

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(

    learning_rate=0.00001), metrics=['accuracy'])

    return model
use_tpu = True

if use_tpu:

    # Create distribution strategy

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)



    # Create model

    with strategy.scope():

        model = create_model()

else:

    model = create_model()

    

model.summary()
train_x = tf.constant(train_input_ids)

train_y = tf.constant(train.toxic.values)

train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))

val_x = tf.constant(val_input_ids)

val_y = tf.constant(val.toxic.values)

val_data = tf.data.Dataset.from_tensor_slices((val_x, val_y))
batch_size = 256



model.fit(train_data.batch(batch_size),

          validation_data = val_data.batch(batch_size),

          verbose = 1, epochs = 2, batch_size = batch_size)
preds = np.round(model.predict(np.array(val_input_ids[::100])))
preds = [np.max(i) for i in preds]

preds
yes = 0

total = 0



for i,j in zip(preds, val.toxic.values[::100]):

    if i==j: yes += 1

    total += 1

print(yes/total)
test = pd.read_csv("../input/jigsaw-toxic-comment-classification-cleaned-data/test_data.csv")
test.head()
dummy = train.cleaned_text.values[0]

test.cleaned_text[pd.isnull(test.cleaned_text)] = dummy
test_input_ids = [tokenizer.encode(str(i), max_length = max_seq_length , pad_to_max_length = True) for i in test.cleaned_text.values]
len(test)
np.shape(test_input_ids)
preds = [np.max(i) for i in model.predict(test_input_ids)]
preds[:10]
evaluation = test.id.copy().to_frame()

evaluation['toxic'] = np.round(preds)

evaluation
evaluation.to_csv("submission.csv", index=False)