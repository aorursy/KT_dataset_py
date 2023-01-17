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
train = pd.read_csv("../input/jigsaw-toxic-comment-classification-cleaned-data/train_data.csv")

val = pd.read_csv("../input/jigsaw-toxic-comment-classification-cleaned-data/val_data.csv")

test = pd.read_csv("../input/jigsaw-toxic-comment-classification-cleaned-data/test_data.csv")

print(len(train), len(val), len(test))
dummy = train.cleaned_text.values[0]

test.cleaned_text[pd.isnull(test.cleaned_text)] = dummy

test[pd.isnull(test.cleaned_text)]
len(train[train.toxic == 1])
new_train = pd.DataFrame()

new_train = pd.concat((train[train.toxic == 1][:20000],train[train.toxic == 0][:30000]))
new_train
import tensorflow as tf

import transformers

model = 'roberta-large'

tokenizer = transformers.AutoTokenizer.from_pretrained(model)
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

new_train = new_train[pd.notnull(new_train.cleaned_text)]



train = train[pd.notnull(train.cleaned_text)]

train, validation = train_test_split(train, test_size = 0.2)
max_seq_length = 200



train_input_ids = [tokenizer.encode(i, max_length = max_seq_length , pad_to_max_length = True) for i in train.cleaned_text.values[::10]]

val_input_ids = [tokenizer.encode(i, max_length = max_seq_length , pad_to_max_length = True) for i in validation.cleaned_text.values[::10]]
def create_model(): 

    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,

                                           name="input_word_ids")

    bert_layer = transformers.TFAutoModel.from_pretrained( 'roberta-large')

    bert_outputs = bert_layer(input_word_ids)[0]

    pred = tf.keras.layers.Conv1D(128,2,padding='same')(bert_outputs)

    pred = tf.keras.layers.LeakyReLU()(pred)

    pred = tf.keras.layers.Dropout(0.3)(pred)

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
model.fit(np.array(train_input_ids[::10]),np.array(train.toxic.values[::100]),

          validation_data = (np.array(val_input_ids[::10]),np.array(validation.toxic.values[::100])),

          verbose = 1, epochs = 30, batch_size = 128)
model.fit(np.array(val_input_ids[::10]),np.array(validation.toxic.values[::100]), epochs = 10, verbose = 1)
test_input_ids = [tokenizer.encode(i, max_length = max_seq_length , pad_to_max_length = True) for i in test.cleaned_text.values]
preds = model.predict(test_input_ids)
np.where(preds > 0.5)
preds = [max(preds[i]) for i in range(len(preds))]
evaluation = test.id.copy().to_frame()

evaluation['toxic'] = np.round(preds)

evaluation
evaluation.to_csv("submission.csv", index=False)