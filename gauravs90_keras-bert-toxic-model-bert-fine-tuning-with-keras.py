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
!cp -r '../input/keras-bert-by-cyberzhg-github/keras_bert/keras_bert' '/kaggle/working'

!cp -r '../input/keras-bert-by-cyberzhg-github/keras_bert/keras_pos_embd' '/kaggle/working'

!cp -r '../input/keras-bert-by-cyberzhg-github/keras_bert/keras_embed_sim' '/kaggle/working'

!cp -r '../input/keras-bert-by-cyberzhg-github/keras_bert/keras_position_wise_feed_forward' '/kaggle/working'

!cp -r '../input/keras-bert-by-cyberzhg-github/keras_bert/keras_layer_normalization' '/kaggle/working'

!cp -r '../input/keras-bert-by-cyberzhg-github/keras_bert/keras_self_attention' '/kaggle/working'

!cp -r '../input/keras-bert-by-cyberzhg-github/keras_bert/keras_multi_head' '/kaggle/working'

!cp -r '../input/keras-bert-by-cyberzhg-github/keras_bert/keras_transformer' '/kaggle/working'
import tensorflow as tf

import keras as keras

import keras.backend as K

from keras.models import load_model



from keras_bert import load_trained_model_from_checkpoint, load_vocabulary

from keras_bert import Tokenizer

from keras_bert import AdamWarmup, calc_train_steps



import pandas as pd

import numpy as np

from tqdm import tqdm

import gc
SEQ_LEN = 64

BATCH_SIZE = 128

EPOCHS = 1

LR = 1e-4



pretrained_path = '../input/pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12'

config_path = os.path.join(pretrained_path, 'bert_config.json')

checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')

vocab_path = os.path.join(pretrained_path, 'vocab.txt')



DATA_COLUMN = 'comment_text'

LABEL_COLUMN = 'target'
token_dict = load_vocabulary(vocab_path)

tokenizer = Tokenizer(token_dict)
def convert_data(data_df):

    global tokenizer

    indices, targets = [], []

    for i in tqdm(range(len(data_df))):

        ids, segments = tokenizer.encode(data_df[DATA_COLUMN][i], max_len=SEQ_LEN)

        indices.append(ids)

        targets.append(data_df[LABEL_COLUMN][i])

    items = list(zip(indices, targets))

    np.random.shuffle(items)

    indices, targets = zip(*items)

    indices = np.array(indices)

    return [indices, np.zeros_like(indices)], np.array(targets)
def load_data(path):

    data_df = pd.read_csv(path, nrows=10000)

    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)

    data_x, data_y = convert_data(data_df)

    return data_x, data_y
train_x, train_y = load_data('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

gc.collect()
model = load_trained_model_from_checkpoint(

    config_path,

    checkpoint_path,

    training=True,

    trainable=True,

    seq_len=SEQ_LEN,

)
inputs = model.inputs[:2]

dense = model.layers[-3].output

outputs = keras.layers.Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),

                             name = 'real_output')(dense)



decay_steps, warmup_steps = calc_train_steps(

    train_y.shape[0],

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

)



model = keras.models.Model(inputs, outputs)

model.compile(

    AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=LR),

    loss='binary_crossentropy',

    metrics=['accuracy']

)
model.summary()
sess = K.get_session()

uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])

init_op = tf.variables_initializer(

    [v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables]

)

sess.run(init_op)
model.fit(

        train_x,

        train_y,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

    )
def convert_test(test_df):

    global tokenizer

    indices = []

    for i in tqdm(range(len(test_df))):

        ids, segments = tokenizer.encode(test_df[DATA_COLUMN][i], max_len=SEQ_LEN)

        indices.append(ids)

    indices = np.array(indices)

    return [indices, np.zeros_like(indices)]



def load_test(path):

    data_df = pd.read_csv(path, nrows=5000)

    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)

    data_x = convert_test(data_df)

    return data_x
test_x = load_test('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

gc.collect()
prediction = model.predict(test_x)

submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id', nrows=5000)

submission['prediction'] = prediction

submission.reset_index(drop=False, inplace=True)

submission.to_csv('submission.csv', index=False)