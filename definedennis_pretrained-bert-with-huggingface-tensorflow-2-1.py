!pip install --upgrade transformers
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from tqdm.notebook import tqdm



import tensorflow as tf

from tensorflow import keras

import tensorflow.keras.backend as K

from tensorflow.keras import layers

from tensorflow.keras.utils import plot_model

from transformers import (

    BertTokenizer,

    TFBertForSequenceClassification,

    TFBertModel,

    BertConfig,

)

tf.__version__
MAX_SEQUENCE_LENGTH = 255

PRETRAINED_MODEL_NAME = 'bert-base-uncased'

BATCH_SIZE = 32
df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df.head()
df['target'].value_counts()
df.isnull().sum()
data = df['text'].values

targets = df['target'].values
def create_model():

    bert_model = TFBertModel.from_pretrained(PRETRAINED_MODEL_NAME)

    

    input_ids = layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_ids')

    token_type_ids = layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='token_type_ids')

    attention_mask = layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='attention_mask')

    

    # Use pooled_output(hidden states of [CLS]) as sentence level embedding

    pooled_output = bert_model({'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids})[1]

    x = layers.Dropout(rate=0.1)(pooled_output)

    x = layers.Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(inputs={'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}, outputs=x)

    return model
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

model = create_model()
model.summary()
plot_model(model, to_file='model.png', expand_nested=True, show_shapes=True)
opt = tf.keras.optimizers.Adam(learning_rate=3e-5)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
X_train, X_val, y_train, y_val = train_test_split(data, targets, test_size=0.33, random_state=42, stratify=targets)
X_train = tokenizer.batch_encode_plus(X_train, max_length=MAX_SEQUENCE_LENGTH, pad_to_max_length=True, return_tensors='tf')

X_val = tokenizer.batch_encode_plus(X_val, max_length=MAX_SEQUENCE_LENGTH, pad_to_max_length=True, return_tensors='tf')
history = model.fit(

    x=X_train,

    y=y_train,

    validation_data=(X_val, y_val),

    epochs=3,

    batch_size=BATCH_SIZE

)
y_pred = model.predict(x=X_val, batch_size=BATCH_SIZE)
y_pred_bin = (y_pred > 0.5).astype(int).reshape(-1)
print(classification_report(y_val, y_pred_bin))
df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
df_test.head()
X_test = df_test['text'].values

X_test = tokenizer.batch_encode_plus(X_test, max_length=MAX_SEQUENCE_LENGTH, pad_to_max_length=True, return_tensors='tf')

y_pred = model.predict(x=X_test, batch_size=BATCH_SIZE)
y_pred_bin = (y_pred > 0.5).astype(int).reshape(-1)

df_test['target'] = y_pred_bin

df_test[['id', 'target']].to_csv('submission.csv', index=False)