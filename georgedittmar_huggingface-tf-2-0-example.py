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
!pip install transformers
from transformers import TFBertForSequenceClassification, BertTokenizer, TFDistilBertForSequenceClassification, DistilBertTokenizer, glue_convert_examples_to_features

import tensorflow as tf

import pandas as pd



tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

#model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
# get the data and read into pandas frames from the csv

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train
train_text_df = train[['text', 'target']]

train_text_df = train_text_df.dropna()

train_X = train_text_df['text']

train_y = train_text_df['target'].to_numpy()
train_x = tokenizer.batch_encode_plus(train_X, pad_to_max_length=True, return_tensors="tf")
train_x
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bce = tf.keras.losses.BinaryCrossentropy()

metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')



model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

model.fit(x=train_x['input_ids'], y=train_y, epochs=15, batch_size=128, verbose=1)
test_x = test['text'].to_numpy()
test_x = tokenizer.batch_encode_plus(test_x, pad_to_max_length=True, return_tensors="tf")
predictions = model.predict(test_x['input_ids'])
predictions_label = [ np.argmax(x) for x in predictions[0]]
submission = pd.DataFrame({'id': test['id'], 'target': predictions_label})

submission['target'] = submission['target'].astype('int')

submission.to_csv('submission.csv', index=False)