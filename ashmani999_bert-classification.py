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
!pip install tqdm

!pip install transformers
import numpy as np

import pandas as pd

import tensorflow as tf

import transformers

import matplotlib.pyplot as plt



%matplotlib inline



# fix random seed for reproducibility

seed = 42

np.random.seed(seed)

tf.random.set_seed(seed)



print("TF Version: ", tf.__version__)

print("Eager mode: ", tf.executing_eagerly())

print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
dataset=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

df_test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
tweets = dataset['text'].values

label = dataset['target'].values



train_reviews = tweets[:7000]

val_reviews = tweets [7000:]

test_reviews=df_test.text.values









train_sentiments = label[:7000]

val_sentiments = label[7000:]

#test_sentiments = df_test['target'].values
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
import tqdm



def create_bert_input_features(tokenizer, docs, max_seq_length):

    

    all_ids, all_masks = [], []

    for doc in tqdm.tqdm(docs, desc="Converting docs to features"):

        tokens = tokenizer.tokenize(doc)

        if len(tokens) > max_seq_length-2:

            tokens = tokens[0 : (max_seq_length-2)]

        tokens = ['[CLS]'] + tokens + ['[SEP]']

        ids = tokenizer.convert_tokens_to_ids(tokens)

        masks = [1] * len(ids)

        # Zero-pad up to the sequence length.

        while len(ids) < max_seq_length:

            ids.append(0)

            masks.append(0)

        all_ids.append(ids)

        all_masks.append(masks)

    encoded = np.array([all_ids, all_masks])

    return encoded
MAX_SEQ_LENGTH = 500



inp_id = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name="bert_input_ids")

inp_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name="bert_input_masks")

inputs = [inp_id, inp_mask]



hidden_state = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')(inputs)[0]

pooled_output = hidden_state[:, 0]    

dense1 = tf.keras.layers.Dense(256, activation='relu')(pooled_output)

drop1 = tf.keras.layers.Dropout(0.25)(dense1)

dense2 = tf.keras.layers.Dense(256, activation='relu')(drop1)

drop2 = tf.keras.layers.Dropout(0.25)(dense2)

output = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)





model = tf.keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer=tf.optimizers.Adam(learning_rate=2e-5, 

                                           epsilon=1e-08), 

              loss='binary_crossentropy', metrics=['accuracy'])



model.summary()
train_features_ids, train_features_masks= create_bert_input_features(tokenizer,train_reviews,max_seq_length=MAX_SEQ_LENGTH)

val_features_ids, val_features_masks= create_bert_input_features(tokenizer, val_reviews,max_seq_length=MAX_SEQ_LENGTH)
print('Train Features:', train_features_ids.shape, train_features_masks.shape)

print('Val Features:', val_features_ids.shape, val_features_masks.shape)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 

                                      patience=1,

                                      restore_best_weights=True,

                                      verbose=1)

model.fit([train_features_ids, 

           train_features_masks], train_sentiments, 

          validation_data=([val_features_ids, 

                            val_features_masks], val_sentiments),

          epochs=5, 

          batch_size=9, 

          callbacks=[es],

          shuffle=True,

          verbose=1)
test_features_ids, test_features_masks= create_bert_input_features(tokenizer,test_reviews,max_seq_length=MAX_SEQ_LENGTH)

print('Test Features:', test_features_ids.shape, test_features_masks.shape)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



predictions = [1 if pr > 0.55 else 0 

                   for pr in model.predict([test_features_ids, 

                                            test_features_masks], verbose=0).ravel()]



ids=df_test['id']

df=pd.DataFrame(list(zip(ids,predictions)),

               columns=['id','target'])

df.to_csv('output2.csv',index=False)