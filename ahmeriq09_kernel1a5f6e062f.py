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
print(tf.__version__)

tf.keras.__version__
import pandas as pd
import os

data_path = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/'

TEST_PATH = os.path.join(data_path, "test.csv")
VAL_PATH = os.path.join(data_path, "validation.csv")
TRAIN_PATH = os.path.join(data_path, "jigsaw-toxic-comment-train.csv")

val_data = pd.read_csv(VAL_PATH)
test_data = pd.read_csv(TEST_PATH)
train_data = pd.read_csv(TRAIN_PATH)

print(train_data.shape)

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
#Clean Text
import re

def clean(text):
    text = text.fillna("fillna").str.lower()
    #replace newline characters with space
    text = text.map(lambda x: re.sub('\\n',' ', str(x)))
    text = text.map(lambda x: re.sub('\[\[User.*', '', str(x)))
    text = text.map(lambda x: re.sub("\(http://.*?\s\(http://.*\)",'',str(x)))
    return text

val_data["comment_text"] = clean(val_data["comment_text"])
test_data["content"] = clean(test_data["content"])
train_data["comment_text"] = clean(train_data["comment_text"])   
# Load DistilBERT tokenizer
import transformers

tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
import numpy as np
import tqdm

def create_bert_input(tokenizer, docs, max_seq_len):
    all_input_ids, all_mask_ids = [], []
    for doc in tqdm.tqdm(docs, desc="Converting docs to features"):
        tokens = tokenizer.tokenize(doc)
        if len(tokens) > max_seq_len - 2:
            tokens = tokens[0: (max_seq_len-2)]
        tokens = ['[CLS]']+tokens+['[SEP]']
        ids = tokenizer.convert_tokens_to_ids(tokens)
        masks = [1]*len(ids)
        while len(ids) < max_seq_len:
            ids.append(0)
            masks.append(0)
        all_input_ids.append(ids)
        all_mask_ids.append(masks)
    
    encoded = np.array([all_input_ids, all_mask_ids])
    return encoded
train_comments = train_data.comment_text.astype(str).values
val_comments = val_data.comment_text.astype(str).values
test_comments = test_data.content.astype(str).values

y_valid = val_data.toxic.values
y_train = train_data.toxic.values
import gc
gc.collect()
#Encode the comments in train_set
MAX_SEQ_LENGTH = 500

train_feature_ids, train_feature_masks = create_bert_input(tokenizer, train_comments, max_seq_len=MAX_SEQ_LENGTH)

val_feature_ids, val_feature_masks = create_bert_input(tokenizer, val_comments, max_seq_len=MAX_SEQ_LENGTH)

test_feature_ids, test_feature_masks = create_bert_input(tokenizer, test_comments, max_seq_len=MAX_SEQ_LENGTH)
#Configure TPU
from kaggle_datasets import KaggleDatasets

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)

GCS_DS_PATH = KaggleDatasets().get_gcs_path('jigsaw-multilingual-toxic-comment-classification')

EPOCHS = 1
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
# Create TensorFlow datasets for better performance
train_ds = (
    tf.data.Dataset
    .from_tensor_slices(((train_feature_ids, train_feature_masks), y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.experimental.AUTOTUNE)
)
    
valid_ds = (
    tf.data.Dataset
    .from_tensor_slices(((val_feature_ids, val_feature_masks), y_valid))
    .repeat()
    .batch(BATCH_SIZE)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

test_ds = (
    tf.data.Dataset
    .from_tensor_slices((test_feature_ids, test_feature_masks))
    .repeat()
    .batch(BATCH_SIZE)
    .prefetch(tf.data.experimental.AUTOTUNE)
)
#Create training ready model
def get_training_model():
    # Build the model
    print('Build model...')
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units = 32, activation = 'tanh', return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units = 32, activation = 'tanh', return_sequences=True))
    model.add(tf.keras.layers.LSTM(units = 32, activation = 'tanh', return_sequences=True))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model
# Train the model
import time

start = time.time()

# Compile the model with TPU Strategy
with strategy.scope():
    model = get_training_model()
    
model.fit(train_ds, steps_per_epoch=train_data.shape[0] // BATCH_SIZE, validation_data=valid_ds,validation_steps=val_data.shape[0] // BATCH_SIZE, epochs=EPOCHS, verbose=1)
end = time.time() - start
print("Time taken ",end)
