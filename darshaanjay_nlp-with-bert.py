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
import tensorflow as tf
tf.__version__
import pandas as pd

import os
val_data = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

test_data = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

train_data = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
train_data.shape, test_data.shape, val_data.shape
test_data.head()
import re
def clean(text):

    # repalce na values

    text = text.fillna("fillna").str.lower()

    #replace newline characters with space

    text = text.map(lambda x: re.sub('\\n',' ', str(x)))

    # remove extra characters

    text = text.map(lambda x: re.sub('\[\[User.*', '', str(x)))

    text = text.map(lambda x: re.sub("\(http://.*?\s\(http://.*\)",'',str(x)))

    return text
val_data["comment_text"] = clean(val_data["comment_text"])

test_data["content"] = clean(test_data["content"])

train_data["comment_text"] = clean(train_data["comment_text"])
import transformers
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
import numpy

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
train_comments
import gc

gc.collect()
MAX_SEQ_LENGTH = 500
train_feature_ids, train_feature_masks = create_bert_input(tokenizer, train_comments, max_seq_len=MAX_SEQ_LENGTH)
val_feature_ids, val_feature_masks = create_bert_input(tokenizer, val_comments, max_seq_len=MAX_SEQ_LENGTH)
train_feature_ids.shape, train_feature_masks.shape, y_train.shape
val_feature_ids.shape, val_feature_masks.shape, y_valid.shape
## TPU configuration

from kaggle_datasets import KaggleDatasets



tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)

strategy = tf.distribute.experimental.TPUStrategy(tpu)
GCS_DS_PATH = KaggleDatasets().get_gcs_path('jigsaw-multilingual-toxic-comment-classification')



EPOCHS = 2

BATCH_SIZE = 32 * strategy.num_replicas_in_sync
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
def get_training_model():

    inp_id = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int64, name="bert_input_ids")

    inp_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int64, name="bert_input_masks")

    inputs = [inp_id, inp_mask]

    

    hidden_state = transformers.TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased')(inputs)[0]

    pooled_output = hidden_state[:, 0]

    dense1 = tf.keras.layers.Dense(128, activation='relu')(pooled_output)

    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense1)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=2e-5, 

                                            epsilon=1e-08), 

                loss='binary_crossentropy', metrics=['accuracy'])

    return model
# Authorize wandb

import wandb



wandb.login()

from wandb.keras import WandbCallback



# Initialize wandb

wandb.init(project="jigsaw-toxic", id="distilbert-tpu-kaggle-weighted")
# Create 32 random indices from the English only test comments

RANDOM_INDICES = np.random.choice(test_comments.shape[0], 32)
!pip install -q googletrans
# Demo examples of translations

from googletrans import Translator



sample_comment = test_comments[48649]

print(sample_comment)

translated_comment = Translator().translate(sample_comment)
print(translated_comment.text)
# Create a sample prediction logger

# A custom callback to view predictions on the above samples in real-time

class TextLogger(tf.keras.callbacks.Callback):

    def __init__(self):

        super(TextLogger, self).__init__()



    def on_epoch_end(self, logs, epoch):

        samples = []

        for index in RANDOM_INDICES:

            # Grab the comment and translate it

            comment = test_comments[index]

            translated_comment = Translator().translate(comment).text

            # Create BERT features

            comment_feature_ids, comment_features_masks = create_bert_input(tokenizer,  

                                    comment, max_seq_len=MAX_SEQ_LENGTH)

            # Employ the model to get the prediction and parse it

            predicted_label = self.model.predict([comment_feature_ids, comment_features_masks])

            predicted_label = np.argmax(predicted_label[0])

            if predicted_label==0: predicted_label="Non-Toxic"

            else: predicted_label="Toxic"

            

            sample = [comment, translated_comment, predicted_label]

            

            samples.append(sample)

        wandb.log({"text": wandb.Table(data=samples, 

                                       columns=["Comment", "Translated Comment", "Predicted Label"])})
gc.collect()
# Account for the class imbalance

from sklearn.utils import class_weight



class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(y_train),

                                                 y_train)

class_weights
# Train the model

import time



start = time.time()



# Compile the model with TPU Strategy

with strategy.scope():

    model = get_training_model()

    

model.fit(train_ds, 

          steps_per_epoch=train_data.shape[0] // BATCH_SIZE,

          validation_data=valid_ds,

          validation_steps=val_data.shape[0] // BATCH_SIZE,

          epochs=EPOCHS,

          class_weight=class_weights,

          callbacks=[WandbCallback(), TextLogger()],

          verbose=1)

end = time.time() - start

print("Time taken ",end)

wandb.log({"training_time":end})
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
sub.head()
test_feature_ids, test_feature_masks = create_bert_input(tokenizer, test_comments, max_seq_len=MAX_SEQ_LENGTH)
#sub['toxic'] = model.predict(test_ds, verbose=1)

sub['toxic'] = model.predict([test_feature_ids, test_feature_masks], verbose=1)

#sub.to_csv('submission.csv', index=False)
sub.head()
sub.to_csv('submission.csv', index=False)