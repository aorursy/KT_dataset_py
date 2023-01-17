import tensorflow as tf
print(tf.__version__)
!ls /kaggle/input/jigsaw-multilingual-toxic-comment-classification/
# Load datasets
import pandas as pd
import os

DATA_PATH = "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/"

TEST_PATH = os.path.join(DATA_PATH, "test.csv")
VAL_PATH = os.path.join(DATA_PATH, "validation.csv")
TRAIN_PATH = os.path.join(DATA_PATH, "jigsaw-toxic-comment-train.csv")

val_data = pd.read_csv(VAL_PATH)
test_data = pd.read_csv(TEST_PATH)
train_data = pd.read_csv(TRAIN_PATH)
# Preview train set
train_data.sample(5)
val_data.sample(5)
test_data.sample(5)
# Remove usernames and links
import re

val = val_data
train = train_data

def clean(text):
    # fill the missing entries and convert them to lower case
    text = text.fillna("fillna").str.lower()
    # replace the newline characters with space 
    text = text.map(lambda x: re.sub('\\n',' ',str(x)))
    text = text.map(lambda x: re.sub("\[\[User.*",'',str(x)))
    # remove usernames and links
    text = text.map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
    text = text.map(lambda x: re.sub("\(http://.*?\s\(http://.*\)",'',str(x)))
    return text

val["comment_text"] = clean(val["comment_text"])
test_data["content"] = clean(test_data["content"])
train["comment_text"] = clean(train["comment_text"])
# Load DistilBERT tokenizer
import transformers

tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
import numpy as np
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
# Segregate the comments and their labels (not applicable for test set)
train_comments = train.comment_text.astype(str).values
val_comments = val_data.comment_text.astype(str).values
test_comments = test_data.content.astype(str).values

y_valid = val.toxic.values
y_train = train.toxic.values
import gc
gc.collect()
# Encode the comments
MAX_SEQ_LENGTH = 500

train_features_ids, train_features_masks = create_bert_input_features(tokenizer, train_comments, 
                                                                      max_seq_length=MAX_SEQ_LENGTH)
val_features_ids, val_features_masks = create_bert_input_features(tokenizer, val_comments, 
                                                                  max_seq_length=MAX_SEQ_LENGTH)
# test_features = create_bert_input_features(tokenizer, test_comments, 
#                                            max_seq_length=MAX_SEQ_LENGTH)
# Verify the shapes
print(train_features_ids.shape, train_features_masks.shape, y_train.shape)
print(val_features_ids.shape, val_features_masks.shape, y_valid.shape)
# Configure TPU
from kaggle_datasets import KaggleDatasets

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)

GCS_DS_PATH = KaggleDatasets().get_gcs_path('jigsaw-multilingual-toxic-comment-classification')

EPOCHS = 2
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
# Create TensorFlow datasets for better performance
train_ds = (
    tf.data.Dataset
    .from_tensor_slices(((train_features_ids, train_features_masks), y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.experimental.AUTOTUNE)
)
    
valid_ds = (
    tf.data.Dataset
    .from_tensor_slices(((val_features_ids, val_features_masks), y_valid))
    .repeat()
    .batch(BATCH_SIZE)
    .prefetch(tf.data.experimental.AUTOTUNE)
)
# Create utility function to get a training ready model on demand
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
RANDOM_INDICES
!pip install -q googletrans
# Demo examples of translations
from googletrans import Translator

sample_comment = test_comments[48649]
print("Original comment:", sample_comment)
translated_comment = Translator().translate(sample_comment)
print("\n")
print("Translated comment:", translated_comment.text)
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
            comment_feature_ids, comment_features_masks = create_bert_input_features(tokenizer,  
                                    comment, max_seq_length=MAX_SEQ_LENGTH)
            # Employ the model to get the prediction and parse it
            predicted_label = self.model.predict([comment_feature_ids, comment_features_masks])
            predicted_label = np.argmax(predicted_label[0])
            if predicted_label==0: predicted_label="Non-Toxic"
            else: predicted_label="Toxic"
            
            sample = [comment, translated_comment, predicted_label]
            
            samples.append(sample)
        wandb.log({"text": wandb.Table(data=samples, 
                                       columns=["Comment", "Translated Comment", "Predicted Label"])})
# Garbage collection
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
# Create utility function to get a training ready model on demand
def get_training_model_cnn():
    inp_id = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int64, name="bert_input_ids")
    inp_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int64, name="bert_input_masks")
    inputs = [inp_id, inp_mask]

    hidden_state = transformers.TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased')(inputs)[0]
    pooled_output = hidden_state[:, 0]    
    reshaped_pooled = tf.keras.layers.Reshape((768,1), input_shape=(768,))(pooled_output)
    conv_1 = tf.keras.layers.Conv1D(64, 2, activation='relu')(reshaped_pooled)
    pooled_2 = tf.keras.layers.GlobalAveragePooling1D()(conv_1)
    dense_1 = tf.keras.layers.Dense(128, activation='relu')(pooled_2)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense_1)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=2e-5, 
                                            epsilon=1e-08), 
                loss='binary_crossentropy', metrics=['accuracy'])

    return model
# Garbage collection
gc.collect()

# Reinitialize wandb
wandb.init(project="jigsaw-toxic", id="distilbert-tpu-kaggle-weighted-cnn")
# Train the CNN-based model
start = time.time()

# Compile the model with TPU Strategy
with strategy.scope():
    model = get_training_model_cnn()
    
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