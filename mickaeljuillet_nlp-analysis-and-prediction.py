import numpy as np
import pandas as pd

import os
DIR_PATH = "../input/contradictory-my-dear-watson"

FILENAME_TRAIN_DATA = "train.csv"
FILENAME_TEST_DATA = "test.csv"

MODEL_NAME = 'bert-base-multilingual-cased'

MAX_LEN = 50
df = pd.read_csv(os.path.join(DIR_PATH, FILENAME_TRAIN_DATA)) 
# How many language in the dataset ?
df.language.nunique()
# But now, we need to understand the language distribution
bbox = dict(boxstyle="round", fc="0.8")

ax = df.language.value_counts().plot.bar(figsize=(20,4), rot=45);

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x()+.125, p.get_height()+250),bbox=bbox)
ax = df.label.value_counts().plot.barh(figsize=(8,2.5), xlim=[3800, 4200]);
temp_df = df.groupby(['language', 'label']).id.count().unstack()

ax = temp_df.div(temp_df.sum(1), 0).sort_values([0, 1, 2]).plot.bar(stacked=True)
ax.hlines([0.33, 0.66], -1, 15, color='r');
from string import punctuation
en_df = df[df.language == "English"]
en_df['length_pre'] = en_df.premise.apply(len)
en_df['length_hyp'] = en_df.hypothesis.apply(len)
en_df.assign(n_v=lambda x: np.log(x.length_pre/x.length_hyp)).n_v.hist(bins=50)
ax = en_df.assign(n_v=lambda x: x.length_pre/x.length_hyp).groupby('label').n_v.plot.hist(bins=100, alpha=.8, xlim=[0, 6]);

os.environ["WANDB_API_KEY"] = "0" ## to silence warning
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy() # for CPU and single GPU
    print('Number of replicas:', strategy.num_replicas_in_sync)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
def enc_sent(s):
    tokens = list(tokenizer.tokenize(s)) + ['[SEP]']
    return tokenizer.convert_tokens_to_ids(tokens)
def enc_bert(hyp, prem, tokenizer):
    s1 = tf.ragged.constant([enc_sent(s) for s in list(hyp)])
    s2 = tf.ragged.constant([enc_sent(s) for s in list(prem)])
    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*s1.shape[0]
    input_word_ids = tf.concat([cls, s1, s2], axis=-1)
    
    input_mask = tf.ones_like(input_word_ids).to_tensor()
    
    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(s1)
    type_s2 = tf.ones_like(s2)    
    
    input_type_ids = tf.concat([type_cls, type_s1, type_s2], axis=-1).to_tensor()
    
    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}
    
    return inputs
train_input = enc_bert(df.premise.values, df.hypothesis.values, tokenizer)
# CREATING & TRAINING MODEL
def create_model():
    bert_encoder = TFBertModel.from_pretrained(MODEL_NAME)
    
    input_word_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_mask")
    input_type_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_type_ids")    
    
    embedding = bert_encoder([input_word_ids, input_mask, input_type_ids])[0]
    output = tf.keras.layers.Dense(3, activation='softmax')(embedding[:, 0, :])
    
    model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=output)
    model.compile(
        tf.keras.optimizers.Adam(lr=1e-5), 
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    
    return model
with strategy.scope():
    model = create_model()
    model.summary()

model.fit(train_input, df.label.values, epochs = 2, verbose = 1, batch_size = 64, validation_split = 0.2)
test_df = pd.read_csv(os.path.join(DIR_PATH, FILENAME_TEST_DATA))
test_input = enc_bert(test_df.premise.values, test_df.hypothesis.values, tokenizer)
predictions = [np.argmax(i) for i in model.predict(test_input)]

submission = test_df.id.copy().to_frame()
submission['prediction'] = predictions

submission.to_csv("submission.csv", index = False)
