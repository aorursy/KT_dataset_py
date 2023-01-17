import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers
from tokenizers import decoders, processors
train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
train2['toxic'] = train2.toxic.round().astype(int)

validation_data = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test_data = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
train1.head()
train2.head()
validation_data.head()
validation_data['lang'].value_counts()
train = pd.concat([train1[['comment_text', 'toxic']],train2[['comment_text', 'toxic']].query('toxic==1'),train2[['comment_text', 'toxic']].query('toxic==0').sample(n=100000, random_state=0)])
train.head()
train.toxic.value_counts()
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
def func1(texts, tokenizer,chunk_size=256,max_len=512):
    tokenizer.enable_truncation(max_length=max_len)
    tokenizer.enable_padding(max_length=max_len)
    all_ids = []
    
    for i in tqdm(range(0, len(texts),chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)
def build_model_func(transformer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    history = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=history)
    model.compile(Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
AUTO = tf.data.experimental.AUTOTUNE
KDP = KaggleDatasets().get_gcs_path()
MODEL = 'jplu/tf-xlm-roberta-large'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
def func2(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    return np.array(enc_di['input_ids'])
x_train = func2(train['comment_text'].values, tokenizer, maxlen=192)
x_valid = func2(validation_data['comment_text'].values, tokenizer, maxlen=192)
x_test = func2(test_data['content'].values, tokenizer, maxlen=192)

y_train = train['toxic'].values
y_valid = validation_data['toxic'].values
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
bs = BATCH_SIZE
train_data = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(bs)
    .prefetch(AUTO)
)

valid_data = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(bs)
    .cache()
    .prefetch(AUTO)
)

test_data = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(bs)
)
with strategy.scope():
    transformer_layer = TFAutoModel.from_pretrained(MODEL)
    model = build_model_func(transformer_layer, max_len=192)
model.summary()
steps = x_train.shape[0] // bs
history = model.fit(
    train_data,
    steps_per_epoch=steps,
    validation_data=valid_data,
    epochs= 1
)
steps = x_valid.shape[0] // bs
history_2 = model.fit(
    valid_data.repeat(),
    steps_per_epoch=steps,
    epochs=10
)
sample_sub=pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
sample_sub.head()
sample_sub['toxic'] = model.predict(test_data, verbose=1)
sample_sub.to_csv('sample_submission.csv', index=False)