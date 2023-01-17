import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers
from transformers import TFAutoModel, AutoTokenizer, AutoConfig
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

from fastai.text import *
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten

# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
AUTO = tf.data.experimental.AUTOTUNE

# Data access
# GCS_DS_PATH = KaggleDatasets().get_gcs_path('')

# Configuration
EPOCHS = 2
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
from catalyst.data.sampler import DistributedSamplerWrapper, BalanceClassSampler
XLM_PROCESSED_PATH = Path("/kaggle/input/xlmrobertabase/xlm_roberta_processed/"); XLM_PROCESSED_PATH.ls()
# def one_hot_encode(x): return np.array([[1,0], [0,1]])[x]
class JigsawArrayDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids:np.array, attention_mask:np.array, toxic:np.array=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.toxic = toxic
    
    def __getitem__(self, idx):
        xb = (tensor(self.input_ids[idx]), tensor(self.attention_mask[idx]))
        yb = tensor(0.) if self.toxic is None else tensor(self.toxic[idx])
        return xb,yb    
        
    def __len__(self):
        return len(self.input_ids)
# train_ds
train_input_ids = np.load(XLM_PROCESSED_PATH/'translated_train_inputs/input_ids.npy')
train_attetion_mask = np.load(XLM_PROCESSED_PATH/'translated_train_inputs/attention_mask.npy').astype(np.int32)
train_toxic = np.load(XLM_PROCESSED_PATH/'translated_train_inputs/toxic.npy').astype(np.float32)
train_lang = np.load(XLM_PROCESSED_PATH/'translated_train_inputs/lang.npy', allow_pickle=True)
# labels for stratified batch sampler
train_stratify_labels = array(s1+s2 for (s1,s2) in zip(train_lang, train_toxic.astype(str)))
labels2int = {v:k for k,v in enumerate(np.unique(train_stratify_labels))}
labels = [labels2int[o] for o in train_stratify_labels]
balanced_sampler = BalanceClassSampler(labels)
idxs1 = list(iter(balanced_sampler))
idxs2 = list(iter(balanced_sampler))
idxs3 = list(iter(balanced_sampler))
idxs4 = list(iter(balanced_sampler))
idxs5 = list(iter(balanced_sampler))
# reshape for loss and metric
train_toxic = train_toxic.reshape(-1,1)
train_input_ids_sampled = np.vstack([train_input_ids[idxs] for idxs in [idxs1,idxs2,idxs3,idxs4,idxs5]])
train_attetion_mask_sampled = np.vstack([train_attetion_mask[idxs] for idxs in [idxs1,idxs2,idxs3,idxs4,idxs5]])
train_toxic_sampled = np.vstack([train_toxic[idxs] for idxs in [idxs1,idxs2,idxs3,idxs4,idxs5]])
train_input_ids_sampled.shape, train_attetion_mask_sampled.shape, train_toxic_sampled.shape
del train_input_ids, train_attetion_mask, train_toxic, train_lang
gc.collect()
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices(((train_input_ids_sampled, train_attetion_mask_sampled), train_toxic_sampled))
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)
# def generate(self):
#     for idx in idxs1:
#         yield ((train_input_ids[idx], train_attetion_mask[idx]), train_toxic[idx])
# train_dataset = (
#     tf.data.Dataset
#     .from_generator(generate, 
#                     ((tf.int32, tf.int32), tf.float32),
#                     ((tf.TensorShape([256]), tf.TensorShape([256])), tf.TensorShape([1]))
#                    )
#     .repeat()
#     .shuffle(2048)
#     .batch(BATCH_SIZE)
#     .prefetch(AUTO)
# )
valid_input_ids = np.load(XLM_PROCESSED_PATH/'valid_inputs/input_ids.npy')
valid_attention_mask = np.load(XLM_PROCESSED_PATH/'valid_inputs/attention_mask.npy').astype(np.int32)
valid_toxic = np.load(XLM_PROCESSED_PATH/'valid_inputs/toxic.npy').astype(np.float32).reshape(-1,1)
# valid_toxic = one_hot_encode(valid_toxic).astype(np.float32)
valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices(((valid_input_ids, valid_attention_mask), valid_toxic))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)
MODEL = 'jplu/tf-xlm-roberta-large'
def get_xlm_roberta(modelname=MODEL):        
    conf = AutoConfig.from_pretrained(modelname)
    conf.output_hidden_states = True
    model = TFAutoModel.from_pretrained(modelname, config=conf)
    return model
def build_model(xlm_roberta, max_len=256, p=0.5):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
    
    _, _, hidden_states = xlm_roberta([input_ids, attention_mask])
    x = tf.concat(hidden_states[-2:], -1)
    x = tf.concat((tf.reduce_mean(x, 1), tf.reduce_max(x, 1)), -1)    
    x = Dropout(rate=0.5)(x)
    out = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=(input_ids, attention_mask), outputs=out)
    
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1, name='binary_crossentropy')
    
    model.compile(Adam(lr=1e-5), loss=loss_fn, metrics=[tf.metrics.AUC()])
    
    return model
with strategy.scope():
    xlm_roberta = get_xlm_roberta()
    model = build_model(xlm_roberta)
model.summary()
BATCH_SIZE, EPOCHS = 128*strategy.num_replicas_in_sync, 2
BATCH_SIZE, EPOCHS
checkpoint_filepath = 'bestmodel'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_auc',
    mode='max',
    save_best_only=True)
n_steps = len(balanced_sampler) // BATCH_SIZE

train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    callbacks=[model_checkpoint_callback]
)



