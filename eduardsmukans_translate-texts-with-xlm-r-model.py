!pip install 'tensorflow~=1.15'
!pip install 'gast==0.2.2'
import tensorflow as tf
import tensorflow_hub as hub

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
import collections

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

import pandas as pd
from datetime import datetime
tfe = tf.contrib.eager
tfe.enable_eager_execution()
import zipfile

!gsutil -m cp gs://translator-lv-en/annotations.zip annotations.zip

with zipfile.ZipFile('annotations.zip', 'r') as zip_ref:
    zip_ref.extractall()
raw_en_annotations = []
annotation_en_file = 'annotations/annotations.en'
with open(annotation_en_file, 'r') as f:
    raw_en_annotations = f.readlines()

def prepare_caption(caption: str):
  return f'<start> {caption.strip()} <end>'

skip_index_list = []
for i in range(len(raw_en_annotations[:5000])):
  if len(raw_en_annotations[i].split()) > 15:
    skip_index_list.append(i)

raw_en_annotations = [sentence for i, sentence in enumerate(raw_en_annotations[:5000]) if i not in skip_index_list]
prepared_en_caption_list = shuffle(raw_en_annotations[:1000], random_state=1)
prepared_en_caption_list = [prepare_caption(caption) for caption in prepared_en_caption_list]

raw_lv_annotations = []
annotation_lv_file = 'annotations/annotations.lv'
with open(annotation_lv_file, 'r') as f:
    raw_lv_annotations = f.readlines()

raw_lv_annotations = [sentence for i, sentence in enumerate(raw_lv_annotations[:5000]) if i not in skip_index_list]
prepared_lv_caption_list = shuffle(raw_lv_annotations[:1000], random_state=1)
prepared_lv_caption_list = [prepare_caption(caption) for caption in prepared_lv_caption_list]
len(prepared_lv_caption_list)
!pip install sentencepiece
import torch
xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')
xlmr.eval()
word_embeddings = [xlmr.extract_features(xlmr.encode(sentence))[0].tolist() for sentence in prepared_en_caption_list]
text = [word for word in word_embeddings if len(word) < 33]
expanded_embeddings  = []

for word in text:
  for _ in range(32 - len(word)):
    word.append(np.zeros(1024))
  expanded_embeddings.append(word)
for word in expanded_embeddings:
  assert len(word) == 32

assert len(expanded_embeddings) == 1000 
# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)
# Choose the top 1000 words from the vocabulary
top_k = 1000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(prepared_lv_caption_list)
train_seqs = tokenizer.texts_to_sequences(prepared_lv_caption_list)
len(tokenizer.word_counts)
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'
# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(prepared_lv_caption_list)
# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(train_seqs)
# Create training and validation sets using an 80-20 split
word_train, word_val, cap_train, cap_val = train_test_split(expanded_embeddings,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)
len(word_train), len(cap_train), len(word_val), len(cap_val)
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = top_k + 1
num_steps = len(word_train) // BATCH_SIZE
# Shape of the vector extracted from XML-R is (32, 1024)
# These two variables represent that vector shape
features_shape = 1024
dataset = tf.data.Dataset.from_tensor_slices((word_train, cap_train))

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, 64, hidden_size)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    # attention_weights shape == (batch_size, 64, 1)
    # you get 1 at the last axis because you are applying score to self.V
    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x
class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights
    # return x, state

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)
checkpoint_path = "/kaggle/working/checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
  # restoring the latest checkpoint in checkpoint_path
  ckpt.restore(ckpt_manager.latest_checkpoint)
# adding this in a separate cell because if you run the training cell
# many times, the loss_plot array will be reset
loss_plot = []
@tf.function
def train_step(img_tensor, target):
  loss = 0

  # initializing the hidden state for each batch
  # because the captions are not related from image to image
  hidden = decoder.reset_state(batch_size=target.shape[0])

  dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

  with tf.GradientTape() as tape:
      features = encoder(img_tensor)

      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)
          loss += loss_function(target[:, i], predictions)

          # using teacher forcing
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)

  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss
def evaluate_annotation(img_tensor):
  img_tensor = tf.expand_dims(img_tensor, 0)
  features = encoder(img_tensor)
  hidden = decoder.reset_state(batch_size=1)
  dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
  result = []
  for i in range(max_length):
    predictions, hidden, _ = decoder(dec_input, features, hidden)
    predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()

    result.append(tokenizer.index_word[predicted_id])

    if tokenizer.index_word[predicted_id] == '<end>':
        return result
    dec_input = tf.expand_dims([predicted_id], 0)
  return result
EPOCHS = 100

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0
    last_tensor = None
    last_target = None

    for (batch, (img_tensor, target)) in enumerate(dataset):
        last_tensor = img_tensor
        last_target = target
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)

    # if epoch % 10 == 0:
    print('Real:')
    print(' '.join([tokenizer.index_word[predicted_id.numpy()] for predicted_id in last_target[-1]]))
    print('Predicted:')
    print(' '.join(evaluate_annotation(last_tensor[-1])))
    ckpt_manager.save()

    print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                         total_loss/num_steps))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.show()
def evaluate(word):
    hidden = decoder.reset_state(batch_size=1)

    word_tensor_val = tf.expand_dims(word, 0)
    features = encoder(word_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, _ = decoder(dec_input, features, hidden)

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result

        dec_input = tf.expand_dims([predicted_id], 0)
    return result
# captions on the validation set
rid = np.random.randint(0, len(word_val))
word = word_val[rid]
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])

result = evaluate(word)

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))

def translate(en_text):
  embedding = xlmr.extract_features(xlmr.encode(en_text))[0].tolist()

  assert len(embedding) <= 32
  for _ in range(32 - len(embedding)):
    embedding.append(np.zeros(1024))

  return ' '.join(evaluate(embedding))
for _ in range(20):
    rid = np.random.randint(0, len(prepared_en_caption_list))
    en_text = prepared_en_caption_list[rid]
    lv_text = translate(en_text)

    print ('EN:', en_text.replace('<start> ', '').replace(' <end>', '').lower())
    print ('LV:', lv_text.replace('<start> ', '').replace(' <end>', ''))
    print('')