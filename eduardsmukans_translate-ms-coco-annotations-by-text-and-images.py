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


!gsutil -m cp gs://translator-lv-en/en-lv-annotations.zip en-lv-annotations.zip

with zipfile.ZipFile('en-lv-annotations.zip', 'r') as zip_ref:
  zip_ref.extractall()
raw_en_annotations = []
annotation_file = 'en-lv-annotations.json'
with open(annotation_file, 'r') as f:
    raw_annotations = json.load(f)

def prepare_caption(caption: str):
    return f'<start> {caption.strip()} <end>'

prepared_caption_list = []
for annotation in raw_annotations:
    caption = {
        'en': prepare_caption(annotation['en']),
        'lv': prepare_caption(annotation['lv']),
        'image': annotation['image']
    }
    prepared_caption_list.append(caption)

train_caption_list = shuffle(prepared_caption_list, random_state=1)
# Limit dataset
train_caption_list = train_caption_list[:2000]

print(len(train_caption_list))
from IPython.display import HTML, display
import time

def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))
import requests
import hashlib

image_dir = '/kaggle/working/images'
os.makedirs(image_dir, exist_ok=True)

max_progress = len(train_caption_list)
progress_counter = 0

out = display(progress(0, max_progress), display_id=True)

for caption in train_caption_list:
    image = requests.get(caption['image'])

    try:
        tf.image.decode_jpeg(image.content, channels=3)
    except tf.errors.InvalidArgumentError:
        progress_counter += 1
        out.update(progress(progress_counter, max_progress))
        continue

    extension = caption['image'].split('.')[-1]
    image_hash = hashlib.sha224(caption['image'].encode()).hexdigest()
    image_path = os.path.join(image_dir, 'coco-' + image_hash)

    caption['local_image_path'] = f'{image_path}.{extension}'

    with open(caption['local_image_path'], 'wb') as file:
        file.write(image.content)

    progress_counter += 1
    out.update(progress(progress_counter, max_progress))

train_caption_list = [caption for caption in train_caption_list if caption.get('local_image_path')]

len(train_caption_list)
!pip install sentencepiece
import torch
xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')
xlmr.eval()
max_progress = len(train_caption_list)
progress_counter = 0

out = display(progress(0, max_progress), display_id=True)
for sentence in train_caption_list:
    sentence['text_embedding'] = xlmr.extract_features(xlmr.encode(sentence.get('en')))[0].tolist()

    progress_counter += 1
    out.update(progress(progress_counter, max_progress))
filtered_caption_list = []

for sentence in train_caption_list:
    if len(sentence['text_embedding']) > 65:
        continue
    filtered_caption_list.append(sentence)

print(len(filtered_caption_list))
for sentence in filtered_caption_list:
    extended_sentence = []
    for subword in sentence['text_embedding']:
        extended_sentence.append(np.append(subword, np.zeros(1024)))

    for _ in range(64 - len(extended_sentence)):
        extended_sentence.append(np.zeros(2048))
    sentence['text_embedding'] = np.array(extended_sentence).astype(np.float32)
for sentence in filtered_caption_list:
    assert len(sentence['text_embedding']) == 64
    assert len(sentence['text_embedding'][0]) == 2048
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
import requests

def load_image(local_image_path):
    img = tf.io.read_file(local_image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, local_image_path

def download_image(image_url):
    img = requests.get(image_url).content
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    return img
max_progress = len(filtered_caption_list)
progress_counter = 0

out = display(progress(0, max_progress), display_id=True)

image_path_list = [caption['local_image_path'] for caption in filtered_caption_list]
image_dataset = tf.data.Dataset.from_tensor_slices(image_path_list)
image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)
image_embedding_list = []

for img, local_image_path in image_dataset:
    batch_features = image_features_extract_model(img)
    batch_features = tf.reshape(batch_features,
                                (batch_features.shape[0], -1, batch_features.shape[3]))

    for bf, p in zip(batch_features, local_image_path):
        image_embedding_list.append({
          'image_embedding': bf.numpy().astype(np.float32),
          'local_image_path': p.numpy().decode()
        })
        progress_counter += 1
        out.update(progress(progress_counter, max_progress))
for caption in filtered_caption_list:
    caption['image_embedding'] = [image['image_embedding'] for image in image_embedding_list if image['local_image_path'] == caption['local_image_path']][0]
print(filtered_caption_list[0].keys())
# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)
# Choose the top 1000 words from the vocabulary
top_k = 3000

train_captions = [caption['lv'] for caption in filtered_caption_list]
train_captions *= 2

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
train_seqs = tokenizer.texts_to_sequences(train_captions)
len(tokenizer.word_counts)
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'
# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)
# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(train_seqs)
# Create training and validation sets using an 80-20 split
text_embeddings = [caption['text_embedding'] for caption in filtered_caption_list]
image_embeddings = [caption['image_embedding'] for caption in filtered_caption_list]

train_embeddings = text_embeddings + image_embeddings
word_train, word_val, cap_train, cap_val = train_test_split(train_embeddings,
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
features_shape = 2048
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

def evaluate_image(image):
    attention_plot = np.zeros((max_length, 64))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(download_image(image), 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def plot_attention(image, result, attention_plot):
    temp_image = np.array(download_image(image) * 255).astype(np.uint8)

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()
for _ in range(30):
  rid = np.random.randint(0, len(word_val))
  embedding = word_val[rid]
  caption = None
  i = 0
  for sentence in filtered_caption_list:
    if np.allclose(sentence['image_embedding'], embedding) or np.allclose(sentence['text_embedding'], embedding):
      caption = sentence
      break
    i += 1

  real_caption = caption['en']

  text_result = evaluate(caption['text_embedding'])
  image_result = evaluate(caption['image_embedding'])

  print('Real Caption:', real_caption.replace(' <end>', '').replace('<start> ', ''))
  print('Real Caption:', i)
  print('Caption image:', caption['image'])
  print('Predicted Caption by text:', ' '.join(text_result).replace(' <end>', '').replace('<start> ', ''))
  print('Predicted Caption by image:', ' '.join(image_result).replace(' <end>', '').replace('<start> ', ''))
  print('')
# image caption
rid = np.random.randint(0, len(filtered_caption_list))
caption = filtered_caption_list[rid]
real_caption = caption['lv']


result, attention_plot = evaluate_image(caption['image'])

print ('Real Caption:', real_caption)
print ('Predicted Caption:', ' '.join(result))

plot_attention(caption['image'], result, attention_plot)