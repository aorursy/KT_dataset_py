import io

import math

import gzip

import nltk

import time

import random

import numpy as np

import tensorflow as tf

import gensim.downloader as api

import tensorflow_datasets as tfds

nltk.download('stopwords')



from collections import Counter

from nltk.corpus import stopwords

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import skipgrams

def load_data():

  text8_zip_file_path = api.load('text8', return_path=True)

  with gzip.open(text8_zip_file_path, 'rb') as file:

    file_content = file.read()

  wiki = file_content.decode()

  return wiki



wiki = load_data()
def get_drop_prob(x, threshold_value):

  return 1 - np.sqrt(threshold_value/x)



def subsample_words(words, word_counts):

  threshold_value = 1e-5

  total_count = len(words)

  freq_words = {word: (word_counts[word]/total_count) for word in set(words)}

  subsampled_words = [word for word in words if random.random() < (1 - get_drop_prob(freq_words[word], threshold_value))]

  return subsampled_words



def preprocess_text(text):

  # Replace punctuation with tokens so we can use them in our model

  text = text.lower()

  text = text.strip()

  text = text.replace('.', ' <PERIOD> ')

  text = text.replace(',', ' <COMMA> ')

  text = text.replace('"', ' <QUOTATION_MARK> ')

  text = text.replace(';', ' <SEMICOLON> ')

  text = text.replace('!', ' <EXCLAMATION_MARK> ')

  text = text.replace('?', ' <QUESTION_MARK> ')

  text = text.replace('(', ' <LEFT_PAREN> ')

  text = text.replace(')', ' <RIGHT_PAREN> ')

  text = text.replace('--', ' <HYPHENS> ')

  text = text.replace('?', ' <QUESTION_MARK> ')

  text = text.replace(':', ' <COLON> ')

  words = text.split()



  # Remove stopwords

  stopwords_eng = set(stopwords.words('english'))

  words = [word for word in words if word not in stopwords_eng]

  # Remove all the words with frequency less than 5

  word_counts = Counter(words)

  print("Count of words: %s" % (len(words)))

  filtered_words = [word for word in words if word_counts[word] >= 5]

  print("Count of filtered words: %s" % (len(filtered_words)))

  # Subsample words with threshold of 10^-5

  subsampled_words = subsample_words(filtered_words, word_counts)

  print("Count of subsampled words: %s" % (len(subsampled_words)))



  return word_counts, subsampled_words



word_counts, preprocessed_words = preprocess_text(wiki[:5000000])
preprocessed_words[1500:1550]
EMBEDDING_DIM = 128

BUFFER_SIZE = 1024

BATCH_SIZE = 64

EPOCHS = 5
tokenizer = Tokenizer()

tokenizer.fit_on_texts(preprocessed_words)

VOCAB_SIZE = len(tokenizer.word_counts)

vectorized_words = [tokenizer.word_index[word] for word in preprocessed_words]



pairs, labels = skipgrams(vectorized_words, VOCAB_SIZE, window_size=3, negative_samples=1.0, shuffle=True)

target_words = [p[0] for p in pairs]

context_words = [q[1] for q in pairs]



SAMPLE_SIZE = len(labels)

labels_sample = labels[:SAMPLE_SIZE]

target_words_sample = target_words[:SAMPLE_SIZE]

context_words_sample = context_words[:SAMPLE_SIZE]

train_size = int(len(labels_sample) * 0.9)

train_target_words, train_context_words, train_labels = target_words_sample[:train_size], context_words_sample[:train_size], labels_sample[:train_size]

test_target_words, test_context_words, test_labels = target_words_sample[train_size:], context_words_sample[train_size:], labels_sample[train_size:]



train_dataset = tf.data.Dataset.from_tensor_slices((train_target_words, train_context_words, train_labels)).shuffle(BUFFER_SIZE)

train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

test_dataset = tf.data.Dataset.from_tensor_slices((test_target_words, test_context_words, test_labels)).shuffle(BUFFER_SIZE)

test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)
print("# (train, test) batches: " + str(len(list(train_dataset.as_numpy_iterator()))) + ", " + str(len(list(test_dataset.as_numpy_iterator()))))
class SkipGramModel(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim):

      super(SkipGramModel, self).__init__()

      self.shared_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1, name='word_embeddings')

      self.flatten = tf.keras.layers.Flatten(name='flatten')

      self.dense1 = tf.keras.layers.Dense(64, activation=tf.nn.relu, name='dense_one')

      self.dropout1 = tf.keras.layers.Dropout(0.2, name = 'dropout1')

      self.dense2 = tf.keras.layers.Dense(32, activation=tf.nn.relu, name='dense_two')

      self.dropout2 = tf.keras.layers.Dropout(0.2, name = 'dropout2')

      self.pred = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name='predictions')



    def call(self, target_word, context_word, training=True):

      x = self.shared_embedding(target_word)

      y = self.shared_embedding(context_word)

      x = self.flatten(x)

      y = self.flatten(y)

      shared = tf.multiply(x, y)

      dense_output1 = self.dense1(shared)

      if training: dense_output1 = self.dropout1(dense_output1)

      dense_output2 = self.dense2(dense_output1)

      if training: dense_output2 = self.dropout2(dense_output2)

      output = self.pred(dense_output2)

      return tf.reshape(output, [-1])



model = SkipGramModel(VOCAB_SIZE+1, EMBEDDING_DIM)
optimiser = tf.keras.optimizers.Adam()

loss_fn = tf.keras.losses.BinaryCrossentropy()

train_acc_metric = tf.keras.metrics.BinaryAccuracy()

val_acc_metric = tf.keras.metrics.BinaryAccuracy()
@tf.function

def train_step(target_words, context_words, labels):

    with tf.GradientTape() as tape:

      preds = model(target_words, context_words)

      loss = loss_fn(labels, preds)

    gradients = tape.gradient(loss, model.trainable_variables)

    optimiser.apply_gradients(zip(gradients, model.trainable_variables))

    train_acc_metric.update_state(labels, preds)

    return loss



@tf.function

def test_step(target_words, context_words, labels):

    preds = model(target_words, context_words, training=False)

    loss = loss_fn(labels, preds)

    val_acc_metric.update_state(labels, preds)

    return loss



for epoch in range(EPOCHS):

  start_time = time.time()

  print("Starting epoch: %d " % (epoch,))

  cumm_loss = 0

  for step, (target_words, context_words, labels) in enumerate(train_dataset):

    train_loss = train_step(target_words, context_words, labels)

    cumm_loss += train_loss

  train_acc = train_acc_metric.result()

  print("Training acc over epoch: %.4f" % (float(train_acc),))

  train_acc_metric.reset_states()

  print("Cummulative loss: %.4f " % (cumm_loss,))



  test_cumm_loss = 0

  for step, (target_words, context_words, labels) in enumerate(test_dataset):

    test_loss = test_step(target_words, context_words, labels)

    test_cumm_loss += test_loss

  val_acc = val_acc_metric.result()

  print("Validation acc over epoch: %.4f" % (float(val_acc),))

  val_acc_metric.reset_states()

  print("Cummulative test loss: %f " % (test_cumm_loss,))

  print("Time taken: %.2fs" % (time.time() - start_time))
# Save weights to a Tensorflow Checkpoint file

model.save_weights('./skip_gram_weights_wiki_5000000')
word_embeddings_layer = model.layers[0]

weights = word_embeddings_layer.get_weights()[0]

print("Word Embeddings shape: %s" % (weights.shape,))



out_v = io.open('vecs.tsv', 'w', encoding='utf-8')

out_m = io.open('meta.tsv', 'w', encoding='utf-8')



for num, word in tokenizer.index_word.items():

  vec = weights[num] # skip 0, it's padding.

  out_m.write(word + "\n")

  out_v.write('\t'.join([str(x) for x in vec]) + "\n")

out_v.close()

out_m.close()