from google.colab import drive
drive.mount('/content/drive')

import os
import re
import numpy as np
import pickle
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Bidirectional, Dropout, Embedding, LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
def tokenize(sent):
  return [x.strip() for x in re.split(r'(\W+)', sent) if x.strip()]
with open('drive/My Drive/final_project/train-v2.0.json', 'r') as f:
  content = json.loads(f.read())
type(content)
data = content['data']
data[0].keys()
def parse_data(data):
  vocab_set = set()
  vocab = {}
  triplex_list = []
  context_list = []
  question_list = []
  answer_list = []
  original_answer_list = []


  # Context and questions extracting
  for topic in data:
    for part in topic['paragraphs']:
      blocks = part['qas']
      for block in blocks:
        if len(block['answers']) == 1:
          context = part['context']
          vocab_set |= set(tokenize(context))
          context_list.append(context)
          vocab_set |= set(tokenize(block['question']))
          question_list.append(block['question'])
  
  # Making dictionary with shape {'token': number}, where numbers are in range 1..
  i = 1
  for token in vocab_set:
    vocab[token] = i
    i += 1
  
  # Context vectorization and finding of context_maxlen
  context_vectors = []
  context_maxlen = 0
  for context in context_list:
    vectorized_context = []
    tokens = tokenize(context)
    for token in tokens:
      vectorized_context.append(vocab[token])
    context_vectors.append(vectorized_context)
    if len(tokens) > context_maxlen:
      context_maxlen = len(tokens)
  context_vectors = pad_sequences(context_vectors, maxlen=context_maxlen, padding='post')
  
  # Answer extracting
  for topic in data:
    for part in topic['paragraphs']:
      blocks = part['qas']
      for block in blocks:
        if len(block['answers']) == 1:
          context = part['context']
          tokens = tokenize(context)
          answer_vector = np.zeros(len(tokens))
          answer_start = block['answers'][0]['answer_start']
          text = block['answers'][0]['text']
          before_answer = context[:answer_start]
          tokens_before = tokenize(before_answer)
          answer_symbols = context[answer_start:answer_start + len(text)]
          answer_tokens = tokenize(answer_symbols)
          if answer_tokens != tokenize(text):
            print('Mistake')
            break
          answer_vector[len(tokens_before): len(tokens_before) + len(answer_tokens)] = 1
          original_answer_list.append(text)
          answer_list.append(answer_vector)
  
  answer_vectors = pad_sequences(answer_list, maxlen=context_maxlen, padding='post')
  
  # Question vectorization and question_maxlen finding 
  question_vectors = []
  question_maxlen = 0
  for question in question_list:
    vectorized_question = []
    tokens = tokenize(question)
    for token in tokens:
      vectorized_question.append(vocab[token])
    question_vectors.append(vectorized_question)
    if len(tokens) > question_maxlen:
      question_maxlen = len(tokens)
  question_vectors = pad_sequences(question_vectors, maxlen=question_maxlen, padding='post')

  return context_vectors, question_vectors, answer_vectors,\
   vocab, context_maxlen, question_maxlen, context_list, question_list, answer_list, original_answer_list
context_vectors, question_vectors, answer_vectors, vocab, context_maxlen,\
 question_maxlen, context_list, question_list, answer_list, original_answer_list = parse_data(data)
from sklearn.model_selection import train_test_split
context_train, context_test, question_train, question_test,\
  answer_train, answer_test = train_test_split(context_vectors,
  question_vectors, answer_vectors, shuffle=False, test_size=0.2, random_state=42)
EMBED_HIDDEN_SIZE = 100
CONTEXT_HIDDEN_SIZE = 200
QUESTION_HIDDEN_SIZE = 200

vocab_size = len(vocab) + 1
glove_dir = 'drive/My Drive/final_project'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
  values = line.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embeddings_index[word] = coefs
f.close()

len(embeddings_index)
embedding_matrix = np.zeros((vocab_size, EMBED_HIDDEN_SIZE))
for word, i in vocab.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector

embedding_matrix.shape
# build model

context = layers.Input(shape=(context_maxlen,), dtype='int32')
encoded_context = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(context)
context_1_LSTM = layers.Bidirectional(layers.LSTM(CONTEXT_HIDDEN_SIZE,
                                                  return_sequences=True))(encoded_context)
context_2_LSTM = layers.Bidirectional(layers.LSTM(CONTEXT_HIDDEN_SIZE))(context_1_LSTM)
#dropout_1 = layers.Dropout(0.5)(context_2_LSTM)

question = layers.Input(shape=(question_maxlen,), dtype='int32')
encoded_question = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
question_1_LSTM = layers.Bidirectional(layers.LSTM(QUESTION_HIDDEN_SIZE,
                                                  return_sequences=True))(encoded_question)
question_2_LSTM = layers.Bidirectional(layers.LSTM(QUESTION_HIDDEN_SIZE))(question_1_LSTM)
#dropout_2 = layers.Dropout(0.5)(question_2_LSTM)

merged = layers.concatenate([context_2_LSTM, question_2_LSTM])

preds_1 = layers.Dense(10 * context_maxlen, activation='relu')(merged)
dropout_ = layers.Dropout(0.2)(preds_1)
preds_2 = layers.Dense(context_maxlen, activation='sigmoid')(dropout_)

model = Model([context, question], preds_2)
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

model.layers[3].set_weights([embedding_matrix])
model.layers[3].trainable = False
optimizer = keras.optimizers.RMSprop(learning_rate=0.1)

model.compile(optimizer=optimizer,
              loss=BinaryCrossentropy(),
              metrics=['accuracy', \
                       CosineSimilarity(axis=1)])

model.summary()
model.load_weights('/content/drive/My Drive/final_project/weights_file_16.h5')
BATCH_SIZE = 256
EPOCHS = 10

callback = ModelCheckpoint(filepath='/content/drive/My Drive/final_project/weights_file_1.h5',
              monitor='val_loss',
              mode='auto',
              save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=0.001)


print('Training')
history = model.fit([context_train, question_train], answer_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.05,
          callbacks=[callback, reduce_lr])
