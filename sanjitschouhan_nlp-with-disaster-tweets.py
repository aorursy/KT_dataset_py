! pip install tensorflow_datasets -q

# ! pip install tensorflow-gpu -q

! pip install tensorflow_addons -q
import tensorflow as tf

import tensorflow_datasets as tfds

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sbn

import tensorflow_addons as tfa
tf.__version__
train_input_dataset = pd.read_csv("../input/nlp-getting-started/train.csv")

test_input_dataset = pd.read_csv("../input/nlp-getting-started/test.csv")
train_input_dataset.head()
test_input_dataset.head()
train_target = train_input_dataset.pop('target')
train_dataset = tf.data.Dataset.from_tensor_slices((train_input_dataset.text.values, train_target.values))

test_dataset = tf.data.Dataset.from_tensor_slices(test_input_dataset.text.values)
train_dataset
test_dataset
for feat, targ in train_dataset.take(3):

    print(feat, targ)
for feat in test_dataset.take(3):

    print(feat)
tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set([''])
for text, label in train_dataset:

    tokens = tokenizer.tokenize(text.numpy())

    vocabulary_set.update(tokens)



for text in test_dataset:

    tokens = tokenizer.tokenize(text.numpy())

    vocabulary_set.update(tokens)
len(vocabulary_set)
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
def encode(text, label):

    encoded_text = encoder.encode(text.numpy())

    return encoded_text, label
def encode_map_fn(text, label):

    encoded_text, label = tf.py_function(encode,

                                inp=[text, label],

                                Tout=(tf.int64, tf.int64))

    encoded_text.set_shape([None])

    label.set_shape([])

    return encoded_text, label



def encode_map_fn_test(text):

        encoded_text, _ = tf.py_function(encode,

                                inp=[text, tf.constant(0,dtype=tf.int64)],

                                Tout=(tf.int64, tf.int64))

        encoded_text.set_shape([None])

        return encoded_text
encoded_train_dataset = train_dataset.map(encode_map_fn)

encoded_test_dataset = test_dataset.map(encode_map_fn_test)
for encoded_text, label in encoded_train_dataset.take(5):

    print(encoded_text, label)
for encoded_text in encoded_test_dataset.take(5):

    print(encoded_text)
encoded_test_dataset, encoded_train_dataset
TAKE_SIZE = int(len(train_target)*0.7)

BUFFER_SIZE = 1000

BATCH_SIZE = 64

TAKE_SIZE
shuffled_encoded_train_dataset = encoded_train_dataset.shuffle(len(train_target))

train_data = shuffled_encoded_train_dataset.take(TAKE_SIZE).shuffle(BUFFER_SIZE)

train_data = train_data.padded_batch(BATCH_SIZE, ((None,),()))



val_data = shuffled_encoded_train_dataset.skip(TAKE_SIZE)

val_data = val_data.padded_batch(BATCH_SIZE, ((None,),()))
for text, label in train_data.take(1):

    print(text, label)
embedding_dim = 16
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(encoder.vocab_size, embedding_dim),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),

    tf.keras.layers.Dense(64, 'relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1)

])
model.summary()
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=[tfa.metrics.F1Score(1), 'accuracy'])
model.fit(train_data, validation_data=val_data, epochs=20)
history = model.history.history
plt.plot(history['loss'], label="loss")

plt.plot(history['val_loss'], label="val_loss")

plt.legend()

plt.show()
plt.plot(history['accuracy'], label="accuracy")

plt.plot(history['val_accuracy'], label="val_accuracy")

plt.legend()

plt.show()
plt.plot(history['f1_score'], label="f1_score")

plt.plot(history['val_f1_score'], label="val_f1_score")

plt.legend()

plt.show()
pred = model.predict(encoded_test_dataset.padded_batch(BATCH_SIZE, ((None,))))
pred_col = (pred>0.5)*1
pred_col.shape
test_dataset
test_input_dataset.shape
test_input_dataset['target'] = pred_col[:,0]
test_input_dataset
result = test_input_dataset.drop(["keyword", "location", "text"], axis=1)
result.to_csv("tensorflow_result.csv",index=None)
!pip install spacy -q
train_input_dataset = pd.read_csv("../input/nlp-getting-started/train.csv")

test_input_dataset = pd.read_csv("../input/nlp-getting-started/test.csv")

train_input_dataset.head()
import spacy
nlp = spacy.load('en')
textcat = nlp.create_pipe("textcat", config= {

    "exclusive_classes": True,

    "architecture": "bow"

})
nlp.add_pipe(textcat)
textcat.add_label("1")

textcat.add_label("0")
train_texts = train_input_dataset.text
train_labels = [{'cats': {'1': label == 1,

                          '0': label == 0}} 

                for label in train_input_dataset['target']]
train_data = list(zip(train_texts, train_labels))

train_data[:3]
import random

from spacy.util import minibatch

from tqdm import tqdm



random.seed(1)

spacy.util.fix_random_seed(1)

optimizer = nlp.begin_training()



losses = {}

for epoch in range(10):

    random.shuffle(train_data)

    batches = minibatch(train_data, size=64)

    for batch in tqdm(batches):

        texts, labels = zip(*batch)

        nlp.update(texts, labels, sgd=optimizer, losses=losses)

    print(losses)
test_texts = test_input_dataset.text
test_docs = [nlp.tokenizer(text) for text in test_texts]
textcat = nlp.get_pipe('textcat')
scores, _ = textcat.predict(test_docs)
scores
predicted_labels = scores.argmax(axis=1)
predicted_labels
test_input_dataset['target'] = predicted_labels
test_input_dataset
test_output = test_input_dataset[['id','target']]
test_output.set_index('id', inplace=True)
test_output
test_output.to_csv("spacy_result.csv")