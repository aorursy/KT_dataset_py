!pip install -q keras-bert keras-rectified-adam

!wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

!unzip -o uncased_L-12_H-768_A-12.zip
SEQ_LEN = 128

BATCH_SIZE = 1024

EPOCHS = 15

LR = 1e-4
import os



pretrained_path = 'uncased_L-12_H-768_A-12'

config_path = os.path.join(pretrained_path, 'bert_config.json')

checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')

vocab_path = os.path.join(pretrained_path, 'vocab.txt')



# TF_KERAS must be added to environment variables in order to use TPU

os.environ['TF_KERAS'] = '1'
import tensorflow as tf

from keras_bert import get_custom_objects



# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
import codecs

from keras_bert import load_trained_model_from_checkpoint



token_dict = {}

with codecs.open(vocab_path, 'r', 'utf8') as reader:

    for line in reader:

        token = line.strip()

        token_dict[token] = len(token_dict)



with tpu_strategy.scope():

    model = load_trained_model_from_checkpoint(

        config_path,

        checkpoint_path,

        training=True,

        trainable=True,

        seq_len=SEQ_LEN,

    )
import pandas as pd



train_df = pd.read_csv('../input/nlp-getting-started/train.csv', index_col='id')

test_df = pd.read_csv('../input/nlp-getting-started/test.csv', index_col='id')
import string, re

from bs4 import BeautifulSoup



def strip_html(text):

    soup = BeautifulSoup(text, "html.parser")

    return soup.get_text()



#Removing the square brackets

def remove_between_square_brackets(text):

    return re.sub('\[[^]]*\]', '', text)



# Removing URL's

def remove_url(text):

    return re.sub(r'http\S+', '', text)



def add_space(text):

    return re.sub('%20', ' ', text)



def remove_hashtags(text):

    return re.sub('#', '', text)



def remove_at_signs(text):

    return re.sub('@', '', text)



#Removing the stopwords from text

def remove_punctuation(text):

    return text.translate(str.maketrans('', '', string.punctuation))



#Removing the noisy text

def denoise_text(text):

    text = strip_html(text)

    text = remove_between_square_brackets(text)

    text = add_space(text)

    text = remove_url(text)

    text = remove_hashtags(text)

    text = remove_at_signs(text)

    text = remove_punctuation(text)

    return text
def preprocess_df(df):

    df = df.fillna("")

    df['text'] = df['keyword'] + " " + df['text']

    del df['keyword']

    df['location'] = df['location'].astype('category')

    df['location'] = df['location'].cat.codes

    df['text'] = df['text'].apply(denoise_text)

    return df



train_df = preprocess_df(train_df)

test_df = preprocess_df(test_df)
import numpy as np



np.random.seed(1)



msk = np.random.rand(len(train_df)) < 0.8

train_df, dev_df = train_df[msk], train_df[~msk]
from tqdm import tqdm

from keras_bert import Tokenizer

import numpy as np



tokenizer = Tokenizer(token_dict)



def tokenize(df): 

    X, y = [], []

    for i, index in enumerate(tqdm(df.index.values)):

        ids, segments = tokenizer.encode(df.text.values[i], max_len=SEQ_LEN)

        X.append(ids)

        try:

            label = df.target.values[i]

            y.append(label)

        except:

            y.append(0)

    items = list(zip(X, y))

    np.random.shuffle(items)

    indices, sentiments = zip(*items)

    indices = np.array(indices)

    mod = indices.shape[0] % BATCH_SIZE

    if mod > 0:

        indices, sentiments = indices[:-mod], sentiments[:-mod]

    return [indices, np.zeros_like(indices)], np.array(sentiments)





X_train, y_train = tokenize(train_df)

X_dev, y_dev = tokenize(dev_df)
X_test = []

for i, index in enumerate(tqdm(test_df.index.values)):

    ids, segments = tokenizer.encode(test_df.text.values[i], max_len=SEQ_LEN)

    X_test.append(ids)

X_test = [np.array(X_test), np.zeros_like(X_test)]
from tensorflow.python import keras

from keras_radam import RAdam



with tpu_strategy.scope():

    inputs = model.inputs[:2]

    dense = model.get_layer('NSP-Dense').output

    outputs = keras.layers.Dense(units=2, activation='softmax')(dense)

    model = keras.models.Model(inputs, outputs)

    model.compile(

        RAdam(lr=LR),

        loss='sparse_categorical_crossentropy',

        metrics=['sparse_categorical_accuracy'],

    )
model.summary()
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy', patience=2, verbose=1,factor=0.5, min_lr=1e-5)
hist = model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[learning_rate_reduction])
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(10, 10))



epochs = np.arange(EPOCHS)

plt.subplot(2, 2, 1)

plt.xlabel('epochs')

plt.ylabel('loss')

sns.lineplot(epochs, hist.history['loss'])



plt.subplot(2, 2, 2)

plt.xlabel('epochs')

plt.ylabel('accuracy')

sns.lineplot(epochs, hist.history['sparse_categorical_accuracy'])



plt.subplot(2, 2, 3)

plt.xlabel('epochs')

plt.ylabel('val_loss')

sns.lineplot(epochs, hist.history['val_loss'])



plt.subplot(2, 2, 4)

plt.xlabel('epochs')

plt.ylabel('val_accuracy')

sns.lineplot(epochs, hist.history['val_sparse_categorical_accuracy'])



plt.show()
print("Accuracy of the model on Training Data is - {} %".format(model.evaluate(X_train,y_train)[1]*100))

print("Accuracy of the model on Dev Data is - {} %".format(model.evaluate(X_dev,y_dev)[1]*100))
classes = model.predict(X_test)[:, 0]
submission = pd.DataFrame(

    {'id': list(test_df.index.values),

     'target': list((classes < 0.5).astype(int)),

    }).set_index('id')
submission.to_csv('submission.csv')