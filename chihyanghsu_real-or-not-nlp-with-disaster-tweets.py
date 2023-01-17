# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
# Data Loading

train_set_path = '/kaggle/input/nlp-getting-started/train.csv'

train_set = pd.read_csv(train_set_path) # returns pd.DataFrame
# Data Preview

print('---Input Analysis---')

print('Number of rows in Input:', train_set.shape[0]) 

print('Number of columns in Input:', train_set.shape[1])# (7613,5)

print('First 5 rows: \n')

print(train_set.head()) # id, keyword, location, text, target
# ID

print('---ID Analysis---')

print('Unique IDs')

print(train_set.id.unique())

print('ID Counts')

print(train_set.id.value_counts())
# Keyword

print('---Keyword Analysis---')

print('Unique Keywords')

print(train_set.keyword.unique())

print('Keyword Counts')

print(train_set.keyword.value_counts())

train_set.keyword.value_counts().plot.barh(figsize=(5,100), title='Keyword Counts', )
# Text

print('---Text Analysis---')

print('Text Review')

for text in train_set.text:

    print(text)



# Character Counts

train_set['char_counts'] = train_set['text'].str.len()

train_set.hist(column='char_counts')



# Word Counts

train_set['word_counts'] = train_set['text'].str.split().map(lambda x: len(x))

train_set.hist(column='word_counts')
# Target Distribution

print('---Target Analysis---')

print('Unique Targets')

print(train_set.target.unique())

print('Target Counts')

print(train_set.target.value_counts())

print(train_set.target.value_counts(normalize=True))

train_set.target.value_counts().plot.bar()

plt.title('Class Distribution')
# Keyword counts per class

is_disaster = train_set['target'] == 1

disaster_keyword_counts = train_set[is_disaster].keyword.value_counts()

not_disaster_keyword_counts = train_set[~is_disaster].keyword.value_counts()

joint_keyword_counts = pd.concat([disaster_keyword_counts, not_disaster_keyword_counts], axis=1, join='outer',

                                 keys=['disaster', 'not_disaster'])

print(joint_keyword_counts)

joint_keyword_counts.plot.barh(subplots=False, figsize=(5,100))



bool_disaster_exclusive = pd.isnull(joint_keyword_counts.not_disaster)

disaster_exclusive_counts = joint_keyword_counts[bool_disaster_exclusive]



print('Disaster Exclusive Keywords:')

print(disaster_exclusive_counts)



bool_not_disaster_exclusive = pd.isnull(joint_keyword_counts.disaster)

not_disaster_exclusive_counts = joint_keyword_counts[bool_not_disaster_exclusive]



print('Not Disaster Exclusive Keywords:')

print(not_disaster_exclusive_counts)
# Character Counts per Class

train_set.hist(column='char_counts', by='target')



# Word Counts per Class

train_set.hist(column='word_counts', by='target')
import re

import string
# remove_punctiations:

# URLs

def remove_urls(text): 

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



# Punctuations

def remove_punctuations(text):

    table = str.maketrans('', '', string.punctuation)

    return text.translate(table)

train_set['clean_text'] = train_set['text']

train_set['clean_text'] = train_set['clean_text'].apply(lambda x : remove_urls(x))

train_set['clean_text'] = train_set['clean_text'].apply(lambda x : remove_punctuations(x))



for text in train_set.clean_text:

    print(text)
train_set = train_set.sample(frac=1)

train_set.head()
inputs = train_set.clean_text.tolist()

targets = train_set.target.tolist()

    

print(inputs)

print(targets)
from nltk.corpus import stopwords

stop_words = None

stop_words = set(stopwords.words('english'))

print(stop_words)



def remove_stopwords(inputs, stop_words):

    if not stop_words:

        return corpus

    

    new_inputs = []

    for sentence in inputs:

        for word in stop_words:

            word_with_space = " " + word + " " # Assuming words leads and trails with space. If not, subwords might be replaced.

            sentence = sentence.replace(word_with_space, " ")

        new_inputs.append(sentence)

            

    return new_inputs



inputs = remove_stopwords(inputs, stop_words)

print(inputs)
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



max_length = 50

padding_type = 'post'

trunc_type = 'post'



tokenizer = Tokenizer()

tokenizer.fit_on_texts(inputs)



word_index = tokenizer.word_index

vocab_size = len(word_index)



sequences = tokenizer.texts_to_sequences(inputs)

padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print('Vocabulary Size: ', len(word_index))
print('Number of Inputs:', len(inputs))

print('Number of Targets:', len(targets))



train_size = 7000

train_inputs = padded[:train_size]

train_targets = targets[:train_size]

train_targets = np.array(train_targets)



print(train_inputs)

print(train_targets)



dev_inputs = padded[train_size:]

dev_targets = targets[train_size:]

dev_targets = np.array(dev_targets)



print(dev_inputs)

print(dev_targets)



plt.hist(train_targets)

plt.hist(dev_targets)
embedding_path = '/kaggle/input/glove6b100dtxt/glove.6B.100d.txt'



embeddings_index = {}

with open(embedding_path) as f:

    for line in f:

        values = line.split()

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs



print(len(embeddings_index))



vocab_size = len(word_index)

embedding_dim = 100

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embeddings_matrix[i] = embedding_vector



print(len(embeddings_matrix))
from tensorflow.keras.optimizers import Adam



compile_opts = {}

compile_opts['loss'] = 'binary_crossentropy'

compile_opts['optimizer'] = Adam(learning_rate=1e-3)

compile_opts['metrics'] = ['accuracy']
fit_options = {}

fit_options['x'] = train_inputs

fit_options['y'] = train_targets

fit_options['validation_data'] = (dev_inputs, dev_targets)

fit_options['batch_size'] = 128

fit_options['epochs'] = 100

fit_options['verbose'] = 2

fit_options['callbacks'] = None
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py

    

from tensorflow.keras.layers import Dense, Input

import tensorflow_hub as hub

import tokenization



def bert_encode(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

            

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)

        tokens += [0] * pad_len

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        segment_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)



def build_model(bert_layer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(clf_output)

    

    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model



module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)



vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)



train_input = bert_encode(train_set.text.values, tokenizer, max_len=50)

#test_input = bert_encode(test.text.values, tokenizer, max_len=160)

train_labels = train_set.target.values



bert_model = build_model(bert_layer, max_len=50)

bert_model.summary()

bert_history = bert_model.fit(train_input, train_labels, validation_split=0.2, epochs=5, batch_size=16, verbose=2)
import tensorflow as tf



lstm_model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, dropout=0.5)), # recurrent_dropout too slow

    tf.keras.layers.Dense(1, activation='sigmoid')

])



lstm_model.compile(**compile_opts)

lstm_model.summary()

lstm_history = lstm_model.fit(**fit_options)



print("Training Complete")
gru_model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, dropout=0.5)),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



gru_model.compile(**compile_opts)

gru_model.summary()

gru_history = gru_model.fit(**fit_options)



print("Training Complete")
conv_model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same'),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



conv_model.compile(**compile_opts)

conv_model.summary()

conv_history = conv_model.fit(**fit_options)



print("Training Complete")
def plot_graphs(history):

    plt.subplot(1,2,1)

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.xlabel('Epochs')

    plt.ylabel('accuracy')

    plt.legend(['accuracy', 'val_accuracy'])

    plt.show()

    

    plt.subplot(1,2,2)

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.xlabel('Epochs')

    plt.ylabel('loss')

    plt.legend(['loss', 'val_loss'])

    plt.show()



plot_graphs(lstm_history)

plot_graphs(gru_history)

plot_graphs(conv_history)

plot_graphs(bert_history)
import io



reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])



def decode_sentence(text):

    return ' '.join([reverse_word_index.get(i, '?') for i in text])



e = lstm_model.layers[0]

weights = e.get_weights()[0]

print(weights.shape) # shape: (vocab_size, embedding_dim)



# Expected output

# (1000, 16)



out_v = io.open('/kaggle/working/vecs.tsv', 'w', encoding='utf-8')

out_m = io.open('/kaggle/working/meta.tsv', 'w', encoding='utf-8')

for word_num in range(1, vocab_size):

    word = reverse_word_index[word_num]

    embeddings = weights[word_num]

    out_m.write(word + "\n")

    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")

out_v.close()

out_m.close()
lstm_pred = lstm_model.predict(dev_inputs)

lstm_pred = [x[0] for x in lstm_pred]

gru_pred = gru_model.predict(dev_inputs)

conv_pred = conv_model.predict(dev_inputs)



dev_sentences = inputs[train_size:]

dev_df = pd.DataFrame(dev_sentences, columns=['Sentence'])

dev_df['Truth'] = dev_targets == 1

dev_df['LSTM'] = lstm_pred

dev_df['LSTM'] = dev_df['LSTM'] >= 0.5

dev_df['GRU'] = gru_pred

dev_df['LSTM'] = dev_df['GRU'] >= 0.5

dev_df['CONV'] = conv_pred

dev_df['LSTM'] = dev_df['CONV'] >= 0.5

print(dev_df)

FN = []

for truth, pred in zip(dev_targets, lstm_pred):

    if truth == 1 and pred < 0.5:

        FN.append(True)

    else:

        FN.append(False)

        

print(dev_df[FN]['Sentence'])

    
test_set = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test_set['clean_text'] = test_set['text']

test_set['clean_text'] = test_set['clean_text'].apply(lambda x : remove_urls(x))

test_set['clean_text'] = test_set['clean_text'].apply(lambda x : remove_punctuations(x))



for text in test_set.clean_text:

    print(text)

    

test_inputs = test_set.clean_text.tolist()

test_inputs = remove_stopwords(test_inputs, stop_words)

test_sequences = tokenizer.texts_to_sequences(test_inputs)

test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
lstm_pred = lstm_model.predict(test_padded)

print(lstm_pred)

num_test_samples = len(lstm_pred)
sample_sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

pred = np.round(lstm_pred).astype(int).reshape(3263)

sub = pd.DataFrame({'id':sample_sub['id'].values.tolist(), 'target':pred})

sub.to_csv('/kaggle/working/submission.csv', index=False)