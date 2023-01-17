!pip install tldextract
import numpy as np
import json
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import gc
import random
import os
import pickle
import tensorflow as tf
from tensorflow.python.util import deprecation
from urllib.parse import urlparse
import tldextract
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models, layers, backend, metrics
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
# set random seed
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)
# other setup
%config InlineBackend.figure_format = 'retina'
pd.set_option('max_colwidth', 500)
deprecation._PRINT_DEPRECATION_WARNINGS = False
# load data
data = pd.read_csv('../input/data.csv')
# shuffle data
data = data.sample(frac=1, random_state=0)
print(f'Data size: {data.shape}')
data.head()
val_size = 0.2
train_data, val_data = train_test_split(data, test_size=val_size, stratify=data['label'], random_state=0)
print(f'Train shape: {train_data.shape}, Validation shape: {val_data.shape}')
data.label.value_counts().plot.barh()
plt.title('All Data')
plt.show()
good, bad = data.label.value_counts()
print(f'Ratio of data between target labels (bad & good) is {bad//bad}:{good//bad}')
def parsed_url(url):
    # extract subdomain, domain, and domain suffix from url
    # if item == '', fill with '<empty>'
    subdomain, domain, domain_suffix = ('<empty>' if extracted == '' else extracted for extracted in tldextract.extract(url))
    return [subdomain, domain, domain_suffix]
def extract_url(data):
    # parsed url
    extract_url_data = [parsed_url(url) for url in data['url']]
    extract_url_data = pd.DataFrame(extract_url_data, columns=['subdomain', 'domain', 'domain_suffix'])
    # concat extracted feature with main data
    data = data.reset_index(drop=True)
    data = pd.concat([data, extract_url_data], axis=1)
    return data
train_data = extract_url(train_data)
val_data = extract_url(val_data)
print(val_data.head())
def plot(train_data, val_data, column):
    plt.figure(figsize=(10, 17))
    plt.subplot(411)
    plt.title(f'Train data {column}')
    plt.ylabel(column)
    train_data[column].value_counts().head(10).plot.barh()
    plt.subplot(412)
    plt.title(f'Validation data {column}')
    plt.ylabel(column)
    val_data[column].value_counts().head(10).plot.barh()
    plt.subplot(413)
    plt.title(f'Train data {column} (groupped)')
    plt.ylabel(f'(label, {column})')
    train_data.groupby('label')[column].value_counts().head(10).plot.barh()
    plt.subplot(414)
    plt.title(f'Validation data {column} (groupped)')
    plt.ylabel(f'(label, {column})')
    val_data.groupby('label')[column].value_counts().head(10).plot.barh()
    plt.show()
plot(train_data, val_data, 'subdomain')
plot(train_data, val_data, 'domain')
plot(train_data, val_data, 'domain_suffix')
train_data[(train_data['domain'] == 'google') & (train_data['label'] == 'bad')].head()
train_data[(train_data['domain'] == 'twitter') & (train_data['label'] == 'bad')].head()
tokenizer = Tokenizer(filters='', char_level=True, lower=False, oov_token=1)
# fit only on training data
tokenizer.fit_on_texts(train_data['url'])
n_char = len(tokenizer.word_index.keys())
print(f'N Char: {n_char}')
train_seq = tokenizer.texts_to_sequences(train_data['url'])
val_seq = tokenizer.texts_to_sequences(val_data['url'])
print('Before tokenization: ')
print(train_data.iloc[0]['url'])
print('\nAfter tokenization: ')
print(tokenizer.word_index)
print(val_seq[0])

sequence_length = np.array([len(i) for i in train_seq])
sequence_length = np.percentile(sequence_length, 99).astype(int)
print(f'Sequence length: {sequence_length}')
train_seq = pad_sequences(train_seq, padding='post', maxlen=sequence_length)
val_seq = pad_sequences(val_seq, padding='post', maxlen=sequence_length)
print('After padding: ')
print(train_seq[0])
print(val_seq[0])
train_seq = train_seq / n_char
val_seq = val_seq / n_char
with open('tokenizer.josn', 'wb') as f:
    pickle.dump(tokenizer, f)
    
   
def encode_label(label_index, data):
    try:
        return label_index[data]
    except:
        return label_index['<unknown>']
unique_value = {}
for feature in ['subdomain', 'domain', 'domain_suffix']:
    # get unique value
    label_index = {label: index for index, label in enumerate(train_data[feature].unique())}
    # add unknown label in last index
    label_index['<unknown>'] = list(label_index.values())[-1] + 1
    # count unique value
    unique_value[feature] = label_index['<unknown>']
    # encode
    train_data.loc[:, feature] = [encode_label(label_index, i) for i in train_data.loc[:, feature]]
    val_data.loc[:, feature] = [encode_label(label_index, i) for i in val_data.loc[:, feature]]

    # save label index
    with open(f'{feature}.pkl', 'wb') as f:
        pickle.dump(label_index, f)

    with open(f'{feature}.json','w') as file:
        json.dump(label_index,file)
with open('domain_suffix.pkl', 'rb') as f:
    data = pickle.load(f)
for data in [train_data, val_data]:
    data.loc[:, 'label'] = [0 if i == 'good' else 1 for i in data.loc[:, 'label']]
print(f"Unique subdomain in Train data: {unique_value['subdomain']}")
print(f"Unique domain in Train data: {unique_value['domain']}")
print(f"Unique domain suffix in Train data: {unique_value['domain_suffix']}")
def convolution_block(x):
    # 3 sequence conv layer
    conv_3_layer = layers.Conv1D(64, 3, padding='same', activation='elu')(x)
    # 5 sequence conv layer
    conv_5_layer = layers.Conv1D(64, 5, padding='same', activation='elu')(x)
    # concat conv layer
    conv_layer = layers.concatenate([x, conv_3_layer, conv_5_layer])
    # flatten
    conv_layer = layers.Flatten()(conv_layer)
    return conv_layer
def embedding_block(unique_value, size):
    input_layer = layers.Input(shape=(1,))
    embedding_layer = layers.Embedding(unique_value, size, input_length=1)(input_layer)
    return input_layer, embedding_layer
def create_model(sequence_length, n_char, n_subdomain, n_domain, n_domain_suffix):
    input_layer = []
    # sequence input layer
    sequence_input_layer = layers.Input(shape=(sequence_length,))
    input_layer.append(sequence_input_layer)
    # convolution block
    char_embedding = layers.Embedding(n_char + 1, 32, input_length=sequence_length)(sequence_input_layer)
    conv_layer = convolution_block(char_embedding)
    # entity embedding
    entity_embedding = []
    for n in [n_subdomain, n_domain,n_domain_suffix]:
        size = 4
        input_l, embedding_l = embedding_block(n, size)
        embedding_l = layers.Reshape(target_shape=(size,))(embedding_l)
        input_layer.append(input_l)
        entity_embedding.append(embedding_l)
    # concat all layer
    fc_layer = layers.concatenate([conv_layer, *entity_embedding])
    fc_layer = layers.Dropout(rate=0.5)(fc_layer)
    # dense layer
    fc_layer = layers.Dense(128, activation='elu')(fc_layer)
    fc_layer = layers.Dropout(rate=0.2)(fc_layer)
    # output layer
    output_layer = layers.Dense(1, activation='sigmoid')(fc_layer)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrics.Precision(), metrics.Recall()])
    return model
# reset session
backend.clear_session()
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)
# create model
model = create_model(sequence_length, n_char, unique_value['subdomain'], unique_value['domain'], unique_value['domain_suffix'])
model.summary()
plot_model(model, to_file='model.png')
model_image = mpimg.imread('model.png')
plt.figure(figsize=(75, 75))
plt.imshow(model_image)
plt.show()
train_x = []
train_y = train_data['label']
val_x = [val_seq, val_data['subdomain'], val_data['domain'], val_data['domain_suffix']]
val_y = val_data['label']
train_x = [train_seq, train_data['subdomain'], train_data['domain'], train_data['domain_suffix']]
train_y = train_data['label'].values

early_stopping = [EarlyStopping(monitor='val_precision', patience=5, restore_best_weights=True, mode='max')]
history = model.fit(train_x, train_y, batch_size=64, epochs=25, verbose=1, validation_split=0.2, shuffle=True, callbacks=early_stopping)
model.save('model.h5')
plt.figure(figsize=(20, 5))
for index, key in enumerate(['loss', 'precision', 'recall']):
    plt.subplot(1, 3, index+1)
    plt.plot(history.history[key], label=key)
    plt.plot(history.history[f'val_{key}'], label=f'val {key}')
    plt.legend()
    plt.title(f'{key} vs val {key}')
    plt.ylabel(f'{key}')
    plt.xlabel('epoch')
val_pred = model.predict(val_x)
data = parsed_url("drive.google.com/uc?export=download&amp;id=0B7XzN8DNbJKiQlFNRHdVTmpCd0U")
print(data)
print(encode_label(label_index,'com'))
print(encode_label(label_index,'drive'))
print(encode_label(label_index,'google'))

data = parsed_url("en.wikipedia.org/wiki/Claude_Lemieux	")
print(data)
print(encode_label(label_index,'en'))
print(encode_label(label_index,'wikipedia'))
print(encode_label(label_index,'org'))
val_pred = model.predict(val_x)

val_pred = np.where(val_pred[:, 0] >= 0.5, 1, 0)
print(f'Validation Data:\n{val_data.label.value_counts()}')
print(f'\n\nConfusion Matrix:\n{confusion_matrix(val_y, val_pred)}')
print(f'\n\nClassification Report:\n{classification_report(val_y, val_pred)}')