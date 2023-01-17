import numpy as np

import regex as re

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statistics

import math

import os



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



import tensorflow as tf

import tensorflow.keras.backend as K

import tokenizers

from transformers import RobertaTokenizer, TFRobertaModel



from collections import Counter



import warnings

warnings.filterwarnings("ignore")
# Detect hardware, return appropriate distribution strategy (you can see that it is pretty easy to set up).

try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is set (always set in Kaggle)

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    print('Running on TPU ', tpu.master())

except ValueError:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print('Number of replicas:', strategy.num_replicas_in_sync)
MODEL_NAME = 'roberta-base'

MAX_LEN = 256

ARTIFACTS_PATH = '../artifacts/'



BATCH_SIZE = 8 * strategy.num_replicas_in_sync

EPOCHS = 3



if not os.path.exists(ARTIFACTS_PATH):

    os.makedirs(ARTIFACTS_PATH)
df = pd.read_csv('/kaggle/input/bbc-articles-cleaned/tfidf_dataset.csv')

df.head()
X_data = df[['text']].to_numpy().reshape(-1)

y_data = df[['category']].to_numpy().reshape(-1)
categories = df[['category']].values.reshape(-1)



counter_categories = Counter(categories)

category_names = counter_categories.keys()

category_values = counter_categories.values()



y_pos = np.arange(len(category_names))



plt.figure(1, figsize=(10, 5))

plt.bar(y_pos, category_values, align='center', alpha=0.5)

plt.xticks(y_pos, category_names)

plt.ylabel('Number of texts')

plt.title('Distribution of texts per category')

plt.gca().yaxis.grid(True)

plt.show()



print(counter_categories)
def calculate_stats(df, split_char=' '):

    categories = df['category'].unique()

    

    all_lengths = []

    per_category = {

        'lengths': {c:[] for c in categories},

        'mean': {c:0 for c in categories},

        'stdev': {c:0 for c in categories}

    }



    for index, row in df.iterrows():

        text = row['text']

        text = re.sub(r"\s+", ' ', text) # Normalize

        text = text.split(split_char)

        l = len(text)

        

        category = row['category']

        

        all_lengths.append(l)

        per_category['lengths'][category].append(l)

    

    for c in categories:

        per_category['mean'][c] = statistics.mean(per_category['lengths'][c])

        per_category['stdev'][c] = statistics.stdev(per_category['lengths'][c])

    

    global_stats = {

        'mean': statistics.mean(all_lengths),

        'stdev': statistics.stdev(all_lengths),

        'lengths': all_lengths

    }

    

    return {

        'global': global_stats,

        'per_category': pd.DataFrame(per_category)

    }





def display_lengths_histograms(df_stats, n_cols=3):

    categories = df['category'].unique()

    n_rows = math.ceil(len(categories) / n_cols)

    

    plt.figure(figsize=(15, 8))

    plt.suptitle('Distribution of lengths')

    

    # Subplot of all lengths

    plt.subplot(n_rows, n_cols, 1)

    plt.title('All categories')

    lengths = df_stats['global']['lengths']

    plt.hist(lengths, color='r')



    # Subplot of each category

    index_subplot = 2

    for c in categories:

        plt.subplot(n_rows, n_cols, index_subplot)

        plt.title('Category: %s' % c)

        

        lengths = df_stats['per_category']['lengths'][c]

        plt.hist(lengths, color='b')



        index_subplot += 1



    plt.show()
df_stats = calculate_stats(df)

df_stats['per_category']
display_lengths_histograms(df_stats)
n_texts = len(X_data)

print('Texts in dataset: %d' % n_texts)



categories = df['category'].unique()

n_categories = len(categories)

print('Number of categories: %d' % n_categories)



print('Done!')
def roberta_encode(texts, tokenizer):

    ct = len(texts)

    input_ids = np.ones((ct, MAX_LEN), dtype='int32')

    attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')

    token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32') # Not used in text classification



    for k, text in enumerate(texts):

        # Tokenize

        tok_text = tokenizer.tokenize(text)

        

        # Truncate and convert tokens to numerical IDs

        enc_text = tokenizer.convert_tokens_to_ids(tok_text[:(MAX_LEN-2)])

        

        input_length = len(enc_text) + 2

        input_length = input_length if input_length < MAX_LEN else MAX_LEN

        

        # Add tokens [CLS] and [SEP] at the beginning and the end

        input_ids[k,:input_length] = np.asarray([0] + enc_text + [2], dtype='int32')

        

        # Set to 1s in the attention input

        attention_mask[k,:input_length] = 1



    return {

        'input_word_ids': input_ids,

        'input_mask': attention_mask,

        'input_type_ids': token_type_ids

    }
# Transform categories into numbers

category_to_id = {}

category_to_name = {}



for index, c in enumerate(y_data):

    if c in category_to_id:

        category_id = category_to_id[c]

    else:

        category_id = len(category_to_id)

        category_to_id[c] = category_id

        category_to_name[category_id] = c

    

    y_data[index] = category_id



# Display dictionary

category_to_name
# Split into train and test datasets

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=777) # random_state to reproduce results
# Import tokenizer from HuggingFace

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
X_train = roberta_encode(X_train, tokenizer)

X_test = roberta_encode(X_test, tokenizer)



y_train = np.asarray(y_train, dtype='int32')

y_test = np.asarray(y_test, dtype='int32')
def build_model(n_categories):

    with strategy.scope():

        input_word_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_word_ids')

        input_mask = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_mask')

        input_type_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_type_ids')



        # Import RoBERTa model from HuggingFace

        roberta_model = TFRobertaModel.from_pretrained(MODEL_NAME)

        x = roberta_model(input_word_ids, attention_mask=input_mask, token_type_ids=input_type_ids)



        # Huggingface transformers have multiple outputs, embeddings are the first one,

        # so let's slice out the first position

        x = x[0]



        x = tf.keras.layers.Dropout(0.1)(x)

        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(256, activation='relu')(x)

        x = tf.keras.layers.Dense(n_categories, activation='softmax')(x)



        model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=x)

        model.compile(

            optimizer=tf.keras.optimizers.Adam(lr=1e-5),

            loss='sparse_categorical_crossentropy',

            metrics=['accuracy'])



        return model
with strategy.scope():

    model = build_model(n_categories)

    model.summary()
with strategy.scope():

    print('Training...')

    history = model.fit(X_train,

                        y_train,

                        epochs=EPOCHS,

                        batch_size=BATCH_SIZE,

                        verbose=1,

                        validation_data=(X_test, y_test))
# This plot will look much better if we train models with more epochs, but anyway here is

plt.figure(figsize=(10, 10))

plt.title('Accuracy')



xaxis = np.arange(len(history.history['accuracy']))

plt.plot(xaxis, history.history['accuracy'], label='Train set')

plt.plot(xaxis, history.history['val_accuracy'], label='Validation set')

plt.legend()
def plot_confusion_matrix(X_test, y_test, model):

    y_pred = model.predict(X_test)

    y_pred = [np.argmax(i) for i in model.predict(X_test)]



    con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()



    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    label_names = list(range(len(con_mat_norm)))



    con_mat_df = pd.DataFrame(con_mat_norm,

                              index=label_names, 

                              columns=label_names)



    figure = plt.figure(figsize=(10, 10))

    sns.heatmap(con_mat_df, cmap=plt.cm.Blues, annot=True)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1] * 100))
plot_confusion_matrix(X_test, y_test, model)