!pip install tensorflow_datasets

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import spacy

import re

import nltk

import seaborn as sns



import tensorflow as tf

import tensorflow_datasets as tfds

from tensorflow import keras

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import LabelEncoder

from collections import defaultdict

from collections import Counter



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn import model_selection, naive_bayes

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import MaxAbsScaler

from scipy import stats

from sklearn.metrics import make_scorer, roc_auc_score

from scipy.linalg import svd

from numpy import diag

from scipy.sparse import csr_matrix



from numpy import zeros

from sklearn import svm



from nlp_functions import *

np.random.seed(500)
def encode_train(text_tensor, label):

    text = text_tensor.numpy()[0]

    encoded_text = encoder.encode(text)

    return encoded_text, label



def encode_map_fn_train(text, label):

    return tf.py_function(encode_train, inp=[text, label], Tout=(tf.int64, tf.int64))



def encode_unseen(text_tensor):

    text = text_tensor.numpy()[0]

    encoded_text = encoder.encode(text)

    return encoded_text, 1



def encode_map_fn_unseen(text):

    return tf.py_function(encode_unseen, inp=[text], Tout=(tf.int64, 1))
disaster_original = pd.read_csv('../input/nlp-getting-started/train.csv')

disaster_original.head(5)
disaster_original['target'].unique()
disaster_original.shape
unseen_data = pd.read_csv('../input/nlp-getting-started/test.csv')

unseen_id = pd.read_csv('../input/nlp-getting-started/test.csv')
disaster_original = drop_columns(disaster_original)

unseen_data = unseen_data.drop(columns=['id', 'keyword', 'location'])

disaster_original.head(5)
unseen_data['text'] = unseen_data['text'].replace("", "empty")
disaster_tweets = disaster_original.copy()



plt.figure(figsize=(10, 5))

plt.hist(disaster_tweets['target'])

plt.title('Histogram of Target Variable')

plt.xlabel('No / Yes')

plt.ylabel('Frequency of target value')

plt.show()
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



for train_index, test_index in split.split(disaster_tweets, disaster_tweets['target']):

    

    strat_training_set = disaster_tweets.loc[train_index]

    strat_testing_set = disaster_tweets.loc[test_index]
plt.figure(figsize=(10, 5))

plt.hist(strat_testing_set['target'])

plt.xlabel('No / Yes')

plt.ylabel('Frequency of target value')

plt.title('Histogram of Target value - Stratified Test Set')

plt.show()
plt.figure(figsize=(10, 5))

plt.hist(strat_training_set['target'])

plt.xlabel('No / Yes')

plt.ylabel('Frequency of target value')

plt.title('Histogram of Target value - Stratified Train Set')

plt.show()
strat_training_set = pre_process(strat_training_set)

strat_testing_set = pre_process(strat_testing_set)

unseen_data = pre_process(unseen_data)
X_train = strat_training_set.drop(columns=['target'])

Y_train = strat_training_set['target']



X_test = strat_testing_set.drop(columns=['target'])

Y_test = strat_testing_set['target']
disaster_tweets = pre_process(disaster_tweets)



tf_idf_vect = TfidfVectorizer(max_features=300, sublinear_tf=True)

tf_idf_vect.fit(disaster_tweets['text'])



X_train_tfidf = tf_idf_vect.transform(X_train['text'])

X_test_tfidf = tf_idf_vect.transform(X_test['text'])
tSVD = TruncatedSVD(n_components=3)

data_3d = tSVD.fit_transform(X_train_tfidf)



svd_df = pd.DataFrame()

svd_df['svd_one'] = data_3d[:, 0]

svd_df['svd_two'] = data_3d[:, 1]

svd_df['svd_three'] = data_3d[:, 2]

plot_vectors(svd_df, strat_training_set)
nb = naive_bayes.MultinomialNB()

nb.fit(X_train_tfidf, Y_train)
nb_validation_predictions = nb.predict(X_test_tfidf)

nb_training_predictions = nb.predict(X_train_tfidf)
print(f"Naive Bayes Basline Validation Accuracy: {accuracy_score(nb_validation_predictions, Y_test) * 100}")

print(f"Naive Bayes Basline Training Accuracy: {accuracy_score(nb_training_predictions, Y_train) * 100}")
scaler = MaxAbsScaler()

X_train_svm_scaled = scaler.fit_transform(X_train_tfidf)

X_test_svm_scaled = scaler.fit_transform(X_test_tfidf)
Y_train_svm = pd.Series(np.where(Y_train == 0, -1, 1))

Y_test_svm = pd.Series(np.where(Y_test == 0, -1, 1))
svm_ = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

svm_.fit(X_train_svm_scaled, Y_train_svm)
svm_validation_prediction = svm_.predict(X_test_svm_scaled)

svm_training_predictions = svm_.predict(X_train_svm_scaled)
print(f"Support Vector Machines - Training Accuracy: {accuracy_score(svm_training_predictions, Y_train_svm)}")

print(f"Support Vector Machines - Validation Accuract: {accuracy_score(svm_validation_prediction, Y_test_svm)}")
svm_rand_opt_2 = svm.SVC(C=3.2291456156839677,

                      gamma=0.4856666290873001,

                        tol=0.7120239961045746,

                        kernel='rbf')

svm_rand_opt_2.fit(X_train_svm_scaled, Y_train_svm)
svm_rand_valid_predictions = svm_rand_opt_2.predict(X_test_svm_scaled)

svm_rand_train_predictions = svm_rand_opt_2.predict(X_train_svm_scaled)
print(f"Support Vector Machines - Training Accuracy: {accuracy_score(svm_rand_train_predictions, Y_train_svm)}")

print(f"Support Vector Machines - Validation Accuract: {accuracy_score(svm_rand_valid_predictions, Y_test_svm)}")
unseen_text = unseen_data['text']

disaster_text = disaster_original['text']

all_tweets_concat = pd.concat([unseen_text, disaster_text], axis=0)
target = disaster_tweets.pop('target')

tweets_raw = tf.data.Dataset.from_tensor_slices((disaster_tweets.values, target.values))

tweets_unseen = tf.data.Dataset.from_tensor_slices((unseen_data.values))

unseen_raw_all = tf.data.Dataset.from_tensor_slices((all_tweets_concat.values))
for tw in tweets_raw.take(3):

    tf.print(tw[0].numpy()[0][ :50], tw[1])
tweets_raw = tweets_raw.shuffle(7613, reshuffle_each_iteration=False)

tweets_valid = tweets_raw.take(2283)

tweets_train = tweets_raw.skip(2283)
tokeniser = tfds.features.text.Tokenizer()

token_counts = Counter()



for example in unseen_raw_all:

    tokens = tokeniser.tokenize(example.numpy())

    token_counts.update(tokens)
print(f"Vocab SIze: {len(token_counts)}")
encoder = tfds.features.text.TokenTextEncoder(token_counts)

example_string = "This is an example"

print(f"Exmaple String: {example_string}")

print(f"Encoded String: {encoder.encode(example_string)}")
tweets_train = tweets_train.map(encode_map_fn_train)

tweets_valid = tweets_valid.map(encode_map_fn_train)

tweets_unseen_map = tweets_unseen.map(encode_map_fn_unseen)
print("Example Sequences and their length:\n")

example = tweets_train.take(8)

for ex in example:

    print(f"Individual Size: {ex[0].shape}")
print("Batched examples and the sequence length:\n")

batched_example = example.padded_batch(4, padded_shapes=([-1], []))

for batch in batched_example:

    print(f"Batch dimension: {batch[0].shape}")
tweets_train = tweets_train.padded_batch(32, padded_shapes=([-1], []))

tweets_valid = tweets_valid.padded_batch(32, padded_shapes=([-1], []))

tweets_unseen_batched = tweets_unseen_map.padded_batch(32, padded_shapes=([-1], []))
embedding_dimension = 200

vocab_size = len(token_counts) + 2



tf.random.set_seed(42)



bi_lstm_model = tf.keras.Sequential([

    

    tf.keras.layers.Embedding(

        

        input_dim=vocab_size,

        output_dim=embedding_dimension,

        name='embed-layer'),

    

    tf.keras.layers.Bidirectional(

        

        tf.keras.layers.LSTM(64, name='lstm-layer'),

        name='bidir-lstm'),

    

    

    

    tf.keras.layers.Dense(64, activation='relu'),

    

    tf.keras.layers.Dropout(0.2),

    

    tf.keras.layers.Dense(1, activation='sigmoid')

    

])
bi_lstm_model.summary()
bi_lstm_model.compile(optimizer=tf.keras.optimizers.Nadam(1e-3),

                     loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),

                     metrics=['accuracy'])
history = bi_lstm_model.fit_generator(tweets_train, validation_data=tweets_valid,

                           epochs=50, callbacks=[keras.callbacks.EarlyStopping(patience=8)])
predictions = history.model.predict_classes(tweets_unseen_batched, batch_size=None)
output_df = pd.DataFrame(unseen_id['id'])

output_df['target'] = predictions
out = output_df.to_csv('RNN.csv')