import numpy as np
import pandas as pd
import gensim
import string

from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.data_utils import get_file

import nltk
data = pd.read_csv("../input/Womens Clothing E-Commerce Reviews.csv")
data[:2]
print("Preparing text")
reviews = data["Review Text"].dropna()
tokens = [[word for word in nltk.word_tokenize(doc.lower()) if word] for doc in reviews]
# from this gist: https://gist.github.com/maxim5/c35ef2238ae708ccb0e55624e9e0252b
print('Training word2vec')
word_model = gensim.models.Word2Vec(tokens, size=25, min_count=1, window=5, iter=100)
pretrained_weights = word_model.wv.syn0
vocab_size, embedding_size = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)
print('Checking similar words:')
for word in ['dress', 'tall', 'return', 'sweater']:
    most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.wv.most_similar(word)[:8])
    print('  %s -> %s' % (word, most_similar))
model.save("word2vec-clothes-25d.model")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation

N_TOPICS = 10

vec = TfidfVectorizer(stop_words = nltk.corpus.stopwords.words('english'), ngram_range=(1,3), min_df=5, max_df = 0.9)
lda = LatentDirichletAllocation(n_components = N_TOPICS, )

pipeline = Pipeline([("vec", vec), ("lda", lda)])
pipeline.fit(reviews)
vec_model = pipeline.steps[0][1]
lda_model = pipeline.steps[1][1]
from sklearn.metrics import homogeneity_score

cluster_labels = np.argmax(pipeline.transform(reviews), axis=1)
homogeneity_score(data.loc[reviews.index, "Rating"], cluster_labels)
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print("Mean: {:.2f}".format(data.loc[np.where(cluster_labels == topic_idx)[0], "Rating"].mean()))
        print("Std: {:.2f}".format(data.loc[np.where(cluster_labels == topic_idx)[0], "Rating"].std()))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
display_topics(lda_model, vec_model.get_feature_names(), no_top_words)
# some utility code
def word2idx(word):
    return word_model.wv.vocab[word].index
def idx2word(idx):
    return word_model.wv.index2word[idx]
# get mean word2vec of each sentence
X_averaged_word2vec = [np.mean([word_model.wv.word_vec(word) for word in sentence], axis=0) for sentence in tokens]
X_averaged_word2vec = np.array(X_averaged_word2vec)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters= N_TOPICS)
kmeans.fit(X_averaged_word2vec)
from sklearn.metrics import homogeneity_score

cluster_labels = kmeans.predict(X_averaged_word2vec)
homogeneity_score(data.loc[reviews.index, "Rating"], cluster_labels)
stopwords = nltk.corpus.stopwords.words('english')

def get_freq_per_topic(topic_idx):
    topic_sentences = reviews.reindex(np.where(cluster_labels == topic_idx)[0])
    topic_words = [nltk.word_tokenize(v) for v in topic_sentences.dropna()]
    flattened_topic_words = [word.lower() for sublist in topic_words for word 
                             in sublist if len(word) >= 3 and word not in stopwords]
    return pd.Series(nltk.FreqDist(flattened_topic_words))
    
# frequency of all words across all reviews
all_words = [nltk.word_tokenize(v) for v in reviews.dropna()]
flattened_all_words = [word.lower() for sublist in all_words 
                       for word in sublist if len(word) >= 3 and word not in stopwords]
all_words_frequency = pd.Series(nltk.FreqDist(flattened_all_words))
list_series_frequencies = [get_freq_per_topic(v) for v in range(N_TOPICS)]
for cluster in range(N_TOPICS):
    top_words_cluster = list_series_frequencies[cluster].sort_values(ascending=False)[:10000]
    uniqueness_index = top_words_cluster.div(all_words_frequency, ).dropna().sort_values(ascending=False)
    uniqueness_index = uniqueness_index[(uniqueness_index > 0.2)]
    top_20 = list_series_frequencies[cluster].reindex(uniqueness_index.index)\
                                            .sort_values(ascending=False).index.tolist()[:20]
    print("--- Cluster", cluster, "---")
    print("Mean: {:.2f}".format(data.loc[np.where(cluster_labels == cluster)[0], "Rating"].mean()))
    print("Std: {:.2f}".format(data.loc[np.where(cluster_labels == cluster)[0], "Rating"].std()))
    print(top_20)
%%time
print('\nPreparing the data for LSTM...')

maxlen = max([len(v) for v in tokens])
TIMESTEPS = 100

# train_x = np.zeros([len(tokens), maxlen], dtype=np.int32)
# train_y = np.zeros([len(tokens)], dtype=np.int32)

list_x = []
list_y = []

for _, sentence in enumerate(tokens):
    for i in range(len(sentence)):
        if len(sentence) <= i + TIMESTEPS:
            # the last word is the target
            input_words = sentence[:-1]
            target_word = sentence[-1]
            
            list_input_words = []
            for t, word in enumerate(input_words):
                list_input_words.append(word2idx(word))
            list_x.append(list_input_words)
            list_y.append(word2idx(target_word))
            break
        else:
            input_words = sentence[i:i+TIMESTEPS]
            target_word = sentence[i+TIMESTEPS]
            
            list_input_words = []
            for t, word in enumerate(input_words):
                list_input_words.append(word2idx(word))
            list_x.append(list_input_words)
            list_y.append(word2idx(target_word))
from keras.preprocessing.sequence import pad_sequences

x_indexes = pad_sequences(list_x, maxlen=TIMESTEPS, padding='post')
y_indexes = np.array(list_y)
from sklearn.model_selection import train_test_split

# Train/Test splits
X_train, X_test, y_train, y_test = train_test_split(x_indexes, y_indexes, test_size=0.4)
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed
from keras.models import Model

LSTM_SIZE = 50

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, 
                    weights=[pretrained_weights]), )
model.add(LSTM(units=LSTM_SIZE))
model.add(Dense(units=vocab_size))
model.add(Activation('softmax'))
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

# # autoencoder model
# model = Sequential()
# model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[pretrained_weights], input_length=TIMESTEPS))
# model.add(LSTM(units=LSTM_SIZE))
# model.add(RepeatVector(TIMESTEPS))
# model.add(LSTM(embedding_size, return_sequences=True))
# model.add(LSTM(embedding_size, ))
# model.add(TimeDistributed(Dense(1)))

# model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae', 'mse'])
model.summary()
def sample(preds, temperature=1.0):
    if temperature <= 0:
        return np.argmax(preds)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_next(text, num_generated=10):
    word_idxs = [word2idx(word) for word in text.lower().split()]
    for i in range(num_generated):
        prediction = model.predict(x=np.array(word_idxs))
        idx = sample(prediction[-1], temperature=0.7)
        word_idxs.append(idx)
    return ' '.join(idx2word(idx) for idx in word_idxs)

def on_epoch_end(epoch, _):
    print('\nGenerating text after epoch: %d' % epoch)
    texts = [
    'it\'s a very',
    'i',
    'this is',
    'i wanted',
    'again'
    ]
    for text in texts:
        sample = generate_next(text)
        print('%s... -> %s' % (text, sample))

from keras.callbacks import LambdaCallback

history = model.fit(X_train, y_train, validation_split= 0.2,
                  batch_size=2048,
                  epochs=100,
                  callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])

# import keras.backend as K

# K.clear_session()