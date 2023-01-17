# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/explicit-content-detection/train.csv")
test_df = pd.read_csv("/kaggle/input/explicit-content-detection/test.csv")

titles = train_df["title"].values
urls = train_df["url"].values
y = train_df["target"].astype(int).values
train_df.head()
from string import punctuation
import nltk
# Define symbols & words we don't need
translator = {ord(c): ' ' for c in punctuation + '0123456789«»—–…✅№'}
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('russian') + nltk.corpus.stopwords.words('english')
print(stop_words[:10])
'''
Remove stop words
'''
def remove_stops(collection):
    return [w for w in collection if w not in stop_words]


'''
Split titles array into tokens
'''
def split_into_tokens(X):
    n_samples = X.size
    tokens = []

    print('Splitting...')
    for i in range(n_samples):
        tokens.append(remove_stops(X[i].translate(translator).lower().split()))

        if i % 10000 == 0:
            print(f'Done: {i}/{n_samples}')

    print(f'Done: {n_samples}/{n_samples}\n')
    return tokens


from nltk.stem.snowball import SnowballStemmer 

'''
Do stemming with splitted word tokens
'''
def do_stemming(X):
    n_samples = len(X)
    stemmer = SnowballStemmer("russian")
    stemmed = []

    print('Stemming...')
    for i in range(n_samples):
        stemmed.append(list(map(stemmer.stem, X[i])))

        if i % 10000 == 0:
            print(f'Done: {i}/{n_samples}')

    print(f'Done: {n_samples}/{n_samples}\n')
    return stemmed
from sklearn.model_selection import train_test_split
# Split data into test and train sets
train_titles, test_titles, train_urls, test_urls, train_y, test_y = train_test_split(titles, urls, y, test_size=0.33, stratify=y)
n_samples = len(train_y)
X_stemmed_tokens = do_stemming(split_into_tokens(train_titles))
from collections import Counter
'''
Create corpus of all words
'''
def make_corpus(X, y):
    corpus_all = []
    corpus_porn = []

    for sample, is_porn in zip(X, y):
        corpus_all.extend(set(sample))

        if is_porn:
            corpus_porn.extend(set(sample))
            
    return corpus_all, corpus_porn
# Count words
# We need to do this to calculate importance of each word
corpus_titles_all, corpus_titles_porn = make_corpus(X_stemmed_tokens, train_y)
count_titles_all = Counter(corpus_titles_all)
count_titles_porn = Counter(corpus_titles_porn)

# Count urls
corpus_urls_porn = [train_urls[i] for i in range(n_samples) if train_y[i]]
count_urls_all = Counter(train_urls)
count_urls_porn = Counter(corpus_urls_porn)
'''
Entropy of word
We have to estimate, how valuable deviation of given porn share from overall mean 
'''
def word_porn_rate(porn_share, mean_share, penalize):
    if porn_share > mean_share:
        return (porn_share - mean_share) / (1 - mean_share)
    
    fine = mean_share / (1 - mean_share) if penalize else 1
    return (porn_share - mean_share) * fine / mean_share
'''
Define word score based on it's frequency and entropy
'''
def word_score(porn_rate, frequency, f_weight = 0.3):
    return ((frequency * f_weight) + np.abs(porn_rate) * (1 - f_weight)) / 2

'''
Get n best words based on word_score
'''
def get_n_best_words(n, x_size, mean_share, counter_porn, counter_all, min_freq=0.001, f_weight=0.3, penalize=True):
    word_scores = {}
    word_weights = {}
    
    # Evaluate scores for each popular word
    for word, count in counter_all.items():
        if count/x_size >= min_freq:
            porn_count = counter_porn[word]
            porn_share = porn_count / count
            frequency = count / x_size
            porn_rate = word_porn_rate(porn_share, mean_share, penalize)
            word_weights[word] = porn_rate
            word_scores[word] = word_score(porn_rate, frequency, f_weight)
            
    # Get n best words (or all words we have, if n is too big)
    best_words, best_weights = [], []
    
    for word in sorted(word_scores.keys(), key=lambda w: word_scores[w], reverse=True)[:n]:
        best_words.append(word)
        best_weights.append(word_weights[word])
            
    return best_words, np.array(best_weights)
mean = np.mean(train_y)
best_words, best_word_weights = get_n_best_words(100, n_samples, mean, count_titles_porn, count_titles_all, min_freq=0.001, f_weight=0.3)
best_urls, best_url_weights = get_n_best_words(100000, n_samples, mean, count_urls_porn, count_urls_all, min_freq=0.0001, f_weight=0.3, penalize=False)
print('Best words:')
for i in range(70, 100):
    word = best_words[i]
    print(f'{word} : {best_word_weights[i]}, {count_titles_all[word]}')
    
print('\nBest urls:')
for i in range(30):
    url = best_urls[i]
    print(f'{url} : {best_url_weights[i]}, {count_urls_all[url]}')
'''
Transform the tokens array into vectors
'''
def tokens_to_vecs(tokens_X, n_samples, words, n_words):
    vectors_X = np.empty((n_samples, n_words))
    
    print('Vectorising...')
    for i in range(n_samples):
        tokens = tokens_X[i]
        vectors_X[i] = [int(word in tokens) for word in words]
        
        if i % 10000 == 0:
            print(f'Done: {i}/{n_samples}')
        
    print(f'Done: {n_samples}/{n_samples}')
    return vectors_X
'''
Make target prediction based on valuable words
'''
def vec_to_predict(tokens_vec, best_weights):
    return int(np.mean(tokens_vec * best_weights) > 0)
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# test_X_vecs = tokens_to_vecs(test_X, len(test_X), best_words, len(best_words))
# train_X_vecs = tokens_to_vecs(train_X, len(train_X), best_words, len(best_words))

# test_accuracies = []
# train_accuracies = []
# always_false = [0 for _ in test_X_vecs]
# xs = np.arange(10, 210, 10)
# const_accuracies = [1 - mean for i in xs]

# for i in xs:
#     test_predictions = [vec_to_predict(vec[:i], best_weights[:i]) for vec in test_X_vecs]
#     test_accuracies.append(accuracy_score(test_y, test_predictions))
    
# plt.plot(xs, test_accuracies, 'r', xs, const_accuracies, 'g')
# plt.plot(max(xs, key=lambda x: test_accuracies[int(x/10 - 1)]), max(test_accuracies), 'r^')
# plt.show()
# test_predictions = [vec_to_predict(vec[:70], best_weights[:70]) for vec in test_X_vecs]
# print(f'Accuracy: {accuracy_score(test_y, test_predictions)}')
X_urls = train_df["url"].values
# # Split data into test and train sets
train_X_urls, test_X_urls, train_y_urls, test_y_urls = train_test_split(X_urls, y_train, test_size=0.33, stratify=y_train)
porn_urls = [url for i, url in enumerate(train_X_urls) if train_y_urls[i]]

# Count urls
count_urls = Counter(train_X_urls)
count_urls_porn = Counter(porn_urls)
# Get valuable urls
mean = np.mean(train_y_urls)
x_size = len(train_X_urls)
best_urls, best_urls_weights = get_n_best_words(25000, x_size, mean, count_urls_porn, count_urls, min_freq=0.00001, f_weight=0.3, penalize=False)
len(best_urls)
for i in range(100):
    url = best_urls[i]
    print(f'{url}  count: {count_urls[url]}  share: {count_urls_porn[url]/count_urls[url]}  weight: {best_urls_weights[i]}')
def url_to_predict(url, urls, weights):
    try:
        return int(weights[urls.index(url)] > 0)
    except ValueError:
        return 0
# test_accuracies_urls = []
# xs = np.arange(10000, 25500, 500)

# for x in xs:
#     urls, weights = best_urls[:x], best_urls_weights[:x]
#     predictions = [url_to_predict(url, urls, weights) for url in test_X_urls]
#     test_accuracies_urls.append(accuracy_score(test_y_urls, predictions))
    
# plt.plot(xs, test_accuracies_urls, 'r')
# plt.show()
predictions = [url_to_predict(url, best_urls, best_urls_weights) for url in test_X_urls]
accuracy_score(test_y_urls, predictions)
# X_test_urls = test_df["url"].values
# n_samples = X_test_urls.size
# validate_predictions = [bool(url_to_predict(url, best_urls, best_urls_weights)) for url in X_test_urls]

# data = {
#     'id': [i for i in range(135309, 135309 + n_samples)],
#     'target': validate_predictions
# }

# validate_df = pd.DataFrame(data)
# validate_df.to_csv('simple_urls.csv', index=False)