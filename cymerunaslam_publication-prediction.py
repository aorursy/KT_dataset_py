# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#used libraries
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from textblob import TextBlob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#making sure plots work
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import os
import glob
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

one = pd.read_csv('/kaggle/input/articles1.csv')
two = pd.read_csv('/kaggle/input/articles2.csv')
three = pd.read_csv('/kaggle/input/articles3.csv')


#reading data
df = one.append([two, three])


    
print(df)    
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import pandas as pd

df_nyp = df[df['publication'] == 'New York Post']
df_nyp = df_nyp.sample(frac=0.50, random_state=42)

corpus = []
df_nyp['content'].apply(lambda x: corpus.append(x))

stop_words = stopwords.words("english")
additional_sw = ['say', 'said', 'one', 'two']
stop_words += additional_sw

def tokenize(content):
    letters_only = re.sub("[^a-zA-Z]",
                      " ",         
                      content )
    lower_case = letters_only.lower()
    tokens = word_tokenize(lower_case)
    words = [w for w in tokens if not w in stop_words]
    stems = [stemmer.lemmatize(word) for word in words]
    return(stems)

#create dictionary and incorporating 1 to n-gram 
stemmer = WordNetLemmatizer()
vectorizer = CountVectorizer(ngram_range = (1,2),
                             lowercase = True,
                             tokenizer=tokenize,
                             preprocessor = None,
                             max_features=5000)

#bag of words
bow = vectorizer.fit_transform(corpus)
vectorizer.vocabulary_

#term document matrix
tdm = pd.DataFrame(bow.toarray(), columns=vectorizer.get_feature_names())
df_nyp['word_count'] = df_nyp['content'].apply(lambda x : len(x.split()))
df_nyp['char_count'] = df_nyp['content'].apply(lambda x : len(x.replace(" ","")))

df_nyp.describe()

df_nyp.hist(column='word_count', bins=100)
df_nyp.hist(column='char_count', bins=100)

print('number of articles from the New York Post in set: ', df_nyp.shape[0])
df_content = df_nyp.sample(frac=0.03, random_state=42)
#sentiment
df_content['polarity'] = df_content['content'].map(lambda text: TextBlob(text).sentiment.polarity)

df_content['polarity'].iplot(
    kind='hist',
    bins=60,
    xTitle='polarity',
    linecolor='black',
    yTitle='count',
    title='Sentiment Polarity Distribution')
from sklearn.feature_extraction.text import CountVectorizer

def get_top_n_words(corpus, n=None):
    vector = CountVectorizer(tokenizer=tokenize).fit(corpus)
    bag_of_words = vector.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vector.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words = get_top_n_words(df_content['content'], 20)
data_content = pd.DataFrame(common_words, columns = ['text' , 'count'])
data_content.groupby('text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar',orientation='h', yTitle='Count', linecolor='black', title='Unigram - top 20 words in comments')
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), tokenizer=tokenize).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words = get_top_n_bigram(df_content['content'], 20)
data_content = pd.DataFrame(common_words, columns = ['text' , 'count'])
data_content.groupby('text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar',orientation='h', xTitle='Count', linecolor='black', title='Bigram - top 20 bigrams in comments')
def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), tokenizer=tokenize).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words = get_top_n_trigram(df_content['content'], 20)
data_comment = pd.DataFrame(common_words, columns = ['text' , 'count'])
data_comment.groupby('text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar',orientation='h', yTitle='Count', linecolor='black', title='Trigram - top 20 trigrams in comments')
from sklearn.decomposition import LatentDirichletAllocation

nb_topics = 15
lda = LatentDirichletAllocation(n_components=nb_topics, max_iter=20,
                                learning_method='batch')
document_topics = lda.fit_transform(bow)
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
feature_names = np.array(vectorizer.get_feature_names())

def print_topics(topics, feature_names, sorting, topics_per_chunk=6,
                 n_words=20):
    for i in range(0, len(topics), topics_per_chunk):
        these_topics = topics[i: i + topics_per_chunk]
        len_this_chunk = len(these_topics)
        words = []
        for i in range(n_words):
            try:
                words.append(feature_names[sorting[these_topics, i]])
            except:
                pass

    #setting up word dictionary for comparison
    word_dict = {}
    for i in topics:
        word_dict.update({i : [word[i] for word in words]})
    
    return word_dict

lda_topics = print_topics(topics=range(nb_topics), feature_names=feature_names, sorting=sorting, topics_per_chunk=nb_topics, n_words=15)

for i in range(nb_topics):
    print(i,': ' ,lda_topics[i], '\n')
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

svd = TruncatedSVD(n_components=nb_topics)
lsa = svd.fit_transform(bow)

normalizer = Normalizer()
lsa_norm = normalizer.fit_transform(lsa)

def print_top_words(model, feature_names, n_top_words):
    word_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        words = [feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]]

        word_dict.update({topic_idx : words})
#         print(message)
    return word_dict
    
n_top_words = 15
feature_names = vectorizer.get_feature_names()
svd_topics = print_top_words(svd, feature_names, n_top_words)

for i in range(nb_topics):
    print(i,': ' ,svd_topics[i], '\n')
from sklearn.decomposition import NMF

nmf = NMF(n_components=nb_topics)
nmf_data = nmf.fit_transform(bow)

normalizer = Normalizer()
nmf_norm = normalizer.fit_transform(nmf_data)

n_top_words = 15
feature_names = vectorizer.get_feature_names()
nmf_topics = print_top_words(nmf, feature_names, n_top_words)

for i in range(nb_topics):
    print(i,': ' ,nmf_topics[i], '\n')
def get_jaccard_sim(str1, str2):
    #get similairity between two topics (number of words present in both sets)
    a = set(str1)
    b = set(str2)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

# sim_matrix = []
# for lda_topic in range(len(lda_topics)):
#     sim_matrix.append([get_jaccard_sim(lda_topics[lda_topic], svd_topics[svd_topic]) for svd_topic in range(len(svd_topics))])
#the different number of topics
k_list = [3, 10, 25, 50]
dict_k = {}

for k in k_list:
    #SVD
    svd = TruncatedSVD(n_components=k)
    lsa = svd.fit_transform(bow)

    normalizer = Normalizer()
    lsa_norm = normalizer.fit_transform(lsa)
    
    #LDA
    lda = LatentDirichletAllocation(n_components=k, max_iter=20,
                                learning_method='batch')
    document_topics = lda.fit_transform(bow)
    
    #get data about topics
    lda_topics = print_topics(topics=range(k), 
                              feature_names=np.array(vectorizer.get_feature_names()), 
                              sorting=np.argsort(lda.components_, axis=1)[:, ::-1], 
                              topics_per_chunk=k, 
                              n_words=15)
    svd_topics = print_top_words(svd, vectorizer.get_feature_names(), 15)
    
    dict_k.update({k:[lsa_norm, document_topics]})
    
    #plot jaccard similairities matrix
    sim_matrix = []
    for lda_topic in range(len(lda_topics)):
        sim_matrix.append([get_jaccard_sim(lda_topics[lda_topic], 
                                           svd_topics[svd_topic]) for svd_topic in range(len(svd_topics))])

    plt.imshow(sim_matrix)
    plt.xlabel('LDA')
    plt.ylabel('SVD')
    plt.colorbar()
    plt.show()
    
#select k
nr_topics = 15 #k

#select topic from LDA
lda = LatentDirichletAllocation(n_components=nr_topics, max_iter=20,
                                learning_method='batch')
document_topics = lda.fit_transform(bow)

#find 10 articles with highest topic saturation
d = {}
for i in range(document_topics.shape[1]):
    d.update({i:[]})

for doc in document_topics:
    for i in range(len(doc)):
        d[i].append(doc[i])
    
df_text = pd.DataFrame(data=d)
df_text['text'] = corpus

sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
feature_names = np.array(vectorizer.get_feature_names())
topics_study = print_topics(topics=range(nr_topics), feature_names=feature_names, sorting=sorting, topics_per_chunk=nr_topics, n_words=10)

topic_nr = 13
topic_print = 0

for i in range(nr_topics):
    if 'yankee' in topics_study[i]:
        topic_print = i

df_text = df_text.sort_values(by=[topic_print], ascending=False)
df_text[:10]['text'].apply(lambda x: print(x, '\n'))

print(topics_study[topic_print])
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

#select another publication (opposing the previes chosen publication)
df_cnn = df[df['publication'] == 'CNN']
# df_cnn = df_cnn.sample(frac=1)

df_nyp = df[df['publication'] == 'New York Post']
df_nyp = df_nyp.sample(frac=0.65, random_state=42)

df_cnn['label'] = 1
df_nyp['label'] = 0

df_total = pd.concat([df_nyp, df_cnn])

df_sample = df_total.sample(frac=0.50, random_state=42)

X = df_sample['content']
y = df_sample['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import mglearn

pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
param_grid = {'logisticregression__C': [0.001, 0.01],
              "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]}
grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=5, verbose=1)
grid.fit(X_train, y_train)

print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters:\n{}".format(grid.best_params_))

# extract scores from grid_search
scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T

# visualize heat map
heatmap = mglearn.tools.heatmap(
    scores, xlabel="C", ylabel="ngram_range", cmap="viridis", fmt="%.3f",
    xticklabels=param_grid['logisticregression__C'],
    yticklabels=param_grid['tfidfvectorizer__ngram_range'])
plt.colorbar(heatmap)
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

#GridSearchCV on LatentDirichletAllocation Parameters
#grid search parameters has been optimized already
search_params = {'max_iter': [15], 
                 'n_components': [15]}

lda = LatentDirichletAllocation(
    learning_method='batch', random_state=0)

model = GridSearchCV(lda, param_grid=search_params)
model.fit(bow)
best_lda_model = model.best_estimator_

print("Best Model's Params: ", model.best_params_)
print("Best Log Likelihood Score: ", model.best_score_)
print("Model Perplexity: ", best_lda_model.perplexity(bow))
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

nb_iter = 15
nb_topics = 15
lda = LatentDirichletAllocation(n_components=nb_topics, max_iter=nb_iter,
                                learning_method='batch', random_state=0)

bow_train = vectorizer.fit_transform(X_train)
bow_test = vectorizer.transform(X_test)
X_train_lda = lda.fit_transform(bow_train)
X_test_lda = lda.transform(bow_test)
pipe = make_pipeline(RandomForestClassifier(max_depth=10, random_state=0))
param_grid = {
#                'randomforestclassifier__n_estimators': [30],
               'randomforestclassifier__max_features': ['auto', 'sqrt'],
                 'randomforestclassifier__max_depth': [5, 13, 20],
               'randomforestclassifier__min_samples_split': [5, 12, 17],
                'randomforestclassifier__min_samples_leaf': [1, 2, 5]
}

grid = GridSearchCV(pipe, param_grid, cv=3)
grid.fit(X_train_lda, y_train)

print(classification_report(grid.predict(X_test_lda), y_test))
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(LogisticRegression(class_weight='balanced'))
param_grid = {'logisticregression__C': np.linspace(10, 1000, num=100)}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train_lda, y_train)

print(classification_report(grid.predict(X_test_lda), y_test))
from sklearn.svm import LinearSVC

pipe = make_pipeline(LinearSVC(loss='hinge', penalty='l2'))
param_grid = {'linearsvc__C': np.linspace(10, 1000, num=100)}

grid = GridSearchCV(pipe, param_grid, cv=3)
grid.fit(X_train_lda, y_train)

print(classification_report(grid.predict(X_test_lda), y_test))
from sklearn.naive_bayes import MultinomialNB

pipe = make_pipeline(MultinomialNB())
param_grid = {'multinomialnb__alpha': np.linspace(0.0001, 1000, num=100)}

grid = GridSearchCV(pipe, param_grid, cv=3)
grid.fit(X_train_lda, y_train)

print(classification_report(grid.predict(X_test_lda), y_test))
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 2),
                                   lowercase=True, tokenizer=tokenize,
                                   preprocessor=None, max_features=10000)

X_train_tf = tfidf_vectorizer.fit_transform(X_train)
X_test_tf = tfidf_vectorizer.transform(X_test)
pipe = make_pipeline(RandomForestClassifier(max_depth=10, random_state=0))
param_grid = {
#                'randomforestclassifier__n_estimators': [30],
               'randomforestclassifier__max_features': ['auto', 'sqrt'],
                 'randomforestclassifier__max_depth': [5, 13, 20],
               'randomforestclassifier__min_samples_split': [5, 12, 17],
                'randomforestclassifier__min_samples_leaf': [1, 2, 5]
}

grid = GridSearchCV(pipe, param_grid, cv=3)
grid.fit(X_train_tf, y_train)

print(classification_report(grid.predict(X_test_tf), y_test))
pipe = make_pipeline(LogisticRegression(class_weight='balanced'))
param_grid = {'logisticregression__C': np.linspace(0.001, 1000, num=10)}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train_tf, y_train)

print(classification_report(grid.predict(X_test_tf), y_test))
pipe = make_pipeline(LinearSVC(loss='hinge', penalty='l2'))
param_grid = {'linearsvc__C': np.linspace(10, 1000, num=100)}

grid = GridSearchCV(pipe, param_grid, cv=3)
grid.fit(X_train_tf, y_train)

print(classification_report(grid.predict(X_test_tf), y_test))
pipe = make_pipeline(MultinomialNB())
param_grid = {'multinomialnb__alpha': np.linspace(0.0001, 1000, num=100)}

grid = GridSearchCV(pipe, param_grid, cv=3)
grid.fit(X_train_tf, y_train)

print(classification_report(grid.predict(X_test_tf), y_test))
import tensorflow as tf
import json
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, classification_report

vocab_size = 10000
embedding_dim = 16
max_length = 500
trunc_type='post'
oov_tok = "<OOV>"

X = df_total['content']
y = df_total['label']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#optional cleaning of data
# X_test = [' '.join(tokenize(x)) for x in X_test]
# X_train = [' '.join(tokenize(x)) for x in X_train]

X_test =  np.array(X_test)
X_train = np.array(X_train)
y_train = np.array(y_train)
y_test = np.array(y_test)
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(X_train)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(X_test)
testing_padded = pad_sequences(testing_sequences, maxlen=500)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
num_epochs = 10
model.fit(padded, y_train, epochs=num_epochs, validation_data=(testing_padded, y_test))
model.evaluate(testing_padded, y_test)
model.metrics_names 