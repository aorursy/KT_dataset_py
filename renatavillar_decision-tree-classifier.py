import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from sklearn.metrics import confusion_matrix,accuracy_score
data = pd.read_csv('../input/mpst-movie-plot-synopses-with-tags/mpst_full_data.csv', sep=',')
data.head()
imdb_id = data.imdb_id
plot = data.plot_synopsis

movies = pd.DataFrame({'imdb_id': imdb_id, 'plot': plot})
movies.head()
data['imdb_id'] = data['imdb_id'].astype(str)
movies = pd.merge(movies, data[['imdb_id', 'title', 'tags', 'split']], on='imdb_id')
movies.head()
genres_listas = []

for tag in movies.tags:
    genres_listas.append(re.split(',', tag))

movies.tags = genres_listas
all_genres = sum(genres_listas, [])
for i in range(0, len(all_genres)):
    all_genres[i] = all_genres[i].strip()
print(len(set(all_genres)))
all_genres
all_genres = nltk.FreqDist(all_genres)
all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()), 'Count': list(all_genres.values())})
plt.style.use('ggplot')
g = all_genres_df.nlargest(columns='Count', n=50)
plt.figure(figsize=(12,15))
ax = sns.barplot(data=g, x='Count', y='Genre')
ax.set(ylabel='Count')
plt.show()
# function for text cleaning 
def clean_text(text):
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower() 
    return text
import copy
movies_new = copy.deepcopy(movies)
movies_new['clean_plot'] = movies['plot'].apply(lambda x: clean_text(x))
movies_new.head()
def freq_words(x, terms = 30):
    all_words = ' '.join([text for text in x]) 
    all_words = all_words.split() 
    fdist = nltk.FreqDist(all_words) 
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())}) 
  
    # selecting top 20 most frequent words 
    d = words_df.nlargest(columns="count", n = terms) 
  
    # visualize words and frequencies
    plt.figure(figsize=(12,15)) 
    ax = sns.barplot(data=d, x= "count", y = "word") 
    ax.set(ylabel = 'Word') 
    plt.show()
#print 100 most frequent words 
freq_words(movies_new['clean_plot'], 100)
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

movies_new['clean_plot'] = movies_new['clean_plot'].apply(lambda x: remove_stopwords(x))
freq_words(movies_new['clean_plot'], 100)
mlb = MultiLabelBinarizer()
mlb.fit(movies_new[movies_new['split'] == 'train']['tags'])
y = mlb.transform(movies_new[movies_new['split'] == 'train']['tags'])
y.shape
movies_new = pd.concat([movies_new, pd.DataFrame(y, columns=mlb.classes_)], axis=1)
print(len(movies_new.columns))
movies_new.columns
counts = []
genres = mlb.classes_
split = []
for genre in genres:
    counts.append((genre, movies_new[genre].sum()))
movies_new_stats = pd.DataFrame(counts, columns=['genres', 'plots'])
print(len(movies_new.columns))
genres = movies_new_stats.loc[movies_new_stats['plots'] > 100, 'genres'].tolist()
movies_new.drop(movies_new_stats.loc[movies_new_stats['plots'] < 100, 'genres'].tolist(), axis=1, inplace=True)
len(movies_new.columns)
mlb = MultiLabelBinarizer()
mlb.fit(movies_new[(movies_new['split'] == 'train') | (movies_new['split'] == 'val')]['tags'])
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
y = movies_new[movies_new['split'] == 'train'].iloc[:, 6:len(movies_new.columns)+1]
y.fillna(1, inplace=True)
y = np.array(y)
xtrain_tfidf = tfidf_vectorizer.fit_transform(movies_new[movies_new['split'] == 'train']['clean_plot'])
xtrain_tfidf
xval_tfidf = tfidf_vectorizer.fit_transform(movies_new[movies_new['split'] == 'val']['clean_plot'])
xval_tfidf
xtest_tfidf = tfidf_vectorizer.fit_transform(movies_new[movies_new['split'] == 'test']['clean_plot'])
xtest_tfidf
clf = DecisionTreeClassifier(criterion='gini', max_depth = 39, random_state=1)
i = 0
t = []

while i <= y.shape[1]-1:
    s = []
    clf.fit(xtrain_tfidf, y[:,i])
    s.append(clf.predict(xtest_tfidf))
    i += 1
    t.append(s)
y_pred = np.array(t).reshape(np.array(t).shape[2], np.array(t).shape[0])
y_pred.shape
y_true = movies_new[movies_new['split'] == 'test'].iloc[:, 6:len(movies_new.columns)+1]
y_true.fillna(1, inplace=True)
y_true = np.array(y_true)
print(y_pred.shape)
print(y_true.shape)
sklearn.metrics.f1_score(y_true, y_pred, average='micro')
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
sklearn.metrics.multilabel_confusion_matrix(y_true, y_pred)