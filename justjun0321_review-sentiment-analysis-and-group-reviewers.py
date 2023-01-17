import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up matplotlib style 
plt.style.use('ggplot')

# Libraries for wordcloud making and image importing
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image

# And libraries for data transformation
import datetime
from string import punctuation
data = pd.read_csv('../input/amazon-alexa-reviews/amazon_alexa.tsv', delimiter='\t')
data.head()
for col in data.columns:
    print('The distribution of column: ',col)
    print(data[col].value_counts()[:10])
    print('--------------------------------')
sns.countplot(y=data.rating,orient='h',order=[5,4,3,2,1])
data['length'] = data['verified_reviews'].apply(lambda x: len(x))
data[np.isin(data['rating'].tolist(),[3,4])]['length'].hist(bins=25)
plt.title('Length distribution of Rating 3 or 4')
data[np.isin(data['rating'].tolist(),[1,2,5])]['length'].hist(bins=25)
plt.title('Length distribution of Rating 3 or 4')
!pip install textatistic
from textatistic import Textatistic
def Textatistic_edited(x):
    try:
        return Textatistic(x).scores['flesch_score']
    except ZeroDivisionError:
        return 0
data['readability'] = data['verified_reviews'].apply(lambda x: Textatistic_edited(x))
data[np.isin(data['rating'].tolist(),[3,4])]['readability'].hist(bins=25)
plt.title('Readability distribution of Rating 3 or 4')
data[np.isin(data['rating'].tolist(),[1,2,5])]['readability'].hist(bins=25)
plt.title('Readability distribution of Rating 3 or 4')
plt.figure(figsize=(16, 4))

plt.subplot(1,2,1)
data[data['rating']==3]['readability'].hist(bins=25)
plt.title('Readability distribution of Rating 3')

plt.subplot(1,2,2)
data[data['rating']==4]['readability'].hist(bins=25)
plt.title('Readability distribution of Rating 3')
from scipy.cluster.vq import kmeans, vq
from numpy import random

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(data['verified_reviews'])
stop_words_0 = set(stopwords.words('english')) 
stop_words = ['and', 'in', 'of', 'or', 'with','to','on','a','love','use','alexa','music']

def remove_noise(text):
    tokens = word_tokenize(text)
    clean_tokens = []
    lemmatizer=WordNetLemmatizer()
    for token in tokens:
        token = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+', '', token)
        token = lemmatizer.lemmatize(token.lower())
        if len(token) > 1 and token not in stop_words_0 and token not in stop_words:
            clean_tokens.append(token)
            
    return clean_tokens
tfidf_vectorizer = TfidfVectorizer(max_features=100,tokenizer=remove_noise)

# Use the .fit_transform() method on the list plots
tfidf_matrix = tfidf_vectorizer.fit_transform(data['verified_reviews'])
random.seed = 123
distortions = []
num_clusters = range(2, 25)

# Create a list of distortions from the kmeans function
for i in num_clusters:
    cluster_centers, distortion = kmeans(tfidf_matrix.todense(),i)
    distortions.append(distortion)

# Create a data frame with two lists - num_clusters, distortions
elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

# Creat a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data = elbow_plot)
plt.xticks(num_clusters)
plt.title('Clusters and Distortions')
plt.show()

cluster_centers, distortion = kmeans(tfidf_matrix.todense(),6)

# Generate terms from the tfidf_vectorizer object
terms = tfidf_vectorizer.get_feature_names()

for i in range(6):
    # Sort the terms and print top 10 terms
    center_terms = dict(zip(terms, list(cluster_centers[i])))
    sorted_terms = sorted(center_terms, key=center_terms.get, reverse=True)
    print(sorted_terms[:5])
