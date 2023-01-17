import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import string
import re
language = 'english'
sentence_tokenizer = nltk.data.load('tokenizers/punkt/%s.pickle' % language)
stop_words=set(nltk.corpus.stopwords.words(language))
def clean_tokenize(text):
    text = text.lower().replace('b&b', 'bb')
    tmp_tokens = nltk.tokenize.word_tokenize(text)
    no_punctuation = []
    x=re.compile('^[%s]+$' % re.escape(string.punctuation))
    
    for tk in tmp_tokens:
        if not x.match(tk):
            no_punctuation.append(tk)

    return [token for token in no_punctuation if token not in stop_words]

def feature_extractor(text):
    # we don't consider any stop_words
    capital_lett_cnt = len(re.findall(r'[A-Z]', text))
    meaningful_words = clean_tokenize(text)
    vocabulary       = set(meaningful_words)
    sentences        = sentence_tokenizer.tokenize(text)
    sentences_number = float(len(sentences))
    # wps stands for word per sentence
    meaningful_wps   = np.array([len(clean_tokenize(s)) for s in sentences])
    
    # we return a list of features ordered as:
    # 1: mean meaningful_wps
    # 2: std_dev meaningful_wps
    # 3: lexical diversity index (:= len(vocabulary) / len(words)), which
    #    accounts the lexical richness of the text
    # 4: commas per sentence
    # 5: semicolons per sentence
    # 6: colons per sentence 
    # 7: exclamations per sentence (it should be <= 1)
    # 8: capital letters per sentence (it should be >= 1)
    return meaningful_wps.mean(),\
        meaningful_wps.std(),\
        len(vocabulary) / float(len(meaningful_words)),\
        text.count(',') / sentences_number, \
        text.count(';') / sentences_number, \
        text.count(':') / sentences_number, \
        text.count('!') / sentences_number, \
        capital_lett_cnt
reviews = pd.read_csv('../input/reviews.csv', encoding='utf-8')
reviews['review_text'] = reviews['review_text'].str.replace('...More$', '', case=False)
pattern = '^[A-Z][a-z]+ [A-Z]$'
suspect_en = reviews[(reviews['review_user'].str.match(pattern)) & (reviews['review_language'] == 'en')].review_text
len(suspect_en)
scaler = StandardScaler()
features_vector = np.ndarray((len(suspect_en), 8))
for i, text in enumerate(suspect_en):
    features_vector[i] = feature_extractor(text)
features_vector = scaler.fit_transform(features_vector)
db = DBSCAN(min_samples=9)
y_pred = db.fit_predict(features_vector)

# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

print('Estimated number of clusters: %d (noise: %d)' % (n_clusters, np.count_nonzero(db.labels_==-1)))
plt.subplots(figsize=(13,5))
sns.countplot(x=db.labels_[2:])
plt.xticks(rotation=90)
plt.show()
np.count_nonzero(db.labels_==0)
plt.subplots(figsize=(13,5))
sns.countplot(x=db.labels_[db.labels_>0])
plt.xticks(rotation=90)
plt.show()
