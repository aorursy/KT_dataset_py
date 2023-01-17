import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
data = pd.read_csv('../input/tweets.csv')
data.info()
data.head()
regex_str = [
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
 
def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens
vect1 = CountVectorizer(analyzer="word", stop_words="english", min_df=200, decode_error="ignore", ngram_range=(1, 1), dtype=np.int32)

# applying Vectorizer to preprocessed tweets
sub11 = vect1.fit_transform(data["tweets"].map(lambda x: " ".join(preprocess(x, lowercase = True))).tolist())

# creating (word, count) list
sub12 = zip(vect1.get_feature_names(), np.asarray(sub11.sum(axis = 0)).ravel())

# getting Top20 words
sorted(sub12, key = lambda x: x[1], reverse = True)[0:20]
vect2 = CountVectorizer(analyzer="word", min_df=2, decode_error="ignore", ngram_range=(2, 2), dtype=np.int32)

# applying Vectorizer to preprocessed tweets
sub21 = vect2.fit_transform(data["tweets"].map(lambda x: " ".join(preprocess(x, lowercase = True))).tolist())

# creating (word, count) list
sub22 = zip(vect2.get_feature_names(), np.asarray(sub21.sum(axis = 0)).ravel())

# getting Top20 words
sorted(sub22, key = lambda x: x[1], reverse = True)[0:20]
vect3 = CountVectorizer(analyzer="word", min_df=2, decode_error="ignore", ngram_range=(3, 3), dtype=np.int32)

# applying Vectorizer to preprocessed tweets
sub31 = vect3.fit_transform(data["tweets"].map(lambda x: " ".join(preprocess(x, lowercase = True))).tolist())

# creating (word, count) list
sub32 = zip(vect3.get_feature_names(), np.asarray(sub31.sum(axis = 0)).ravel())

# getting Top20 words
sorted(sub32, key = lambda x: x[1], reverse = True)[0:20]
vect4 = CountVectorizer(analyzer="word", min_df=2, decode_error="ignore", ngram_range=(4, 4), dtype=np.int32)

# applying Vectorizer to preprocessed tweets
sub41 = vect4.fit_transform(data["tweets"].map(lambda x: " ".join(preprocess(x, lowercase = True))).tolist())

# creating (word, count) list
sub42 = zip(vect4.get_feature_names(), np.asarray(sub41.sum(axis = 0)).ravel())

# getting Top20 words
sorted(sub42, key = lambda x: x[1], reverse = True)[0:20]
vect5 = CountVectorizer(analyzer="word", min_df=2, decode_error="ignore", ngram_range=(5, 5), dtype=np.int32)

# applying Vectorizer to preprocessed tweets
sub51 = vect5.fit_transform(data["tweets"].map(lambda x: " ".join(preprocess(x, lowercase = True))).tolist())

# creating (word, count) list
sub52 = zip(vect5.get_feature_names(), np.asarray(sub51.sum(axis = 0)).ravel())

# getting Top20 words
sorted(sub52, key = lambda x: x[1], reverse = True)[0:20]
tags = data["tweets"].map(lambda x: [tag for tag in preprocess(x, lowercase=True) if tag.startswith('@')])
tags = sum(tags, [])
tags[0:5]
# Top20
Counter(tags).most_common(20)
hashs = data["tweets"].map(lambda x: [hashs for hashs in preprocess(x, lowercase=True) if hashs.startswith('#')])
hashs = sum(hashs, [])
hashs[0:5]
# Top20
Counter(hashs).most_common(20)
urls = data["tweets"].map(lambda x: [url for url in preprocess(x, lowercase=True) if url.startswith('http:') or url.startswith('https:')])
urls = sum(urls, [])
urls[0:5]
# Top20
Counter(urls).most_common(20)