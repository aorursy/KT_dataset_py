import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import LinearSVR

import string

from joblib import Parallel, delayed

from tqdm import tqdm_notebook as tqdm

%matplotlib inline
import nltk

# nltk.download('stopwords')

# nltk.download('punkt')

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize

from nltk.stem import SnowballStemmer

stop_words = set(stopwords.words('english'))

stem = SnowballStemmer('english')
train_df = pd.read_csv("../input/train.csv")

train_df = train_df[['id','comment_text', 'target']]

test_df = pd.read_csv("../input/test.csv")
train_df.head()
train_df.target.hist()
train_df.shape
test_df.head()
test_df.shape
# train_df = train_df.sample(100000, random_state=42)
train_df.shape
def tokenize(text):

    

    tokens = []

    for token in word_tokenize(text):

        if token in string.punctuation: continue

        if token in stop_words: continue

        tokens.append(stem.stem(token))

    

    return " ".join(tokens)
train_tokens = Parallel(n_jobs=-1, verbose=1)(delayed(tokenize)(text) for text in train_df['comment_text'].tolist())
train_tokens[0]
test_tokens = Parallel(n_jobs=-1, verbose=1)(delayed(tokenize)(text) for text in test_df['comment_text'].tolist())
len(train_tokens + test_tokens)
vect = TfidfVectorizer()

vect.fit(train_tokens + test_tokens)
X = vect.transform(train_tokens)

y = train_df['target']
svr = LinearSVR(random_state=71, tol=1e-3, C=1.2)

svr.fit(X, y)
test_X =  vect.transform(test_tokens)

test_y = svr.predict(test_X)
submisson_df = pd.read_csv("../input/sample_submission.csv")

submisson_df['prediction'] = test_y

submisson_df['prediction'] = submisson_df['prediction'].apply(lambda x: "%.5f" % x if x > 0 else 0.0)
submisson_df.to_csv("submission.csv", index=False)