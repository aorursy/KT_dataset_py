# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import train_test_split
df = pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
df["sentiment"] = df["sentiment"].map(lambda x: 1 if x == "positive" else 0)
df.head()
df.sentiment.value_counts()
X_train, X_test, y_train, y_test = train_test_split(df["review"], 
                                                    df["sentiment"],
                                                    test_size=0.20, 
                                                    random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
X_train[13653]
y_train[13653]
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
def get_sentiment(sentence):
    scores = analyser.polarity_scores(sentence)
    if scores["compound"] > 0:
        return 1
    else:
        return 0
y_preds = X_test.map(get_sentiment)
y_preds
(y_preds == y_test).mean()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes
count_vec = CountVectorizer()
count_vec.fit(X_train)
X_train_tfm = count_vec.transform(X_train)
X_test_tfm = count_vec.transform(X_test)
model = naive_bayes.MultinomialNB()
model.fit(X_train_tfm, y_train)
y_preds = model.predict(X_test_tfm)
(y_preds == y_test).mean()
count_vec = CountVectorizer(analyzer="word", stop_words="english", ngram_range=(1, 3), max_df=.2, min_df=2)
count_vec.fit(X_train)

X_train_tfm = count_vec.transform(X_train)
X_test_tfm = count_vec.transform(X_test)

model = naive_bayes.MultinomialNB()
model.fit(X_train_tfm, y_train)

y_preds = model.predict(X_test_tfm)

(y_preds == y_test).mean()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer(analyzer="word", stop_words="english", ngram_range=(1, 2))
tfidf_vec.fit(X_train)
X_train_tfidf_tfm = tfidf_vec.transform(X_train)
X_test_tfidf_tfm = tfidf_vec.transform(X_test)

model = naive_bayes.MultinomialNB()
model.fit(X_train_tfidf_tfm, y_train)

y_preds = model.predict(X_test_tfidf_tfm)

(y_preds == y_test).mean()
import gensim
from nltk.tokenize import word_tokenize
from tqdm import tqdm
def sentence_to_vec(s, embedding_dict, stop_words, tokenizer):
    words = str(s).lower()
    words = tokenizer(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    
    M = []
    for w in words:
        if w in embedding_dict:
            M.append(embedding_dict[w])
    
    if len(M) == 0:
        return np.zeros(300)
    
    M = np.array(M)
    v = M.sum(axis=0)
    v / np.sqrt((v ** 2).sum())
    return v
embeddings = gensim.models.KeyedVectors.load_word2vec_format(
    "../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin", 
    binary=True
)
X_train_ft = []
for review in tqdm(X_train.values):
    X_train_ft.append(
        sentence_to_vec(
            s = review,
            embedding_dict = embeddings,
            stop_words = [],
            tokenizer = word_tokenize
        )
    )

X_test_ft = []
for review in tqdm(X_test.values):
    X_test_ft.append(
        sentence_to_vec(
            s = review,
            embedding_dict = embeddings,
            stop_words = [],
            tokenizer = word_tokenize
        )
    )

X_train_ft = np.array(X_train_ft)
X_test_ft = np.array(X_test_ft)
X_train_ft.shape, X_test_ft.shape
X_train[1]
X_train_ft[1]
from sklearn import linear_model 
model = linear_model.LogisticRegression(max_iter=1000)
model.fit(X_train_ft, y_train)

y_preds = model.predict(X_test_ft)

(y_preds == y_test).mean()