import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
import eli5

import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')
def generate_features(df):
    df["word_count"] = df["question_text"].apply(lambda x: len(str(x).split()))
    df["unique_word_count"] = df["question_text"].apply(lambda x: len(set(str(x).split())))
    df["char_length"] = df["question_text"].apply(lambda x: len(str(x)))
    df["stop_words_count"] = df["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
    df["punc_count"] = df["question_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    df["upper_words"] = df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    df["title_words"] = df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    df["word_length"] = df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    return df

train = generate_features(train)
test = generate_features(test)
# Get the tfidf vectors
tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
tfidf_vec.fit_transform(train['question_text'].values.tolist() + test['question_text'].values.tolist())
train_tfidf = tfidf_vec.transform(train['question_text'].values.tolist())
test_tfidf = tfidf_vec.transform(test['question_text'].values.tolist())
y_train = train["target"].values

x_train = train_tfidf
x_test = test_tfidf

model = linear_model.LogisticRegression(C=5., solver='sag')
model.fit(x_train, y_train)
y_test = model.predict_proba(x_test)[:,1]
eli5.show_weights(model, vec=tfidf_vec, top=100, feature_filter=lambda x: x != '<BIAS>')
sub['prediction'] = y_test
sub.to_csv('submission.csv',index=False)