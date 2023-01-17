import nltk

import pandas as pd

import re

import string

import seaborn as sns

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV



from matplotlib import pyplot as plt
from sklearn import set_config

set_config(print_changed_only = False)



%matplotlib inline
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

train_df.head()
test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test_df.head()
train_df.shape
train_df.info()
train_df.isnull().sum()
hashtag = r"#[a-zA-Z]\w+"

train_df['hashtag_count'] = train_df['text'].apply(lambda txt: len(re.findall(hashtag, txt)))

train_df.head()
test_df['hashtag_count'] = test_df['text'].apply(lambda txt: len(re.findall(hashtag, txt)))

test_df.head()
train_df['hashtag_count'].describe()
plt.figure(figsize=(12,6))

sns.countplot(x= 'hashtag_count', hue='target', data = train_df)
train_df['keyword'].fillna(value = 'NaN', inplace = True)

test_df['keyword'].fillna(value = 'NaN', inplace = True)



print('Null value count for keywords in training dataset: ', train_df['keyword'].isnull().sum())

print('Null value count for keywords in testing dataset: ', test_df['keyword'].isnull().sum())
kw_le = LabelEncoder()

kw_le.fit(train_df['keyword'])
train_df['keyword'] = kw_le.transform(train_df['keyword'])

train_df.head()
test_df['keyword'] = kw_le.transform(test_df['keyword'])

test_df.head()
train_df['text_len'] = train_df['text'].apply(lambda x: len(x) - x.count(" "))

train_df.head()
test_df['text_len'] = test_df['text'].apply(lambda x: len(x) - x.count(" "))

test_df.head()
train_df['text_len'].describe()
bins = np.linspace(0, 150, 16)



plt.figure(figsize = (12, 6))

plt.hist(train_df[train_df['target'] == 1]['text_len'], bins, alpha = 0.5, density = True, label = 'Real')

plt.hist(train_df[train_df['target'] == 0]['text_len'], bins, alpha = 0.5, density = True, label = 'Not Real')

plt.legend()

plt.show()
wn = nltk.WordNetLemmatizer()

stopwords = nltk.corpus.stopwords.words('english')
def tokenize_text(text):

    text = text.lower()

    text = "".join([word for word in text if word not in string.punctuation])

    tokens = re.split('[^a-z]+', text)

    return tokens



def clean_text(tokens):

    text = [wn.lemmatize(word) for word in tokens if not (word in stopwords or str.isspace(word) or len(word)==0)]

    return text 
train_df['text'] = train_df['text'].apply(lambda txt: tokenize_text(txt))

train_df['word_count'] = train_df['text'].apply(lambda tokens: len(tokens))

train_df.head()
test_df['text'] = test_df['text'].apply(lambda txt: tokenize_text(txt))

test_df['word_count'] = test_df['text'].apply(lambda tokens: len(tokens))

test_df.head()
train_df['word_count'].describe()
bins = np.linspace(0, 32, 17)



plt.figure(figsize = (12, 6))

plt.hist(train_df[train_df['target'] == 1]['word_count'], bins, alpha = 0.5, density = True, label = 'Real')

plt.hist(train_df[train_df['target'] == 0]['word_count'], bins, alpha = 0.5, density = True, label = 'Not Real')

plt.legend()

plt.show()
tfidf_vect = TfidfVectorizer(analyzer= clean_text)

tfidf_vect_fit = tfidf_vect.fit(train_df['text'])
tfidf_vect_columns = ['tf_' + colname for colname in tfidf_vect.get_feature_names()]

tfidf_vect_columns[::1000]
train_tf_df = pd.DataFrame(tfidf_vect_fit.transform(train_df['text']).toarray(), columns = tfidf_vect_columns)

train_tf_df.head()
test_tf_df = pd.DataFrame(tfidf_vect_fit.transform(test_df['text']).toarray(), columns = tfidf_vect_columns)

test_tf_df.head()
X_features = pd.concat([train_df[['keyword', 'hashtag_count', 'text_len', 'word_count']], train_tf_df], axis = 1)

X_features.shape
Y_label = train_df['target']

Y_label.shape
rf = RandomForestClassifier()

rf
param = {'n_estimators': [50, 100, 150],

        'max_depth': [30, 60, 90, None]}



k_fold = KFold(n_splits = 5)



gs = GridSearchCV(rf, param, cv = k_fold)

gs_fit = gs.fit(X_features, Y_label)

pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending = False).head()
gs_fit.best_params_
X_test_features = pd.concat([test_df[['keyword', 'hashtag_count', 'text_len', 'word_count']], test_tf_df], axis = 1)

X_test_features.shape
y_hat = gs_fit.predict(X_test_features)

y_hat.shape
test_df = pd.concat([test_df, pd.DataFrame(y_hat, columns=['target'])], axis = 1)

test_df.head()
submission_df = test_df[['id', 'target']]

submission_df.head()
submission_df.to_csv('disaster_tweet_nlp_mayur.csv', index=False)