



import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from sklearn import metrics

from nltk.tokenize import word_tokenize, TreebankWordTokenizer

from nltk import SnowballStemmer, PorterStemmer, LancasterStemmer

import nltk

from nltk.stem.wordnet import WordNetLemmatizer





from sklearn.feature_extraction import text



import re

import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download('wordnet')





train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
test_id = test_df['id']

test_id
groups = train_df.groupby("type").count()

groups.sort_values("posts", ascending=False, inplace=True)

print ("Personality types", groups.index.values)
groups["posts"].plot(kind="bar", title="Number of Users per Personality type")
def text_separator(df):

    if 'id' not in df.columns:

        df['id'] = df.index

    df["seperate_posts"] = df["posts"].apply(lambda x: x.strip().split("|||"))

    df_temp = pd.DataFrame(df['seperate_posts'].tolist(), index=df['id']).stack().reset_index(level=1, drop=True).reset_index(name='unique_posts')

    df = df_temp.join(df.set_index('id'), on='id', how = 'left')

    df = df.drop(['posts', 'seperate_posts'], axis = 1)

    return df
train_df1 = text_separator(train_df)

test_df1 = text_separator(test_df)
train_df1.head()
def text_cleaner(text):

    result = re.sub(r'http[^\s]*', 'urlweb',text)

    result = re.sub('[0-9]+','', result).lower()

    result = re.sub('@[a-z0-9]+', 'user', result)

    result = ''.join([l for l in result if l not in string.punctuation])

    return result
train_df1['cleaned_post'] = train_df1['unique_posts'].apply(text_cleaner)

test_df1['cleaned_post'] = test_df1['unique_posts'].apply(text_cleaner)
train_df1.head()
train_df1['type'].value_counts().plot(kind = 'bar')

plt.show()
def token_maker(df):

    tokeniser = TreebankWordTokenizer()

    df['tokens'] = df['cleaned_post'].apply(tokeniser.tokenize)

    return df
train_df1 = token_maker(train_df1)

test_df1 = token_maker(test_df1)
train_df1.head()
# find the stem of each word in words

def stemm_maker(words):

    stemm = SnowballStemmer('english')

    return [stemm.stem(word) for word in words]  



train_df1['stem'] = train_df1['tokens'].apply(stemm_maker)

test_df1['stem'] = test_df1['tokens'].apply(stemm_maker)
def lemma_maker(words):

    lemmatizer = WordNetLemmatizer()

    return [lemmatizer.lemmatize(word) for word in words]
train_df1['lemma'] = train_df1['tokens'].apply(lemma_maker)

test_df1['lemma'] = test_df1['tokens'].apply(lemma_maker)
train_df1.head()
train_df1['cleaned_lemma'] = train_df1['lemma'].apply(lambda x: ' '.join(x))

test_df1['cleaned_lemma'] = test_df1['lemma'].apply(lambda x: ' '.join(x))
X_train = train_df1.groupby('id')['cleaned_lemma'].apply(list).reset_index()

X_test = test_df1.groupby('id')['cleaned_lemma'].apply(list).reset_index()



train_df['clean_post'] = X_train['cleaned_lemma'].apply(lambda x: ' '.join(x))

test_df['clean_post'] = X_test['cleaned_lemma'].apply(lambda x: ' '.join(x))

def mbti_classes(df):

    mind = {"I": 0, "E": 1}

    energy = {"S": 0, "N": 1}

    nature = {"F": 0, "T": 1}

    tactics = {"P": 0, "J": 1}

    mbti = [mind, energy, nature, tactics]

    mbti_list = ['mind', 'energy', 'nature', 'tactics']

    for i in range(len(mbti)):

        df[str(mbti_list[i])] = df['type'].astype(str).str[i].map(mbti[i])

    return df
train_df = mbti_classes(train_df)
words2remove = ['infj', 'entp', 'intp', 'intj', 'entj', 'enfj', 'infp', 'enfp',

       'isfp', 'istp', 'isfj', 'istj', 'estp', 'esfp', 'estj', 'esfj', 'infjs', 'entps', 'intps', 'intjs', 'entjs', 'enfjs', 'infps', 'enfps',

       'isfps', 'istps', 'isfjs', 'istjs', 'estps', 'esfps', 'estjs', 'esfjs', 'mbti']
from sklearn.feature_extraction.text import CountVectorizer



vect = CountVectorizer(lowercase=False, stop_words = words2remove, max_features=200, ngram_range= (3,3))

train_vector = vect.fit_transform(train_df['clean_post'])

test_vector = vect.fit_transform(test_df['clean_post'])

tfizer = TfidfTransformer()

tfizer.fit(train_vector)

train_vector = tfizer.fit_transform(train_vector)
tfizer = TfidfTransformer()

tfizer.fit(test_vector)

test_vector = tfizer.fit_transform(test_vector)
vect.get_feature_names()
X_train = train_vector

X_test = test_vector



from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()



y_train1 = train_df['mind']



logreg.fit(X_train, y_train1)

y_pred_mind = logreg.predict(X_test)
y_train2 = train_df['energy']



logreg.fit(X_train, y_train2)

y_pred_energy = logreg.predict(X_test)
y_train3 = train_df['nature']



logreg.fit(X_train, y_train3)

y_pred_nature = logreg.predict(X_test)
y_train4 = train_df['tactics']



logreg.fit(X_train, y_train4)

y_pred_tactics = logreg.predict(X_test)
LogisticRegressor =pd.DataFrame({'id': test_id, 'mind': y_pred_mind, 'energy': y_pred_energy, 'nature':y_pred_nature, 'tactics': y_pred_tactics})

LogisticRegressor.to_csv('LogisticRegressor.csv', index=False)