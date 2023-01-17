# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

train_df.head()
train_df.keyword.value_counts()
train_df.location.value_counts()
train_df.target.value_counts()
train_df['text'].head()[0]


sample_text = ['Hello world', '. ~ #### world', 'this is #sample text']

count_vectorizer = feature_extraction.text.CountVectorizer()

count_vectorizer.fit_transform(sample_text).todense()
count_vectorizer.vocabulary_
tfidvectorizer = feature_extraction.text.TfidfVectorizer()



tfidvectorizer.fit_transform(sample_text).todense()
tfidvectorizer.vocabulary_ # same as count_vectorizer.
hashvectorizer = feature_extraction.text.HashingVectorizer()



hashvectorizer.fit_transform(sample_text).todense()
hashvectorizer.get_stop_words
tfidf_vectorizer = feature_extraction.text.TfidfVectorizer()

count_vectorizer = feature_extraction.text.CountVectorizer()

hash_vectorizer = feature_extraction.text.HashingVectorizer()


def model_score(model, features, target):        

    return model_selection.cross_val_score(model, features, target, cv=5, scoring="f1")



def clean(data):    

    mean = round(data.mean(), 2)

    std = round(data.std(), 2)

    

    return f'mean: {mean} +/- {std}' 
tfidf_features = tfidf_vectorizer.fit_transform(train_df["text"]).todense()

count_features = count_vectorizer.fit_transform(train_df["text"]).todense()
hash_score = model_score(linear_model.RidgeClassifier(), tfidf_features, train_df["target"])

clean(hash_score)
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, Perceptron

from sklearn.tree import ExtraTreeClassifier

from sklearn.naive_bayes import GaussianNB
import re

def only_alpha(x):

    return re.sub(r'[\W_]', ' ', x).lower()



train_df['text_alpha_num'] = train_df['text'].apply(func = only_alpha)
clean(model_score(LogisticRegression(solver = 'lbfgs'),tfidf_vectorizer.fit_transform(train_df['text_alpha_num']).todense() , train_df["target"]))
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
tfidf_vectorizer = feature_extraction.text.TfidfVectorizer(stop_words = stop_words)

clean(model_score(LogisticRegression(solver = 'lbfgs'),tfidf_vectorizer.fit_transform(train_df['text_alpha_num']).todense() , train_df["target"]))
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()



def stemmer(x):

    return ' '.join([porter.stem(x) for x in x.split(' ')])



train_df['stemmed_words'] = train_df['text_alpha_num'].apply(func = stemmer)
train_df['stemmed_words'][0]
clean(model_score(LogisticRegression(solver = 'lbfgs'),tfidf_vectorizer.fit_transform(train_df['stemmed_words']).todense() , train_df["target"]))
stemmed = tfidf_vectorizer.fit_transform(train_df["stemmed_words"]).todense()

stemmed_data = pd.DataFrame(stemmed, columns=tfidf_vectorizer.get_feature_names())



alpha_num = tfidf_vectorizer.fit_transform(train_df["text_alpha_num"]).todense()

alpha_data = pd.DataFrame(alpha_num, columns=tfidf_vectorizer.get_feature_names())



data = pd.concat([stemmed_data,alpha_data], axis =1)



clean(model_score(LogisticRegression(solver = 'lbfgs'), data , train_df["target"]))
train_df['keyword'].value_counts()
train_df['keyword_test'] = train_df['keyword'].fillna('no keyword')
clean(model_score(LogisticRegression(solver = 'lbfgs'),tfidf_vectorizer.fit_transform(train_df['keyword_test']).todense() , train_df["target"]))
clean(model_score(GaussianNB(),tfidf_vectorizer.fit_transform(train_df['keyword_test']).todense() , train_df["target"]))
def get_hashtag(x):

    return ' '.join(list(re.findall(r"#(\w+)", x)))



train_df['tags'] = train_df['text'].apply(func = get_hashtag)
train_df['tags'].value_counts()
train_df.head()


def insert_tags(x):

    if x == '':

        return 'no tags'

    else:

        return x



train_df['tags'] = train_df['tags'].apply(func = insert_tags)
train_df['tags'].value_counts()
clean(model_score(GaussianNB(),tfidf_vectorizer.fit_transform(train_df['tags']).todense() , train_df["target"]))
train_df['location'].fillna('no location').value_counts()
train_df['location'] = train_df['location'].fillna('no location')
clean(model_score(GaussianNB(),tfidf_vectorizer.fit_transform(train_df['location']).todense() , train_df["target"]))
stemmed = tfidf_vectorizer.fit_transform(train_df["stemmed_words"]).todense()

stemmed_data = pd.DataFrame(stemmed, columns=tfidf_vectorizer.get_feature_names())



alpha_num = tfidf_vectorizer.fit_transform(train_df["text_alpha_num"]).todense()

alpha_data = pd.DataFrame(alpha_num, columns=tfidf_vectorizer.get_feature_names())



data = pd.concat([stemmed_data,alpha_data], axis =1)



clean(model_score(LogisticRegression(solver = 'lbfgs'), data , train_df["target"]))
alpha_num = tfidf_vectorizer.fit_transform(train_df["text_alpha_num"]).todense()

alpha_data = pd.DataFrame(alpha_num, columns=tfidf_vectorizer.get_feature_names())



tags_tfidf = tfidf_vectorizer.fit_transform(train_df["tags"]).todense()

tags_data = pd.DataFrame(tags_tfidf, columns=tfidf_vectorizer.get_feature_names())



keys_tfidf = tfidf_vectorizer.fit_transform(train_df["keyword_test"]).todense()

keys_data = pd.DataFrame(keys_tfidf, columns=tfidf_vectorizer.get_feature_names())





location_tfidf = tfidf_vectorizer.fit_transform(train_df["location"]).todense()

location_data = pd.DataFrame(location_tfidf, columns=tfidf_vectorizer.get_feature_names())



data = pd.concat([alpha_data,tags_data,keys_data,location_data], axis =1)
clean(model_score(GaussianNB(),data , train_df["target"]))
train_df['tags'][0] + ' ' + train_df['location'][0]
data['combined'] = train_df['tags'] + ' ' + train_df['location'] + ' ' + train_df["keyword_test"] + ' ' + train_df["text_alpha_num"]
clean(model_score(GaussianNB(), tfidf_vectorizer.fit_transform(data['combined']).todense() , train_df["target"]))
tfidf_vectorizer = feature_extraction.text.TfidfVectorizer()

alpha_num = tfidf_vectorizer.fit_transform(train_df["text_alpha_num"]).todense()

alpha_data = pd.DataFrame(alpha_num, columns=tfidf_vectorizer.get_feature_names())



clf = LogisticRegression(n_jobs = -1, random_state = 1337)

clf.fit(alpha_data , train_df["target"])


test_df['text_alpha_num'] = test_df['text'].apply(func = only_alpha)



alpha_num_test = tfidf_vectorizer.transform(test_df["text_alpha_num"]).todense()

alpha_data_test = pd.DataFrame(alpha_num_test, columns=tfidf_vectorizer.get_feature_names())
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")



sample_submission["target"] = clf.predict(alpha_data_test)



sample_submission.head()



sample_submission.to_csv('submission.csv', index=False)




