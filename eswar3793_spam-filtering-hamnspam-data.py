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
import email

import email.policy

from bs4 import BeautifulSoup
ham_filenames = [name for name in sorted(os.listdir('../input/hamnspam/ham')) ]

spam_filenames = [name for name in sorted(os.listdir('../input/hamnspam/spam'))]
def load_email(is_spam, filename):

    directory = "../input/hamnspam/spam" if is_spam else "../input/hamnspam/ham"

    with open(os.path.join(directory, filename), "rb") as f:

        return email.parser.BytesParser(policy=email.policy.default).parse(f)
ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]

spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]
from collections import Counter



def get_email_structure(email):

    if isinstance(email, str):

        return email

    payload = email.get_payload()

    if isinstance(payload, list):

        return "multipart({})".format(", ".join([

            get_email_structure(sub_email)

            for sub_email in payload

        ]))

    else:

        return email.get_content_type()



def structures_counter(emails):

    structures = Counter()

    for email in emails:

        structure = get_email_structure(email)

        structures[structure] += 1

    return structures



ham_structure = structures_counter(ham_emails)

spam_structure = structures_counter(spam_emails)
def html_to_plain(email):

    try:

        soup = BeautifulSoup(email.get_content(), 'html.parser')

        return soup.text.replace('\n\n','')

    except:

        return "empty"
def email_to_plain(email):

    struct = get_email_structure(email)

    for part in email.walk():

        partContentType = part.get_content_type()

        if partContentType not in ['text/plain','text/html']:

            continue

        try:

            partContent = part.get_content()

        except: # in case of encoding issues

            partContent = str(part.get_payload())

        if partContentType == 'text/plain':

            return partContent

        else:

            return html_to_plain(part)
from sklearn.base import BaseEstimator, TransformerMixin

import nltk

# - Strip email headers

# - Convert to lowercase

# - Remove punctuation

# - Replace urls with "URL"

# - Replace numbers with "NUMBER"

# - Perform Stemming (trim word endings with library)

class EmailToWords(BaseEstimator, TransformerMixin):

    def __init__(self, stripHeaders=True, lowercaseConversion = True, punctuationRemoval = True, 

                 urlReplacement = True, numberReplacement = True, stemming = True):

        self.stripHeaders = stripHeaders

        self.lowercaseConversion = lowercaseConversion

        self.punctuationRemoval = punctuationRemoval

        self.urlReplacement = urlReplacement

        #self.url_extractor = urlextract.URLExtract()

        self.numberReplacement = numberReplacement

        self.stemming = stemming

        self.stemmer = nltk.PorterStemmer()

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        X_to_words = []

        for email in X:

            text = email_to_plain(email)

            if text is None:

                text = 'empty'

            if self.lowercaseConversion:

                text = text.lower()

                

            #if self.urlReplacement:

                #urls = self.url_extractor.find_urls(text)

                #for url in urls:

                #    text = text.replace(url, 'URL')   

                    

            if self.punctuationRemoval:

                text = text.replace('.','')

                text = text.replace(',','')

                text = text.replace('!','')

                text = text.replace('?','')

                

            word_counts = Counter(text.split())

            if self.stemming:

                stemmed_word_count = Counter()

                for word, count in word_counts.items():

                    stemmed_word = self.stemmer.stem(word)

                    stemmed_word_count[stemmed_word] += count

                word_counts = stemmed_word_count

            X_to_words.append(word_counts)

        return np.array(X_to_words)
from scipy.sparse import csr_matrix



class WordCountToVector(BaseEstimator, TransformerMixin):

    def __init__(self, vocabulary_size=1000):

        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):

        total_word_count = Counter()

        for word_count in X:

            for word, count in word_count.items():

                total_word_count[word] += min(count, 10)

        self.most_common = total_word_count.most_common()[:self.vocabulary_size]

        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(self.most_common)}

        return self

    def transform(self, X, y=None):

        rows = []

        cols = []

        data = []

        for row, word_count in enumerate(X):

            for word, count in word_count.items():

                rows.append(row)

                cols.append(self.vocabulary_.get(word, 0))

                data.append(count)

        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))
from sklearn.pipeline import Pipeline



email_pipeline = Pipeline([

    ("Email to Words", EmailToWords()),

    ("Wordcount to Vector", WordCountToVector()),

])
from sklearn.model_selection import cross_val_score, train_test_split



X = np.array(ham_emails + spam_emails)

y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))



X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_augmented_train = email_pipeline.fit_transform(X_train)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neighbors import NearestNeighbors

from sklearn import tree
lr=LogisticRegression()

nb=GaussianNB()

svm=SVC(kernel='rbf')

nn=NearestNeighbors(n_neighbors=2, algorithm='ball_tree')

dt=tree.DecisionTreeClassifier()
lr.fit(X_augmented_train,Y_train)

nb.fit(X_augmented_train.toarray(),Y_train)

svm.fit(X_augmented_train,Y_train)

nn.fit(X_augmented_train,Y_train)

dt.fit(X_augmented_train,Y_train)
X_augmented_test = email_pipeline.transform(X_test)
lr_predicted=lr.predict(X_augmented_test)

nb_predicted=nb.predict(X_augmented_test.toarray())

svm_predicted=svm.predict(X_augmented_test)

nn_predicted=nn.kneighbors(X_augmented_test)

dt_predicted=dt.predict(X_augmented_test)
lr_cls_report=classification_report(Y_test,lr_predicted)

svm_cls_report=classification_report(Y_test,svm_predicted)

nb_cls_report=classification_report(Y_test,nb_predicted)

dt_cls_report=classification_report(Y_test,dt_predicted)

print("Logistic regression : "+lr_cls_report)

print("SVM : "+svm_cls_report)

print("Naive Bayes : "+nb_cls_report)

print("Decision tree : "+dt_cls_report)