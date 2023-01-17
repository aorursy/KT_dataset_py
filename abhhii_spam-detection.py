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


import os

import tarfile

from six.moves import urllib

SPAM_PATH = "/kaggle/input/"

def fetch_spam_data(spam_url="/kaggle/input/spam_dataset/", spam_path="/kaggle/input/spam_dataset/"):

    if not os.path.isdir(spam_path):

        os.makedirs(spam_path)

    for filename, url in (("20030228_easy_ham.tar", spam_url), ("20030228_spam.tar", spam_url)):

        path = os.path.join(spam_path, filename)

        print(path)

        if not os.path.isfile(path):

            urllib.request.urlretrieve(url, path)

        tar_bz2_file = tarfile.open(path)

        tar_bz2_file.extractall(path=SPAM_PATH)

        tar_bz2_file.close()
# fetch_spam_data()

HAM_DIR = os.path.join(SPAM_PATH, "20030228_easy_ham/easy_ham")

SPAM_DIR = os.path.join(SPAM_PATH, "20030228_spam/spam")

ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]

spam_filenames = [name for  name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]
len(ham_filenames)
len(spam_filenames)
import email

import email.policy



def load_email(is_spam, filename, spam_path = SPAM_PATH):

    directory = "20030228_spam/spam" if is_spam else "20030228_easy_ham/easy_ham"

    with open(os.path.join(spam_path, directory, filename), "rb") as f:

        return email.parser.BytesParser(policy=email.policy.default).parse(f)
ham_emails = [load_email(is_spam= False, filename=name) for name in ham_filenames]

spam_emails= [load_email(is_spam = True, filename=name) for name in spam_filenames]
print(ham_emails[1].get_content().strip())
# print(spam_emails[4].get_content().strip())

print(spam_emails[4])
def get_email_structure(email):

    if isinstance(email, str):

        return email

    payload = email.get_payload()

    if isinstance(payload, list):

        return "multipart({})".format(", ".join([

            get_email_structure(sub_email) for sub_email in payload

        ]))

    else:

        return email.get_content_type()
from collections import Counter



def structures_counter(emails):

    structures = Counter()

    for email in emails:

        structure = get_email_structure(email)

        structures[structure]+=1

    return structures
structures_counter(ham_emails).most_common()
structures_counter(spam_emails).most_common()
for header, value in spam_emails[0].items():

    print(header, ":", value)
spam_emails[0]["Subject"]
import numpy as np

from sklearn.model_selection import train_test_split



X = np.array(ham_emails + spam_emails)

y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
from bs4 import BeautifulSoup

def html_to_plain_text_soup(html):

    soup = BeautifulSoup(html, 'html.parser')

    for s in soup('head'):

        s.extract()# drop the head section

    
import re

from html import unescape



def html_to_plain_text(html):

    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)

    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)

    text = re.sub('<.*?>', '', text, flags=re.M | re.S)

    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)

    return unescape(text)
html_spam_emails = [email for email in X_train[y_train == 1]

                   if get_email_structure(email) == "text/html"]

sample_html_spam = html_spam_emails[0]

print(sample_html_spam.get_content().strip())
print(html_to_plain_text(sample_html_spam.get_content()))
def email_to_text(email):

    html = None

    for part in email.walk():

        ctype = part.get_content_type()

        if not ctype in ("text/plain", "text/html"):

            continue

        try:

            content = part.get_content()

        except:  # if problem arises due to encoding

            content = str(part.get_payload())

        if ctype == "text/plain":

            return content

        else:

            html = content

    if html:

        return html_to_plain_text(html)
print(email_to_text(sample_html_spam)[:100])
import nltk

stemmer = nltk.PorterStemmer()

for word in ("Walk", "walks", "walked", "walking", "walkable"):

    print(word, "=>", stemmer.stem(word))
!pip install urlextract
import urlextract

url_extractor = urlextract.URLExtract()

print(url_extractor.find_urls("Which is better github.com or stackoverflow.com"))
from sklearn.base import BaseEstimator, TransformerMixin

# import urlextract

# url_extractor = urlextract.URLExtract()

class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, strip_headers = True, lower_case = True, remove_punctuation=True, 

                replace_urls = True, replace_numbers=True, stemming = True):

        self.strip_headers = strip_headers

        self.lower_case = lower_case

        self.remove_punctuation = remove_punctuation

        self.replace_urls = replace_urls

        self.replace_numbers = replace_numbers

        self.stemming = stemming

    def fit(self, X,y=None):

        return self

    def transform(self,X,y=None):

        X_transformed = []

        for email in X:

            text = email_to_text(email) or ""

            if self.lower_case:

                text = text.lower()

            if self.replace_urls and url_extractor is not None:

                urls = list(set(url_extractor.find_urls(text)))

                urls.sort(key = lambda url: len(url), reverse=True)

                for url in urls:

                    text = text.replace(url, " URL ")

            if self.replace_numbers:

                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)

            if self.remove_punctuation:

                text = re.sub(r'\W+', ' ', text, flags=re.M)

            word_counts = Counter(text.split())

            if self.stemming and stemmer is not None:

                stemmed_word_counts = Counter()

                for word, count in word_counts.items():

                    stemmed_word = stemmer.stem(word)

                    stemmed_word_counts[stemmed_word]+= count

                word_counts = stemmed_word_counts

            X_transformed.append(word_counts)

        return np.array(X_transformed)

                
X_few = X_train[:3]

X_few_word_counts = EmailToWordCounterTransformer().fit_transform(X_few)

X_few_word_counts
from scipy.sparse import csr_matrix



class WordCoutnerToVectorTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, vocabulary_size = 1000):

        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):

        total_count = Counter()

        for word_count in X:

            for word, count in word_count.items():

                total_count[word] += count # or min(count, 10)

        most_common = total_count.most_common()[:self.vocabulary_size]

        self.most_common = most_common

        self.vocabulary_ = {word: index+1 for index, (word, count) in enumerate(most_common)}

        return self

    def transform(self, X, y=None):

        rows, cols, data = [],[],[]

        for row, word_count in enumerate(X):

            for word, count in word_count.items():

                rows.append(row)

                cols.append(self.vocabulary_.get(word, 0))

                data.append(count)

        return csr_matrix((data, (rows, cols)), shape= (len(X), self.vocabulary_size+1))
vocab_transformer = WordCoutnerToVectorTransformer(vocabulary_size=10)

X_few_vectors = vocab_transformer.fit_transform(X_few_word_counts)

X_few_vectors
X_few_vectors.toarray()
vocab_transformer.vocabulary_
from sklearn.pipeline import Pipeline

preprocess_pipeline = Pipeline([

    ("email_to_wordcount", EmailToWordCounterTransformer()),

    ("wordcount_to_vector", WordCoutnerToVectorTransformer()),

])



X_train_transformed = preprocess_pipeline.fit_transform(X_train)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



log_clf = LogisticRegression(solver="lbfgs", random_state=42)

score = cross_val_score(log_clf, X_train_transformed, y_train, cv = 3, verbose = 3)

score.mean()
from sklearn.metrics import precision_score, recall_score



X_test_transformed = preprocess_pipeline.transform(X_test)



log_clf = LogisticRegression(solver = "lbfgs", random_state = 42, max_iter=1000000)

log_clf.fit(X_train_transformed, y_train)



y_pred = log_clf.predict(X_test_transformed)



print("Precision: {:.2f}%".format(100*precision_score(y_test, y_pred)))

print("Recall: {:.2f}%".format(100*recall_score(y_test, y_pred)))