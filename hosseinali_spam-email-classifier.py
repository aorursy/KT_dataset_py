import os

HAM_DIR = "../input/ham-and-spam-dataset/hamnspam/ham/"

SPAM_DIR = "../input/ham-and-spam-dataset/hamnspam/spam/"

ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]

spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]
print('Number of ham files:' , len(ham_filenames) )

print('Number of spam files:' , len(spam_filenames) )
import email

import email.policy



def load_email( is_spam, filename ):

    directory = SPAM_DIR if is_spam else HAM_DIR

    with open(os.path.join( directory, filename ), "rb") as f:

        return email.parser.BytesParser(policy=email.policy.default).parse(f)



ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]

spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]
import numpy as np

from sklearn.model_selection import train_test_split



X = np.array(ham_emails + spam_emails)

y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import re

from html import unescape



def html_to_plain_text(html):

    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)

    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)

    text = re.sub('<.*?>', '', text, flags=re.M | re.S)

    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)

    return unescape(text)



def email_to_text(email):

    html = None

    for part in email.walk():

        ctype = part.get_content_type()

        if not ctype in ("text/plain", "text/html"):

            continue

        try:

            content = part.get_content()

        except: # in case of encoding issues

            content = str(part.get_payload())

        if ctype == "text/plain":

            return content

        else:

            html = content

    if html:

        return html_to_plain_text(html)
import nltk

stemmer = nltk.PorterStemmer()



from collections import Counter



import re

from sklearn.base import BaseEstimator, TransformerMixin



class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,

                 replace_urls=True, replace_numbers=True, stemming=True):

        self.strip_headers = strip_headers

        self.lower_case = lower_case

        self.remove_punctuation = remove_punctuation

        self.replace_urls = replace_urls

        self.replace_numbers = replace_numbers

        self.stemming = stemming

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        X_transformed = []

        for email in X:

            text = email_to_text(email) or ""

            if self.lower_case:

                text = text.lower()

            if self.replace_urls:

                urls = list(set(re.findall( r'(https?://\S+)' , text )))

                urls.sort(key=lambda url: len(url), reverse=True)

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

                    stemmed_word_counts[stemmed_word] += count

                word_counts = stemmed_word_counts

            X_transformed.append(word_counts)

        return np.array(X_transformed)
from scipy.sparse import csr_matrix



class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, vocabulary_size=1000):

        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):

        total_count = Counter()

        for word_count in X:

            for word, count in word_count.items():

                total_count[word] += min(count, 10)

        most_common = total_count.most_common()[:self.vocabulary_size]

        self.most_common_ = most_common

        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}

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

        solu = csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))

        return solu
from sklearn.pipeline import Pipeline



preprocess_pipeline = Pipeline([

    ("email_to_wordcount", EmailToWordCounterTransformer()),

    ("wordcount_to_vector", WordCounterToVectorTransformer()),

])



X_train_transformed = preprocess_pipeline.fit_transform(X_train)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



log_clf = LogisticRegression(solver="liblinear", random_state=42)

score = cross_val_score(log_clf, X_train_transformed, y_train, cv=3, verbose=3)

print('\nScores for all folds: ', score )

print('\nAverage Score: ', score.mean() )

print('\nStandard deviation of Scores: ', score.std() )
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_predict



y_scores = cross_val_predict(log_clf, X_train_transformed, y_train, cv=3, method="decision_function")



from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
import matplotlib.pyplot as plt

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)

    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)

    plt.xlabel("Threshold", fontsize=16)

    plt.legend(loc="upper left", fontsize=16)

    plt.ylim([0, 1])



plt.figure(figsize=(8, 4))

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.show()
def plot_precision_vs_recall(precisions, recalls):

    plt.plot(recalls, precisions, "b-", linewidth=2)

    plt.xlabel("Recall", fontsize=16)

    plt.ylabel("Precision", fontsize=16)

    plt.axis([0, 1, 0, 1])



plt.figure(figsize=(8, 6))

plot_precision_vs_recall(precisions, recalls)

plt.show()