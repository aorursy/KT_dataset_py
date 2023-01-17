import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

from sklearn.base import TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics





import string

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English


sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
train.info()
train.describe()
train.head()
X=train['text']

y=train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



print(f"Train X: {X_train.shape}, y: {y_train.shape}, TEST  X: {X_test.shape}, y: {y_test.shape}")
punctuations = string.punctuation



nlp = English()

stop_words = STOP_WORDS



parser = English()



# Basic tokenizer function

def spacy_tokenizer(sentence):

    mytokens = parser(sentence)

    # Lemmatization - rip words to their lemma equivalent

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Rip out the stop words and punctuations

    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    return mytokens
# Simple transformer implementation to clean the text. 

# This might do something a bit more extensive in a real application like using https://pypi.org/project/tweet-preprocessor/

class TextPreprocessor(TransformerMixin):

    def transform(self, X, **transform_params):

        return [clean_text(text) for text in X]



    def fit(self, X, y=None, **fit_params):

        return self



    def get_params(self, deep=True):

        return {}



def clean_text(text):

    return text.strip().lower()
bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

tfidf = TfidfTransformer()



classifier = LogisticRegression()



pipe = Pipeline([("cleaner", TextPreprocessor()),

                 ('vectorizer', bow_vector),

                 ('tfid', tfidf),

                 ('classifier', classifier)])



pipe.fit(X_train,y_train)
predicted = pipe.predict(X_test)

pred_1_or_0 = np.where(predicted>=0.5, 1, 0)

# Model Accuracy

print("Accuracy:",metrics.accuracy_score(y_test, predicted))

print("Precision:",metrics.precision_score(y_test, predicted))

print("Recall:",metrics.recall_score(y_test, predicted))

print("AUC:",metrics.roc_auc_score(y_test, pred_1_or_0))

print("AUC-PR:",metrics.average_precision_score(y_test, predicted))

print("F1:",metrics.f1_score(y_test, predicted))