# Load packages

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.base import TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.svm import OneClassSVM

from sklearn.utils import shuffle

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import classification_report 

from nltk.corpus import stopwords

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.stem.porter import PorterStemmer

import string

import spacy

from spacy.lang.en import English

spacy.load('en')

parser = English()
# load dataset

bbc_df = pd.read_csv('../input/bbc-text.csv')
bbc_df.head(10)
bbc_df.shape
bbc_df.info()
bbc_df['category'].unique()
bbc_df['category'].value_counts()
sns.countplot(bbc_df['category'])
# change category labels

bbc_df['category'] = bbc_df['category'].map({'sport':1,'business':-1,'politics':-1,'tech':-1,'entertainment':-1})
# create a new dataset with only sport category data

sports_df = bbc_df[bbc_df['category'] == 1]
sports_df.shape
# create train and test data

train_text = sports_df['text'].tolist()

train_labels = sports_df['category'].tolist()



test_text = bbc_df['text'].tolist()

test_labels = bbc_df['category'].tolist()
# stop words list

STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS)) 

# special characters

SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”","''"]
# class for cleaning the text

class CleanTextTransformer(TransformerMixin):

    def transform(self, X, **transform_params):

        return [cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):

        return self

    def get_params(self, deep=True):

            return {}



def cleanText(text):

    text = text.strip().replace("\n", " ").replace("\r", " ")

    text = text.lower()

    return text
# tokenizing the raw text

def tokenizeText(sample):

    

    tokens = parser(sample)

    

    # lemmatization

    lemmas = []

    for tok in tokens:

        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)

    tokens = lemmas

    

    # remove stop words and special characters

    tokens = [tok for tok in tokens if tok.lower() not in STOPLIST]

    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    

    # only take words with length greater than or equal to 3

    tokens = [tok for tok in tokens if len(tok) >= 3]

    

    # remove remaining tokens that are not alphabetic

    tokens = [tok for tok in tokens if tok.isalpha()]

    

    # stemming of words

    porter = PorterStemmer()

    tokens = [porter.stem(word) for word in tokens]

    

    return list(set(tokens))
# lets see tokenized random text

tokenizeText(train_text[9])
# getting features

vectorizer = HashingVectorizer(n_features=20,tokenizer=tokenizeText)



features = vectorizer.fit_transform(train_text).toarray()

features.shape
# OneClassSVM algorithm

clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

pipe_clf = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
# fit OneClassSVM model 

pipe_clf.fit(train_text, train_labels)
# validate OneClassSVM model with train set

preds_train = pipe_clf.predict(train_text)



print("accuracy:", accuracy_score(train_labels, preds_train))
# validate OneClassSVM model with test set

preds_test = pipe_clf.predict(test_text)

preds_test
results = confusion_matrix(test_labels, preds_test) 

print('Confusion Matrix :')

print(results) 

print('Accuracy Score :',accuracy_score(test_labels, preds_test)) 

print('Report : ')

print(classification_report(test_labels, preds_test)) 
# let's take random text from dataset

test_text[3]
# check actual category

test_labels[3]
# let's predict the category of above random text

pipe_clf.predict([test_text[3]])