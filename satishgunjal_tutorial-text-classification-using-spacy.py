import spacy

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.base import TransformerMixin

from sklearn.pipeline import Pipeline
# Loading TSV file

df_amazon = pd.read_csv("/kaggle/input/amazon-alexa-reviews/amazon_alexa.tsv",sep="\t")



print(f'Shape of data: {df_amazon.shape}')

# Show top 5 records

df_amazon.head()
df_amazon.info()
df_amazon.feedback.value_counts()
import string

from spacy.lang.en import English

from spacy.lang.en.stop_words import STOP_WORDS



# Create our list of punchuationmarks

punctuations = string.punctuation



# Create our list of stop words

nlp = spacy.load('en')

stop_words = spacy.lang.en.stop_words.STOP_WORDS



# Load English tokenizer, tagger, parser, NER and word vector

parser = English()



# Creating our tokenzer function

def spacy_tokenizer(sentence):

    """This function will accepts a sentence as input and processes the sentence into tokens, performing lemmatization, 

    lowercasing, removing stop words and punctuations."""

    

    # Creating our token object which is used to create documents with linguistic annotations

    mytokens = parser(sentence)

    

    # lemmatizing each token and converting each token in lower case

    # Note that spaCy uses '-PRON-' as lemma for all personal pronouns lkike me, I etc

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    

    # Removing stop words

    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations]

    

    # Return preprocessed list of tokens

    return mytokens    
# Custom transformer using spaCy

class predictors(TransformerMixin):

    def transform(self, X, **transform_params):

        """Override the transform method to clean text"""

        return [clean_text(text) for text in X]

    

    def fit(self, X, y= None, **fit_params):

        return self

    

    def get_params(self, deep= True):

        return {}



# Basic function to clean the text

def clean_text(text):

    """Removing spaces and converting the text into lowercase"""

    return text.strip().lower()    
bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range = (1,1))
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)
from sklearn.model_selection import train_test_split



X = df_amazon['verified_reviews'] # The features we want to analyse

ylabels = df_amazon['feedback'] # The labels, in this case feedback



X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size = 0.3, random_state = 1)

print(f'X_train dimension: {X_train.shape}')

print(f'y_train dimension: {y_train.shape}')

print(f'X_test dimension: {X_test.shape}')

print(f'y_train dimension: {y_test.shape}')
# Logistic regression classifier

from sklearn.linear_model import LogisticRegression



classifier = LogisticRegression()



# Create pipeline using Bag of Words

pipe = Pipeline ([("cleaner", predictors()),

                 ("vectorizer", bow_vector),

                 ("classifier", classifier)])



# Model generation

pipe.fit(X_train, y_train)
from sklearn import metrics



# Predicting with test dataset

predicted = pipe.predict(X_test)



# Model accuracy score

print(f'Logistic Regression Accuracy: {metrics.accuracy_score(y_test, predicted)}')

print(f'Logistic Regression Precision: {metrics.precision_score(y_test, predicted)}')

print(f'Logistic Regression Recall: {metrics.recall_score(y_test, predicted)}')