from IPython.display import YouTubeVideo      
YouTubeVideo('fKCayvKYqm8')
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin  #custom transform
from sklearn.pipeline import Pipeline
import string  #in this use case,it will handle punctuation mark
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
df_amazon = pd.read_csv("../input/amazon-alexa-reviews/amazon_alexa.tsv", sep="\t")
df_amazon.head()
df_amazon.shape
df_amazon.feedback.value_counts()
# Create our list of punctuation marks
punctuations = string.punctuation
print(punctuations)
# Create our list of stopwords
nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English() #comes from import lang.en

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase , strip all extra spaces,
    # pron is like he,she,i ==> so != pron 

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens
# Custom transformer using spaCy
class predictors(TransformerMixin):

    def fit(self, X, y=None, **fit_params):      #uses the input data to train the transformer
        return self
    
    def transform(self, X, **transform_params):  #takes the input feature and transform them
        # Cleaning Text
        return [clean_text(text) for text in X]

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)
bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
from sklearn.model_selection import train_test_split

X = df_amazon['verified_reviews'] # the features we want to analyze
ylabels = df_amazon['feedback']   # the labels, or answers, we want to test against

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(class_weight='balanced')
# Create pipeline using CountVectoe
pipe1 = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])

# model generation
pipe1.fit(X_train,y_train)
# Create pipeline using tf-idf
pipe2 = Pipeline([("cleaner", predictors()),
                 ('vectorizer', tfidf_vector),
                 ('classifier', classifier)])

# model generation
pipe2.fit(X_train,y_train)
from sklearn import metrics
# Predicting with a test dataset
predicted = pipe1.predict(X_test)

# Model Accuracy
print(" Accuracy:",metrics.accuracy_score(y_test, predicted))
print(" Precision:",metrics.precision_score(y_test, predicted))
print(" Recall:",metrics.recall_score(y_test, predicted))
# Predicting with a test dataset
predicted2 = pipe2.predict(X_test)

# Model Accuracy
print(" Accuracy:",metrics.accuracy_score(y_test, predicted2))
print(" Precision:",metrics.precision_score(y_test, predicted2))
print(" Recall:",metrics.recall_score(y_test, predicted2))

