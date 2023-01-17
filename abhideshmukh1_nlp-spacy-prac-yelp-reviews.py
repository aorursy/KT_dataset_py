# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import spacy



parser = spacy.load('en')



#Test Data

df = pd.read_table("../input/yelp_labelled.txt")



#There is an art, it says, or rather, a knack to flying." \

#                 "The knack lies in learning how to throw yourself at the ground and miss." \

#                 "In the beginning the Universe was created. This has made a lot of people "\

#                 "very angry and been widely regarded as a bad move."

df.columns=['Message', 'Target']

print(df.head())
df.shape

df.to_csv("sentimentdataset.csv")

df.isnull().sum()
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en')
stopwords = list(STOP_WORDS)

stopwords
import string

punctuations = string.punctuation



from spacy.lang.en import English

parser = English()



def spacy_tokenizer(sentence):

    mytokens = parser(sentence)

    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]

    mytokens = [word for word in mytokens if word not in stopwords and word not in punctuations]

    return mytokens
#Let's do some machine learning

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score

from sklearn.base import TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

#from sklearn.linear_model import SDClassifier

from sklearn.linear_model import LogisticRegression
#Vectorization

vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1), binary=True)

classifier = LinearSVC()

rfc = RandomForestClassifier()



print(vectorizer)
#splitting the data

from sklearn.model_selection import train_test_split

X = df.Message

y = df.Target



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
pipe = Pipeline([ 

                ("the vectorizer", vectorizer),

                ("the classifier", classifier),

                ])
pipe.fit(X_train, y_train)
sample_prediction = pipe.predict(X_test)
#Prediction Score

for (sample, pred) in zip(X_test, sample_prediction):

    print(sample, "Prediction => ", pred)
from sklearn.metrics import roc_curve, auc



false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, sample_prediction)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc

print("ROC_AUC: ", roc_auc)

print("Accuracy: ", pipe.score(X_test, y_test))

#print("Accuracy: ", pipe.score(sample_prediction, y_test))
#randomizedsearchCV for hyperparameter tuning
