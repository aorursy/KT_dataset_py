# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re # reguliar expression

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('../input/nlp-getting-started/train.csv')

test_data  =pd.read_csv('../input/nlp-getting-started/test.csv')

train_data.head(10)
train_data.dtypes
train_data[train_data['target']==1]["text"]
train_data[train_data['target']==0]["text"]
import re

def  clean_text(df, text_field, new_text_field_name):

    df[new_text_field_name] = df[text_field].str.lower()

    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  

    # remove numbers

    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))

    return df



data_clean = clean_text(train_data, 'text', 'text_clean')

data_clean_tr = clean_text(train_data, 'text', 'text_clean')

data_clean_ts = clean_text(test_data, 'text', 'text_clean')
data_clean.head()
import nltk

from nltk.corpus import stopwords

print(stopwords.words('english'))
#Cleaning text

import nltk.corpus

nltk.download('stopwords')

from nltk.corpus import stopwords

stop = stopwords.words('english')

data_clean_tr['text_clean'] = data_clean_tr['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

data_clean_ts['text_clean'] = data_clean_ts['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))



data_clean.head()
#text tokenization

import nltk 

nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize

data_clean['text_tokens'] = data_clean['text_clean'].apply(lambda x: word_tokenize(x))

data_clean.head()
# Stemming words with NLTK

from nltk.stem import PorterStemmer 

from nltk.tokenize import word_tokenize

def word_stemmer(text):

    stem_text = [PorterStemmer().stem(i) for i in text]

    return stem_text

data_clean['text_clean_tokens'] = data_clean['text_tokens'].apply(lambda x: word_stemmer(x))

data_clean.head()
# text lemmatisation 

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

def word_lemmatizer(text):

    lem_text = [WordNetLemmatizer().lemmatize(i) for i in text]

    return lem_text

data_clean['text_clean_tokens'] = data_clean['text_tokens'].apply(lambda x: word_lemmatizer(x))

data_clean['text_clean_tokens'] = data_clean['text_clean_tokens'].astype(str)

data_clean.head()
#Tokenization vs Stemming & Lemmatization

X1 = data_clean['text_tokens'][0] 

X2 = data_clean['text_clean_tokens'][0]

print(X1,X2, sep='\n')
# example of a disaster tweet

data_clean[data_clean_tr["target"] == 1]["text_clean"].values[1]
# example of what is NOT a disaster tweet.

data_clean[data_clean_tr["target"] == 0]["text_clean"].values[1]
X = data_clean_tr.iloc[ : ,5]

y = data_clean_tr.iloc[ : ,4]
# Splitting the training data into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#Text Vectorization using TfidfVectorizer //Convert a collection of raw documents to a matrix of TF-IDF features.

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)
# Training the Logistic Regression model on the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(C = 0.1)

classifier.fit(X_train, y_train)
#Explain the linear model

import shap

explainer = shap.LinearExplainer(classifier, X_train, feature_dependence="independent")

shap_values = explainer.shap_values(X_test)

X_test_array = X_test.toarray() # we need to pass a dense version for the plotting functions
#Summarize the effect of all the features

shap.summary_plot(shap_values, X_test_array, feature_names=vectorizer.get_feature_names())
from sklearn.metrics import classification_report

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
#Confusion Matrix Visualisation

import matplotlib.pyplot as plt 

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(classifier, X_test, y_test) 

plt.show()
from sklearn import feature_extraction

count_vectorizer = feature_extraction.text.TfidfVectorizer()



## let's get counts for the first 5 tweets in the data

example_train_vectors = count_vectorizer.fit_transform(data_clean_tr["text"][0:5])
## we use .todense() here because these vectors are "sparse" (only non-zero elements are kept to save space)

print(example_train_vectors[0].todense().shape)

print(example_train_vectors[0].todense())
train_vectors = count_vectorizer.fit_transform(data_clean_tr["text_clean"])



## note that we're NOT using .fit_transform() here. Using just .transform() makes sure

# that the tokens in the train vectors are the only ones mapped to the test vectors - 

# i.e. that the train and test vectors use the same set of tokens.

test_vectors = count_vectorizer.transform(data_clean_ts["text_clean"])
# Training the Logistic Regression model on all the Training set

from sklearn.ensemble import AdaBoostClassifier

classifier = AdaBoostClassifier(n_estimators=100, base_estimator= None,learning_rate=1, random_state = 1)

classifier.fit(train_vectors, data_clean_tr["target"])
from sklearn import model_selection

scores = model_selection.cross_val_score(classifier, train_vectors, train_data["target"], cv=3, scoring="f1")

scores
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
y_pred  = classifier.predict(test_vectors)
sample_submission["target"] = y_pred
sample_submission.to_csv("submission.csv", index=False)

sample_submission.head()