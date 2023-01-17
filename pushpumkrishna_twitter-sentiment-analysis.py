# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#for data wrangling and manipulation

import pandas as pd
import numpy as np

#for NLP text processing and formatting

import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# For word lemmitization
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# for word Stemming
from nltk.stem.porter import PorterStemmer

# for Machine Learning process

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# for Machine Learning model evaluation

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Global Parameters
stop_words = set(stopwords.words('english'))
def preprocess_tweet_text(tweet):
    """
    Function to process the the tweet text and tranform it into format usable by Machine learning models
    """
    
    # to convert all the characters of the tweet into lower case alphabets
    tweet.lower()
    
    # Remove urls from the tweets
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    
    # Remove user related references from the tweets:: '@' and '#' 
    tweet = re.sub(r'\@\w+|\#','', tweet)
    
    # Remove punctuations from the tweets
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stopwords from the tweets
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    joined_text = " ".join(filtered_words)
    
    return joined_text
def get_feature_vector(train_fit):
    """
    Function to Convert a collection of raw documents to a matrix of TF-IDF features.
    TF-IDF - Term Frequency Inverse Documnet Frequency
    """
    
    vector = TfidfVectorizer(sublinear_tf=True)      # Defining the vector
    vector.fit(train_fit)                            # fitting the data into the vector
    return vector                                    # returning the vector as function call
# read data
dataset = pd.read_csv("../input/sentiment-analysis-of-tweets/train.txt",  sep = ",")
print("Train Data has been read")
test = pd.read_csv("../input/sentiment-analysis-of-tweets/test_samples.txt",  sep = ",")
print("Test Data has been read")
# Preprocessing data before feeding it to ML models

processed_text = dataset['tweet_text'].apply(preprocess_tweet_text)

print("Processed text :: \n\n", processed_text)
stemmer = PorterStemmer()

stemmed_words = [stemmer.stem(i) for i in processed_text]
lemmatizer = WordNetLemmatizer()
lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]

tf_vector = get_feature_vector(np.array(dataset["tweet_text"]).ravel())
X = tf_vector.transform(np.array(dataset["tweet_text"]).ravel())     # Predictor Variable
y = np.array(dataset["sentiment"]).ravel()                           # Target varaible
# SPlitting the data into training and testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
# Using Naive Bayes Model :

NB_model = MultinomialNB()

NB_model.fit(X_train, y_train)
# Predicting the values and the Accuracy Score

y_predict_nb = NB_model.predict(X_test)

print("Accuracy Score for Naive Bayes Model is :: ", accuracy_score(y_test, y_predict_nb))
# Classification Report :

print("Classification_Report :: \n\n", classification_report(y_test, y_predict_nb))
# Training Logistics Regression model
LR_model = LogisticRegression(solver='lbfgs')
LR_model.fit(X_train, y_train)
# Predicting the Values :

y_predict_lr = LR_model.predict(X_test)

print("Accuracy Score for Logistic Regression Model is :: ",accuracy_score(y_test, y_predict_lr))
# Classification Report

from sklearn.metrics import classification_report

print("Classification_Report :: \n\n", classification_report(y_test, y_predict_lr))
# Creating text feature of test data :

test.tweet_text = test["tweet_text"].apply(preprocess_tweet_text)

test_feature = tf_vector.transform(np.array(test['tweet_text']).ravel())

# Using Naive Bayes Model for Prediction ::

test_prediction_nb = NB_model.predict(test_feature)

test_prediction_nb
# Creating a Dataframe consising tweets and sentiment in a submission format

submission_result_nb = pd.DataFrame({'tweet_id': test.tweet_id, 'sentiment':test_prediction_nb})
submission_result_nb
# Total number os tweets grouped according sentiment

test_result = submission_result_nb['sentiment'].value_counts()
test_result
#Using Logistic Regression Model for Prediction ::

test_prediction_lr = LR_model.predict(test_feature)

test_prediction_lr
# Creating a Dataframe consising tweets and sentiment

submission_result_lr = pd.DataFrame({'tweet_id': test.tweet_id, 'sentiment':test_prediction_lr})
submission_result_lr
# Total number os tweets grouped according sentiment

test_result2 = submission_result_lr['sentiment'].value_counts()
test_result2
import seaborn as sns
sns.countplot(submission_result_lr['sentiment'])
import seaborn as sns
sns.countplot(submission_result_nb['sentiment'])



