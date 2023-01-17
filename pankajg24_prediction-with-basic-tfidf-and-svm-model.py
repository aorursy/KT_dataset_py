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
import nltk

import matplotlib.pyplot as plt

from nltk.stem import PorterStemmer

from nltk.corpus import stopwords

import re

import string

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC, LinearSVC

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
data.head()
def preprocess(tweets):

   

    # put data to lower case 

    tweets_lower = [tweet.lower() for tweet in tweets]

    

    #remove links

    tweets_re = [re.sub(r'http\S+', '', tweet) for tweet in tweets_lower]

    #Remove Hashtags and @name

    #tweets_re = [re.sub(r'@\w+', '', tweet) for tweet in tweets_re]

    #tweets_re = [re.sub(r'#\w+', '', tweet) for tweet in tweets_re]

    

    

    #tokenize the tweets and remove punctuations and stop words

    stop_words = stopwords.words('english')

    

    tweet_token = []

    ps = PorterStemmer()

    clean_tweet = []

    for tweet in tweets_re:

            words = nltk.word_tokenize(tweet)

            #remove Stopwords / Punctuations & Special characters

            tweet_token = [ps.stem(word) for word in words if word not in stop_words and word not in string.punctuation] # and word.isalnum()]

            tweet_sent = ' '.join(tweet_token)

            clean_tweet.append(tweet_sent)

            

    return clean_tweet
print(preprocess(data.text[:3]))
#temp = data[(data.id > 444) & (data.id < 450)]

temp = data[(data.id == 445)] # & (data.id < 450)]

print(preprocess(temp.text))

temp.columns
data_clean = preprocess(data.text)

target = data.target 

data_clean[:5]
x_train, x_test, y_train, y_test = train_test_split(data_clean, target, test_size = 0.2, 

                                                    stratify = target, random_state = 123)
x_train[:5]
tfidf = TfidfVectorizer().fit(x_train)
tfidf_train = tfidf.transform(x_train)

tfidf_test = tfidf.transform(x_test)
print(tfidf_train.shape)

print(tfidf_test.shape)
svm_gs = LinearSVC(class_weight= 'balanced', random_state= 123)

params = {'C' : [1, 5, 10, 15]}

gs_model = GridSearchCV(svm_gs, param_grid= params, scoring= 'recall', cv = 5)
gs_model.fit(tfidf_train, y_train)

gs_model.best_estimator_

gs_model.best_params_

gs_model.best_score_

gs_model.cv_results_
#gs_predict = gs_model.predict(tfidf_train)

#gs_train_acc_score = accuracy_score(y_train, gs_predict)

#gs_train_con_mat = confusion_matrix(y_train, gs_predict)

#print(gs_train_acc_score)

#print(gs_train_con_mat)
#gs_predict_test = gs_model.predict(tfidf_test)

#gs_test_acc_score = accuracy_score(y_test, gs_predict_test)

#gs_test_con_mat = confusion_matrix(y_test, gs_predict_test)

#print(gs_test_acc_score)

#print(gs_test_con_mat)
#print(classification_report(y_test, gs_predict_test))
#svm_gs = SVC(C = 5, class_weight= 'balanced', random_state= 123)

svm_gs = LinearSVC(random_state= 123)
svm_gs.fit(tfidf_train, y_train)
predict_train_gs = svm_gs.predict(tfidf_train)

accuracy_score(y_train, predict_train_gs)
confusion_matrix(y_train, predict_train_gs)
predict_test_gs = svm_gs.predict(tfidf_test)

accuracy_score_test = accuracy_score(y_test, predict_test_gs)

conf_mat_test = confusion_matrix(y_test, predict_test_gs)
print(accuracy_score_test)

print(conf_mat_test)
pred_out = {}

pred_out['text'] = x_test

pred_out['target'] = y_test

pred_out['predicted'] = predict_test_gs



test_res = pd.DataFrame(pred_out)
test_res.head()
test_res.to_csv('/kaggle/working/test_res.csv', index= None)
test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
test = preprocess(test_data.text)
tfidf_test = tfidf.transform(test)
test_predict = svm_gs.predict(tfidf_test)
out_dict = {}

out_dict['id'] = test_data.id

out_dict['target'] = test_predict

output = pd.DataFrame(out_dict)
output.to_csv('/kaggle/working/result1.csv', index= None)