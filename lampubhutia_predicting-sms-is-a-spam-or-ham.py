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
import pandas as pd

import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn import naive_bayes

from sklearn.metrics import roc_auc_score
#nltk.download()
df= pd.read_csv("../input/sms_spam.csv")
df.head()
df.type.replace('spam', 1, inplace=True)
df.type.replace('ham', 0, inplace=True)
df.head()
df.shape
##Our dependent variable will be 'spam' or 'ham' 

y = df.type
df.text
#TFIDF Vectorizer

stopset = set(stopwords.words('english'))

vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)
#Convert df.txt from text to features

X = vectorizer.fit_transform(df.text)
X.shape
X.data
df.text[0]
## Spliting the SMS to separate the text into individual words

splt_txt1=df.text[0].split()

print(splt_txt1)
## Count the number of words in the first SMS

len(splt_txt1)
X[0]
print(X[0])
vectorizer.get_feature_names()[8585]## 4316 is the position of the word jurong
## Spliting the SMS to separate the text into individual words

splt_txt2=df.text[1].split()

print(splt_txt2)
len(splt_txt2)
X[1]## Second SMS
print (X[1])
## Finding the most frequent word appearing in the second SMS

max(splt_txt2)
## Last word in the vocabulary

max(vectorizer.get_feature_names())
len(vectorizer.get_feature_names())
print (y.shape)

print (X.shape)
##Split the test and train

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train
##Train Naive Bayes Classifier

## Fast (One pass)

## Not affected by sparse data, so most of the 8605 words dont occur in a single observation

clf = naive_bayes.MultinomialNB()

model=clf.fit(X_train, y_train)
clf.feature_log_prob_
clf.coef_
predicted_class=model.predict(X_test)

print(predicted_class)
print(y_test)
df.loc[[19]]
predicted_class[19]## This SMS(SMS no. 19) has been classified as Ham but Actually it's SPAM
prd=model.predict_proba(X_test)
prd
clf.predict_proba(X_test)[:,0]
roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])
##Check model's accuracy

roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
clf.coef_
def get_most_important_features(vectorizer, model, n=5):

    index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}

    

    # loop for each class

    classes ={}

    for class_index in range(model.coef_.shape[0]):

        word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]

        sorted_coeff = sorted(word_importances, key = lambda x : x[0], reverse=True)

        tops = sorted(sorted_coeff[:n], key = lambda x : x[0])

        bottom = sorted_coeff[-n:]

        classes[class_index] = {

            'tops':tops,

            'bottom':bottom

        }

    return classes



importance = get_most_important_features(vectorizer, clf, 20)
print (importance)
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

def plot_important_words(top_scores, top_words, bottom_scores, bottom_words, name):

    y_pos = np.arange(len(top_words))

    top_pairs = [(a,b) for a,b in zip(top_words, top_scores)]

    top_pairs = sorted(top_pairs, key=lambda x: x[1])

    

    bottom_pairs = [(a,b) for a,b in zip(bottom_words, bottom_scores)]

    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)

    

    top_words = [a[0] for a in top_pairs]

    top_scores = [a[1] for a in top_pairs]

    

    bottom_words = [a[0] for a in bottom_pairs]

    bottom_scores = [a[1] for a in bottom_pairs]

    fig = plt.figure(figsize=(10, 10))  



    plt.subplot(121)

    plt.barh(y_pos,bottom_scores, align='center', alpha=0.5)

    plt.title('Ham', fontsize=20)

    plt.yticks(y_pos, bottom_words, fontsize=14)

    plt.suptitle('Key words', fontsize=16)

    plt.xlabel('Importance', fontsize=20)

    

    plt.subplot(122)

    plt.barh(y_pos,top_scores, align='center', alpha=0.5)

    plt.title('Spam', fontsize=20)

    plt.yticks(y_pos, top_words, fontsize=14)

    plt.suptitle(name, fontsize=16)

    plt.xlabel('Importance', fontsize=20)

    

    plt.subplots_adjust(wspace=0.8)

    plt.show()

top_scores = [a[0] for a in importance[0]['tops']]

top_words = [a[1] for a in importance[0]['tops']]

bottom_scores = [a[0] for a in importance[0]['bottom']]

bottom_words = [a[1] for a in importance[0]['bottom']]



plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")