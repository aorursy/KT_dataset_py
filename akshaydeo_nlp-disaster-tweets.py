# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/nlp-getting-started/train.csv')
test_df = pd.read_csv('../input/nlp-getting-started/test.csv')
print(train_df.head(35))
print("train shape : ",train_df.shape)
print("test shape",test_df.shape)
#If we want to use columns keyword and location along with test column. Then we can fill missing values in these with some constant string.
# For now we will be using only text column for TFIDF Vectorizer
train_df.isnull().sum()
test_df.isnull().sum()
#EDA
#Checking Target Class Balance/Imbalance
sns.barplot(x=[0,1], y = train_df['target'].value_counts())
#Analyse char length vs target
train_df['char_len'] = train_df['text'].apply(len)

plt.figure('Char length vs Target label 0 and 1 Histogram and kde')

sns.distplot(train_df[train_df['target'] == 0]['char_len'].values, bins = 20, label = 'label 0 hist')
sns.distplot(train_df[train_df['target'] == 1]['char_len'].values, bins = 20, label = 'label 1 hist')

plt.xlabel("Character length")
plt.ylabel("Density")
plt.legend(loc="best")
plt.show()

#We can see a relation, so we keep the column
#Analyse char length vs target
train_df['word_len'] = train_df['text'].apply(lambda x : len(x.split()))

plt.figure('word_len vs Target label 0 and 1 Histogram and kde')

sns.distplot(train_df[train_df['target'] == 0]['word_len'].values, bins = 20, label = 'label 0 hist')
sns.distplot(train_df[train_df['target'] == 1]['word_len'].values, bins = 20, label = 'label 1 hist')

plt.xlabel("word length")
plt.ylabel("Density")
plt.legend(loc="best")
plt.show()

#We can see a relation, so we keep the column
#Emojis

#Replace emojis in train and test df with demojized text
import emoji
import regex as re

train_df['text'] = train_df['text'].apply(lambda x : emoji.demojize(x))
test_df['text'] = test_df['text'].apply(lambda x : emoji.demojize(x))
print(train_df[train_df['text'].str.match(':(.*?):')]['text'])
str = test_df[test_df['text'].str.match(':(.*?):')]['text'].values
print(str)
#Hashtags contain a lot of information so we keep them. We will just separate # from the word so that the words can be used.

#We also separate all the punctuations from the words so that they can be used

def separate_punc(word):
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    for punc in punctuations:
        return word.replace(punc, f' {punc} ')

train_df['text'] = train_df['text'].apply(separate_punc)



 
#We will split the train data into train and validation set and then apply predictions of our model on test set

from sklearn.model_selection import train_test_split

train_set, val_set  =  train_test_split(train_df,random_state=0,stratify = train_df['target'].values)


# Convert strings into vectors using Bag of words

#The theory behind the model we'll build in this notebook is pretty simple: the words contained in each tweet are a good indicator of whether they're about a real disaster or not (this is not entirely correct, but it's a great place to start).

#We'll use scikit-learn's CountVectorizer to count the words in each tweet and turn them into data our machine learning model can process.

#Note: a vector is, in this context, a set of numbers that a machine learning model can work with. 



from nltk.corpus import stopwords
stop = list(stopwords.words('english'))

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer(decode_error = 'replace',stop_words = stop)

#Features which is text
X_train = vectorizer.fit_transform(train_set['text'].values)
X_val = vectorizer.transform(val_set['text'].values)

# Eg. If There are 54 unique words (or "tokens") in the first five tweets.
#The fit transformed data of The first tweet contains only some of those unique tokens - all of the non-zero counts above are the tokens that DO exist in the first tweet.

#Targets which are 0 and 1
y_train = train_set.target.values
y_val = val_set.target.values

print("X_train.shape : ", X_train.shape)
print("X_val.shape : ", X_val.shape)
print("y_train.shape : ", y_train.shape)
print("y_valid.shape : ", y_val.shape)



#Baseline Model : Naive Bayes

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import datetime

baseline_clf = MultinomialNB()
baseline_clf.fit(X_train,y_train)

baseline_prediction = baseline_clf.predict(X_val)
baseline_accuracy = accuracy_score(y_val,baseline_prediction)
print("training accuracy Score    : ",baseline_clf.score(X_train,y_train))
print("Validdation accuracy Score : ",baseline_accuracy )

f = open("log.txt","w+")
f.write("Date : %s "%datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)"))
f.write("training accuracy Score    : %s" %baseline_clf.score(X_train,y_train))
f.write("Validdation accuracy Score : %s" %baseline_accuracy)
f.write("-------------------------------")
f.close()


test_vectors = vectorizer.transform(test_df["text"].values)

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = baseline_clf.predict(test_vectors)
sample_submission.to_csv("submission.csv", index=False)