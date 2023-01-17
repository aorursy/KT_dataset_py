# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_orig=pd.read_csv("/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv")

test_nolabel=pd.read_csv("/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv")
from nltk.corpus import stopwords

from nltk import word_tokenize

import string

import re

stop_words = set(stopwords.words('english'))



train = train_orig



def remove_stopwords(line):

    word_tokens = word_tokenize(line)

    filtered_sentence = [w for w in word_tokens if not w in stop_words] 

    return " ".join(filtered_sentence)



def preprocess(line):

    line = line.lower()  #convert to lowercase

    line = re.sub(r'\d+', '', line)  #remove numbers

    line = line.translate(line.maketrans("","", string.punctuation))  #remove punctuation

#     line = line.translate(None, string.punctuation)  #remove punctuation

    line = remove_stopwords(line)

    return line

for i,line in enumerate(train.tweet):

    train.tweet[i] = preprocess(line)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(train['tweet'], train['label'], test_size=0.5, stratify=train['label'])



trainp=train[train.label==1]

trainn=train[train.label==0]

print(trainp.info())

trainn.info()
# Let us balance the dataset

train_imbalanced = train

from sklearn.utils import resample

df_majority = train[train.label==0]

df_minority = train[train.label==1]

 

# Upsample minority class

df_minority_upsampled = resample(df_minority, 

                                 replace=True,     # sample with replacement

                                 n_samples=len(df_majority),    # to match majority class

                                 random_state=123) # reproducible results

 

# Combine majority class with upsampled minority class

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

 

# Display new class counts

print("Before")

print(train.label.value_counts())

print("After")

print(df_upsampled.label.value_counts())



X_train, X_test, y_train, y_test = train_test_split(df_upsampled['tweet'], df_upsampled['label'], test_size=0.5, stratify=df_upsampled['label'])
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

# Xtext=train.tweet

# Xtest=test.tweet

# y=train.label

# test

# ytest=test.label
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vect = CountVectorizer()

tf_train=vect.fit_transform(X_train)  #train the vectorizer, build the vocablury

tf_test=vect.transform(X_test)  #get same encodings on test data as of vocabulary built
tf_test_nolabel=vect.transform(test_nolabel.tweet)
# print(tf_train)

# vect.get_feature_names()[:10] #print few features only to avoid slowing down the notebook
model.fit(X=tf_train,y=y_train)
expected = y_test

predicted=model.predict(tf_test)
from sklearn import metrics



print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))
from mlxtend.plotting import plot_confusion_matrix



plot_confusion_matrix(metrics.confusion_matrix(expected, predicted))
print(trainp.iloc[:10])

trainn.iloc[:10]
gg=X_test.reset_index(drop=True)

# print(gg)

for i, p in enumerate(predicted):

#     print(i)

    print (gg[i] + " - " + str(p))

    if i>5:

        break #to avoid a lot of printing and slowing down the notebook
predicted_nolabel=model.predict(tf_test_nolabel)

for i, p in enumerate(tf_test_nolabel):

#     print(i)

    print (test_nolabel.tweet[i] + " - " + str(predicted_nolabel[i]))

    if i>5:

        break #to avoid a lot of printing and slowing down the notebook
test_custom=pd.DataFrame(["racist", "white judge trial", "it is a horrible incident", "@user #white #supremacists want everyone to see the new â  #birdsâ #movie â and hereâs why", " @user #white #supremacists want everyone to see the new â  #birdsâ #movie â and hereâs why", "@user  at work: attorneys for white officer who shot #philandocastile remove black judge from presiding over trial. htâ¦"])

tf_custom = vect.transform(test_custom[0])

model.predict(tf_custom)