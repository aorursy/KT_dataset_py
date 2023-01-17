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
# Reading the data from file 

train_data = pd.read_csv('../input/training_data.tsv',header=0,delimiter="\t" ,quoting=3)
# Function to define if there is reminder 

def found(x):

    if (x[0] == "Not Found"):

        return 0

    else:

        return 1

#Applying the found function

train_data['label_found'] = train_data[['label']].apply( found , axis = 1) 
import re

from nltk.corpus import stopwords

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import accuracy_score
def clean_txt(texts):

    letters = re.sub("[^a-zA-Z]",' ', str(texts))

    lower_case = letters.lower()

    words = lower_case.split()

    stopword = stopwords.words('english')

    meaning_words = [w for w in words if not w in stopword]

    return (" ".join(meaning_words))

#clean_txt can be further extended to clean the data By Stemming but it can create problem as there are English and hindi and emojis in txt

# Its better to remove the Digits and Emojis from the data
train_data['sent_clean'] = [clean_txt(review) for review in train_data["sent"].values]
test_data = pd.read_csv('../input/eval_data1.txt',header=0,delimiter="\t" ,quoting=3)

#Loading the test data as test_data
test_data['sent_clean'] = [clean_txt(review) for review in test_data["sent"].values]

test_data.sample(3)
from sklearn.feature_extraction.text import CountVectorizer

#to vectorizing the reminder
print ("Building Bag-of-Words...")

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, max_features = 12000) 

bow_train = (vectorizer.fit_transform(train_data['sent_clean'])).toarray()

bow_test = (vectorizer.transform(test_data['sent_clean'])).toarray()

print("Done...")
train_data.sample(4)
#Splitting the data in test and train with ratio of 20% and 80%

t_train , t_test , s_train , s_test = train_test_split(bow_train ,train_data['label_found'] , test_size = 0.20 , random_state=101)
#Applying the Logistic regrssion

logreg = LogisticRegression()

logreg = logreg.fit(t_train, s_train)
print(accuracy_score(logreg.predict(t_test),s_test))
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from xgboost.sklearn import XGBClassifier
rfc = RandomForestClassifier()

rfc.fit(t_train , s_train)

print(accuracy_score(rfc.predict(t_test),s_test))

#Accuracy of 70%
dtc = DecisionTreeClassifier()

dtc.fit(t_train , s_train)

print(accuracy_score(s_test, dtc.predict(t_test)))

#67% so the Logistic regression give best score
logreg = LogisticRegression()

logreg = logreg.fit(bow_train ,train_data['label_found'])

pred = logreg.predict(bow_test)
output = pd.DataFrame( data={ "label_found":pred} )

output.to_csv( "result_logistic_regression.csv", index=False, quoting=3 )

output.sample(10)



#Results accuracy can be further improved by using Google Word2Vec and Deep Learning but it will consume lot of time to predict