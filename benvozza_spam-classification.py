# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import string

from nltk.stem import SnowballStemmer

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

data = pd.read_csv("../input/spam.csv",encoding='latin-1')



data.head()

data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

data = data.rename(columns={"v1":"class", "v2":"text"})

data.head()
data['length'] = data['text'].apply(len)

data.head()
def pre_process(text):

    

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

    words = ""

    for i in text:

            stemmer = SnowballStemmer("english")

            words += (stemmer.stem(i))+" "

    return words
textFeatures = data['text'].copy()

textFeatures = textFeatures.apply(pre_process)

vectorizer = TfidfVectorizer("english")

features = vectorizer.fit_transform(textFeatures)



features_train, features_test, labels_train, labels_test = train_test_split(features, data['class'], test_size=0.3, random_state=111)



from sklearn.metrics import accuracy_score

from sklearn.svm import SVC



svc = SVC(kernel='sigmoid', gamma=1.0)

svc.fit(features_train, labels_train)

prediction = svc.predict(features_test)

accuracy_score(labels_test,prediction)



from sklearn.naive_bayes import MultinomialNB



mnb = MultinomialNB(alpha=0.2)

mnb.fit(features_train, labels_train)

prediction = mnb.predict(features_test)

accuracy_score(labels_test,prediction)
