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
data = pd.read_csv(r"../input/ExtractedTweets.csv", encoding="latin1")

data = pd.concat([data.Party,data.Tweet],axis =1)

data.dropna(axis = 0, inplace = True)

data.Party = [1 if each == "Democrat" else 0 for each in data.Party]
data.info()
print(data)
import re
import nltk
from nltk.corpus import stopwords
import nltk as nlp
Tweet_list = []

for Tweet in data.Tweet:

    Tweet =re.sub("[^a-zA-Z]"," ",Tweet)

    Tweet = Tweet.lower()

    Tweet = nltk.word_tokenize(Tweet)

    Tweet = [ word for word in Tweet if not word in set(stopwords.words("english"))]

    lemma = nlp.WordNetLemmatizer()

    Tweet = [lemma.lemmatize(word) for word in Tweet]

    Tweet = " ".join(Tweet)

    Tweet_list.append(Tweet)
print(Tweet)
print(data.Tweet)
Tweet_list
from sklearn.feature_extraction.text import CountVectorizer

max_features = 5000
count_vectorizer = CountVectorizer(max_features=max_features , stop_words= "english")
sparce_matrix = count_vectorizer.fit_transform(Tweet_list).toarray()
print("Most used {} words: {}".format(max_features,count_vectorizer.get_feature_names()))
y = data.iloc[:,0].values

x = sparce_matrix
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state =42)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)
y_pred = nb.predict(x_test)

print("Accuracy of Tweets Democrats: ",nb.score(y_pred.reshape(-1,1),y_test))