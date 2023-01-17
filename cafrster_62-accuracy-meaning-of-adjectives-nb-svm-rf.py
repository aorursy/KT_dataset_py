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
import pandas as pd



#I have decided to use a chunksize of 100000 here, but you could reduce it easily to 10k



chunksize = 100000

for true in pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv", chunksize=chunksize):

    print(true.shape)

 

true['category'] = 1



for fake in pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv", chunksize=chunksize):

    print(fake.shape)



fake['category'] = 0



df = pd.concat([true,fake])

df.head()
texts = df["title"]
import nltk



texts_transformed = []

for review in texts: 

    sentences = nltk.sent_tokenize(review)

    adjectives = []

    

    for sentence in sentences:

        words = nltk.word_tokenize(sentence)

        words_tagged = nltk.pos_tag(words)

        

        for word_tagged in words_tagged:

            if word_tagged[1] == "JJ":

                adjectives.append(word_tagged[0])

                

    texts_transformed.append(" ".join(adjectives))
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
X = texts_transformed

y = df["category"] == 0
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 10)



cv = CountVectorizer(max_features = 50)

cv.fit(X_train)



X_train = cv.transform(X_train)

X_test = cv.transform(X_test)
#Here the Multinomial Naive Bayes is applied. 

model = MultinomialNB()

model.fit(X_train, y_train)



print(model.score(X_test, y_test))
#As an alternative here SVM is applied. 



from sklearn.svm import SVC



model = SVC(kernel = "linear")

model.fit(X_train, y_train)



print(model.score(X_test, y_test))
# let us look at the list of adjectives and their coefficients; the lower the coeffiencient the more likely it is fake news



adj = list(zip(model.coef_[0], cv.get_feature_names()))

adj = sorted(adj)



for i in adj:

    print(i)
#As an alternative here RandomForest is applied.



from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(criterion = "entropy", n_estimators = 30)

model.fit(X_train, y_train)



print(model.score(X_test, y_test))