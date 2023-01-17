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
data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
data.info()
test.info()
data.head(100)
import matplotlib.pyplot as plt

import seaborn as sns
sns.barplot(x="target", y="id", data=data)
data.isnull().sum()
disaster_tweet_length = data[data['target']==1]['text'].str.len()

nondisaster_tweet_length = data[data['target']==0]['text'].str.len()



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))



ax1.hist(disaster_tweet_length, color='red')

ax1.set_title("Disaster Tweets")



ax2.hist(nondisaster_tweet_length, color='green')

ax2.set_title("Non-Disaster Tweets")



fig.suptitle("Characters in tweets")

plt.show()
import nltk as nlp

import re

description_list = []

for description in data["text"]:

    description = re.sub("[^a-zA-Z]",' ',str(description))

    description = description.lower()   # buyuk harftan kucuk harfe cevirme

    description = nlp.word_tokenize(description)

    #description = [ word for word in description if not word in set(stopwords.words("english"))]

    lemma = nlp.WordNetLemmatizer()

    description = [ lemma.lemmatize(word) for word in description]

    description = " ".join(description)

    description_list.append(description) #we hide all word one section
from sklearn.feature_extraction.text import CountVectorizer 

max_features = 3000 #We use the most common word

count_vectorizer = CountVectorizer(max_features = max_features, stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()

print("the most using {} words: {}".format(max_features,count_vectorizer.get_feature_names()))
y = data['target']   # male or female classes

x = sparce_matrix

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.10, random_state = 42)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("the accuracy of our model: {}".format(nb.score(x_test,y_test)))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

lr = LogisticRegression(max_iter = 200)

lr.fit(x_train,y_train)

predict_lr = lr.predict(x_test)

print(accuracy_score(predict_lr, y_test)*100)

print("our accuracy is: {}".format(lr.score(x_test,y_test)))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors =6)

predict_knn=knn.fit(x_train,y_train)

#print('Prediction: {}'.format(prediction))

print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test))
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))

#y_train = np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))



# Initialising the RNN

regressor = Sequential()



# Adding the first LSTM layer and some Dropout regularisation



regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))

regressor.add(Dropout(0.2))



# Adding a second LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(Dropout(0.2))



# Adding a third LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(Dropout(0.2))



# Adding a fourth LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50))

regressor.add(Dropout(0.2))



# Adding the output layer

regressor.add(Dense(units = 1))



# Compiling the RNN

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=["accuracy"])



# Fitting the RNN to the Training set

regressor.fit(x_test, y_test, epochs = 3, batch_size = 32)
