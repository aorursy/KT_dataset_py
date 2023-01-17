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
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
dataset1 = pd.read_csv('../input/Sheet_1.csv',usecols=['response_id','class','response_text'],encoding='latin-1')

dataset2 = pd.read_csv("../input/Sheet_2.csv",encoding='latin-1')
dataset1.head(5)
dataset2.head(5)
dataset1['class'].value_counts()

from wordcloud import WordCloud, STOPWORDS

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

import matplotlib.pyplot as plt

%matplotlib inline

def cloud(text):

    wordcloud = WordCloud(background_color="white",stopwords=stop).generate(" ".join([i for i in text.str.upper()]))

    plt.figure(figsize=(12,8))

    plt.imshow(wordcloud)

    

    plt.axis("off")

    plt.title("Chat Bot Response")

cloud(dataset1['response_text'])
from sklearn import preprocessing

Encode = preprocessing.LabelEncoder()



from sklearn import metrics

#from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
dataset1['Label'] = Encode.fit_transform(dataset1['class'])

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

vector = CountVectorizer()

x = dataset1.response_text

y = dataset1.Label

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)

x_train_dtm = vector.fit_transform(x_train)

x_test_dtm = vector.transform(x_test)

rf = RandomForestClassifier(max_depth=10,max_features=10)

rf.fit(x_train_dtm,y_train)

rf_predict = rf.predict(x_test_dtm)

metrics.accuracy_score(y_test,rf_predict)
dataset2["class"].value_counts()
def wordcloud(dataframe):

    stopwords = set(STOPWORDS)

    wordcloud = WordCloud(background_color="white",stopwords=stopwords).generate(" ".join([i for i in dataframe.str.upper()]))

    plt.figure(figsize=(12,8))

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.title("bag_composition")



wordcloud(dataset2['resume_text'])
dataset2['Label'] = Encode.fit_transform(dataset2['class'])

x = dataset2.resume_text

y = dataset2.Label

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)

x_train_dtm = vector.fit_transform(x_train)

x_test_dtm = vector.transform(x_test)
rf = RandomForestClassifier(max_depth=10,max_features=10)

rf.fit(x_train_dtm,y_train)

rf_predict = rf.predict(x_test_dtm)

metrics.accuracy_score(y_test,rf_predict)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.utils import to_categorical

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/Sheet_1.csv')

x_col = dataset.iloc[1:, 2] 

y_col = dataset.iloc[1:, 1] 
num_classes = len(list(set(y_col)))

y_int_to_label = {idx:label for idx,label in enumerate(set(y_col))}

y_label_to_int = {label:idx for idx,label in enumerate(set(y_col))}

y_int = np.array([y_label_to_int[label] for label in y_col])

print(y_int)
vectorizer = TfidfVectorizer()

x_vect = vectorizer.fit_transform(x_col)

y_vect = to_categorical(y_int)
x_train, x_test, y_train, y_test = train_test_split(x_vect, y_vect, test_size=0.3)
model = Sequential()

model.add(Dense(256, activation='tanh', input_dim=x_vect.shape[1]))

model.add(Dropout(0.4))

model.add(Dense(128, activation='tanh'))

model.add(Dropout(0.2))

model.add(Dense(64, activation='tanh'))

model.add(Dropout(0.1))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
num_epochs=50

hist = model.fit(x_train, y_train, batch_size=10, epochs=num_epochs, verbose=2, validation_split=0.25)
train_loss = hist.history['loss']

val_loss   = hist.history['val_loss']

train_acc  = hist.history['acc']

val_acc    = hist.history['val_acc']

xc         = range(num_epochs)

plt.figure()

plt.plot(xc, train_loss, color='red')

plt.plot(xc, val_loss, color='green')

plt.show()
score = model.evaluate(x_test, y_test)

print('Accuracy ', score[1])
dataset = pd.read_csv('../input/Sheet_2.csv', engine='python')

x_col = dataset.iloc[1:, 2] 

y_col = dataset.iloc[1:, 1] 
num_classes = len(list(set(y_col)))

y_int_to_label = {idx:label for idx,label in enumerate(set(y_col))}

y_label_to_int = {label:idx for idx,label in enumerate(set(y_col))}

y_int = np.array([y_label_to_int[label] for label in y_col])

print(y_int)
vectorizer = TfidfVectorizer()

x_vect = vectorizer.fit_transform(x_col)

y_vect = to_categorical(y_int)
x_train, x_test, y_train, y_test = train_test_split(x_vect, y_vect, test_size=0.3)
model = Sequential()

model.add(Dense(256, activation='tanh', input_dim=x_vect.shape[1]))

model.add(Dropout(0.4))

model.add(Dense(128, activation='tanh'))

model.add(Dropout(0.2))

model.add(Dense(64, activation='tanh'))

model.add(Dropout(0.1))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
num_epochs=50

hist = model.fit(x_train, y_train, batch_size=10, epochs=num_epochs, verbose=2, validation_split=0.25)
train_loss = hist.history['loss']

val_loss   = hist.history['val_loss']

train_acc  = hist.history['acc']

val_acc    = hist.history['val_acc']

xc         = range(num_epochs)

plt.figure()

plt.plot(xc, train_loss, color='red')

plt.plot(xc, val_loss, color='green')

plt.show()
score = model.evaluate(x_test, y_test)

print('Accuracy ', score[1])