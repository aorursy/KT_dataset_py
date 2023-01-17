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
df = pd.read_json("/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json", lines =1)

for i in range(0, 26709):

    df.iloc[i,1] = df.iloc[i,1].replace ("'",'')

    

X = df.iloc[:,1]

y= df.iloc[:,2]



    
from nltk.corpus import stopwords

import re

from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import confusion_matrix
corpus = []

ps = PorterStemmer()

for i in range (0,26709):

    sent = re.sub("[^A-Za-z]", " ", X[i])

    sent = sent.lower().split()

    sent = [ps.stem(word) for word in sent if word not in set(stopwords.words("english"))]

    sent = " ".join(sent)

    corpus.append(sent)

    

    
cv = CountVectorizer(max_features = 1000)

X_processed = cv.fit_transform(X).toarray()
X_train,X_test,y_train,y_test =train_test_split(X_processed, y, test_size = 0.2, random_state  = 10, shuffle = True )
from sklearn.svm import LinearSVC

simple_classifier = LinearSVC()

simple_classifier.fit(X_train, y_train)
y_linearsvc_pred = simple_classifier.predict(X_test)

print(confusion_matrix (y_linearsvc_pred,y_test))

print(accuracy_score (y_linearsvc_pred,y_test))
import keras

from keras.models import Sequential

from keras.layers import Dense
ann_classifier = Sequential()

ann_classifier.add(Dense(output_dim = 128, init ='uniform', activation = 'relu', input_dim = 1000 ))

ann_classifier.add(Dense(output_dim = 2, init ='uniform', activation = 'relu'))

ann_classifier.add(Dense(output_dim =1, init ='uniform', activation ='sigmoid'))



ann_classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'])

ann_classifier.fit(X_train,y_train, batch_size = 100, nb_epoch= 10)

ann_y_pred = ann_classifier.predict(X_test)

ann_y_pred = (ann_y_pred > 0.5)



print(confusion_matrix(ann_y_pred,y_test))

print(accuracy_score(ann_y_pred,y_test))