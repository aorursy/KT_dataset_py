# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
imdb=pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
import re,nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sb
mve=[]
for i in range(len(imdb.review)):
    sent=re.sub('[^a-zA-z]',' ',imdb.review[i])
    sent=sent.lower()
    sent=sent.split()
    wr=WordNetLemmatizer()
    sent=[wr.lemmatize(i) for i in sent if i not in set(stopwords.words('english'))]
    sent=' '.join(sent)
    mve.append(sent)
mve
y=imdb.sentiment.values
sb.countplot(imdb.sentiment)
cdr=LabelEncoder()
ycoder=cdr.fit_transform(y)
tf=TfidfVectorizer()
vect=tf.fit_transform(mve)

xtrain,xtest,ytrain,ytest=train_test_split(vect,ycoder,test_size=0.2,random_state=0)
reg=LogisticRegression()
reg.fit(xtrain,ytrain)
yprdct=reg.predict(xtest)
print(classification_report(ytest,yprdct))
vect.shape
se=Sequential()
se.add(Dense(units=10,activation='relu'))
se.add(Dense(units=5,activation='relu'))
se.add(Dense(units=1,activation='sigmoid'))
se.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
se.fit(xtrain.toarray(),ytrain,epochs=20,batch_size=256)
ydeep=se.predict(xtest.toarray())
ydeep=(ydeep>0.5)
print(classification_report(ytest,ydeep))





vect.shape




y

y




mve

mve




