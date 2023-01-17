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
import seaborn as sns

import nltk

from nltk.corpus import stopwords

from sklearn import naive_bayes

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
df=pd.read_csv('../input/spam.csv', encoding='latin-1')
df.shape
# It shows you the top 5 rows of data

df.head()
sns.countplot(x='v1',data=df)
df.drop(df.iloc[:, 2:5], inplace=True, axis=1)

df.head()
#It will get the stopwords used.

stopset = set(stopwords.words('english'))

vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)
vectorizer.fit(df)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

a=le.fit_transform(df['v1'])

print(list(le.classes_))

print(a)

print(list(le.inverse_transform([0, 1])))
type= pd.DataFrame(data=a, columns=["type"])

df1 = pd.concat([df,type],axis=1)

print(df1.head())

df1.drop('v1',axis=1,inplace=True)

df1.head()
#Dependent variable

Y=df1.type
# Convert a collection of raw documents to a matrix of TF-IDF features.

# Converting text present in all sms into features.

#5572--> total number of SMSs 8578--> total no of words across all SMSs excluding the stopwords

X = vectorizer.fit_transform(df.v2)

X.shape
#Spliting the SMS to separate the text into individual words

SMS_1= df1['v2'][0].split()

print(SMS_1)
#Number of words in SMS_1 are 20 but only 14 tf-idf values are generated as remaining were stopwords which are excluded

print(len(SMS_1))

print(X[0])

## Most frequent word appearing in the SMS

max(SMS_1)
#TF IDF values for all words 

print(X)
#To get the word in a particular position

vectorizer.get_feature_names()[7826]
#Split and train the model 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)
#Train Naive Bayes Classifier

NB = naive_bayes.MultinomialNB()

model=NB.fit(X_train, Y_train)

Y_predict=model.predict(X_test)
print(X_test, Y_predict)
print(Y_test)
Y_predict1=pd.DataFrame(Y_predict)

print(Y_predict1)
print(Y_test.shape,Y_predict.shape)
prob=model.predict_proba(X_test)

prob
model.predict_proba(X_test)[:,1]
#Model's accuracy

roc_auc_score(Y_test, model.predict_proba(X_test)[:,1])