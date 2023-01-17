import numpy as np

import pandas as pd

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectPercentile
# encoding='latin-1' is used to download all special characters and everything in python

data = pd.read_csv('../input/spam.csv',encoding='latin-1')
data.head()
data.columns
data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1,inplace=True)
data1 = data['v1']

data.shape
data.head()
X_train, X_test,y_train, y_test = train_test_split(data.v2,data.v1,test_size=0.3)
X_train.shape
y_train.shape
y_test.shape
X_train.head()
vectorizer = TfidfVectorizer()
#don't fit model on test data

X_train_transformed = vectorizer.fit_transform(X_train)

X_test_transformed = vectorizer.transform(X_test)
X_train_transformed
features_names = vectorizer.get_feature_names()
len(features_names)
selector = SelectPercentile(percentile=5)

selector.fit(X_train_transformed, y_train)

X_train_transformed = selector.transform(X_train_transformed).toarray()

X_test_transformed = selector.transform(X_test_transformed).toarray()
X_train_transformed 
## Applying Naive Bayes
m1 = GaussianNB()



m1.fit(X_train_transformed,y_train)

y_predict = m1.predict(X_test_transformed)

y_predict # Predicted Value

y_test # Actual Value

accuracy_score(y_test,y_predict)

#y_predict

#accuracy_score(y_test,y_predict)

np.mean(y_test == y_predict)

confusion_matrix(y_test,y_predict)
#from sklearn library used to learn accuracy

accuracy_score(y_test,y_predict)
y_test.shape
np.mean(y_test == y_predict)
confusion_matrix(y_test,y_predict)
(1435+187)/(1435+187+30+20)
model_bernb = BernoulliNB()



model_bernb.fit(X_train_transformed,y_train)

y_predict = model_bernb.predict(X_test_transformed)





accuracy_score(y_test,y_predict)



newEmail = pd.Series('hello how are you')
newEmail
newEmail_transformed = vectorizer.transform(newEmail)

newEmail_transformed = selector.transform(newEmail_transformed).toarray()
m1.predict(newEmail_transformed)
newEmail2 = pd.Series('WINNER!! As a valued network customer you have been selected to receivea å£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.')
newEmail2
newEmail_transformed2 = vectorizer.transform(newEmail2)

newEmail_transformed2 = selector.transform(newEmail_transformed2).toarray()
m1.predict(newEmail_transformed2)