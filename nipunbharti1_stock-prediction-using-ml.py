#Importing Libraries

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
#Loading Data

data= pd.read_csv("../input/Combined_News_DJIA.csv")

data.head(4)
#Divide data into train and test

train = data[data['Date'] < '2015-01-01']

test = data[data['Date'] > '2014-12-31']
#Combining all 25 headlines of training dataset into a single string

trainheadlines = []

for row in range(0,len(train.index)):

    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))
#Setting up CountVectorizer

cv = CountVectorizer( min_df=0.1, max_df=0.7, max_features = 200000, ngram_range = (1, 1))

cv_train = cv.fit_transform(trainheadlines)
#Setting up LogisticRegression model

logt = LogisticRegression()

model = logt.fit(cv_train, train["Label"])

model
#Combining all 25 headlines of testing dataset into a single string

testheadlines = []

for row in range(0,len(test.index)):

    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
#Making Predictions

cv_test = cv.transform(testheadlines)

y_pred=logt.predict(cv_test)
#Checking accuracy_score

from sklearn.metrics import accuracy_score

accuracy_score(test['Label'],y_pred)
#Advanced modeling using 2-gram weights

ad_cv = CountVectorizer( min_df=0.03, max_df=0.97, max_features = 200000, ngram_range = (2, 2))

cv_train2g = ad_cv.fit_transform(trainheadlines)
log1 = LogisticRegression()

model1 = log1.fit(cv_train2g, train["Label"])

model1
cv_test2g = ad_cv.transform(testheadlines)

Y_pred1 = log1.predict(cv_test2g)
from sklearn.metrics import accuracy_score

accuracy_score(test['Label'],Y_pred1)
#Confusion_matrix

print(pd.crosstab(test["Label"], Y_pred1, rownames=["Actual"], colnames=["Predicted"]))
# Importing RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
cv = CountVectorizer( min_df=0.1, max_df=0.7, max_features = 200000, ngram_range = (1, 1))

cv_train = cv.fit_transform(trainheadlines)
rfc=RandomForestClassifier()

forest = rfc.fit(cv_train,train["Label"])

forest
cv_test = cv.transform(testheadlines)

y_predrfc=rfc.predict(cv_test)
from sklearn.metrics import accuracy_score

accuracy_score(test['Label'],y_predrfc)
#Advanced modeling using 2-gram weights

ad_cv = CountVectorizer( min_df=0.03, max_df=0.97, max_features = 200000, ngram_range = (2, 2))

cv_train2g = ad_cv.fit_transform(trainheadlines)
forest_2g = rfc.fit(cv_train2g, train["Label"])

forest_2g
cv_test2g = ad_cv.transform(testheadlines)
Y_predrfc1 = rfc.predict(cv_test2g)
from sklearn.metrics import accuracy_score

accuracy_score(test['Label'],Y_predrfc1)
# naive_bayes 

from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB(alpha=0.01)

mnbmodel= mnb.fit(cv_train, train["Label"])

mnbmodel
y_predmnb2=mnb.predict(cv_test)
from sklearn.metrics import accuracy_score

accuracy_score(test['Label'],y_predmnb2)
mnb2gram = mnb.fit(cv_train2g, train["Label"])

mnb2gram
Y_predmnb2 = mnb.predict(cv_test2g)
from sklearn.metrics import accuracy_score
accuracy_score(test['Label'],Y_predmnb2)