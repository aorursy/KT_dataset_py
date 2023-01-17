import pandas as pd

import numpy as np

import warnings





data = pd.read_csv('../input/womenclothingreview/Womens Clothing E-Commerce Reviews.csv', index_col =0)

data.head()
data.info()
round(100 * data.isna().sum()/len(data),2)
# picking columns which are required for this analysis. 

# ss = subset

data_ss = data[["Review Text", "Class Name"]]
data_ss.rename(columns={"Review Text" : "review_text", "Class Name": "class"}, inplace=True)

data_ss.head()
# Removing nulls 

data_clean = data_ss

data_clean = data_clean[~data_clean.review_text.isna()]

data_clean = data_clean[~data_clean["class"].isna()]

round(100 * data_clean.isna().sum()/len(data_clean),2)
data_clean["class"].value_counts()



# slecting only top 3 classes 

data_clean = data_clean[data_clean["class"].isin(["Dresses", "Knits", "Blouses"])]



data_clean['class'] = data_clean['class'].map({'Dresses':0, 'Knits':1, 'Blouses':2})

data_clean.head()

data_clean.info()
X = data_clean["review_text"]

y = data_clean["class"]

print(f'x: {len(X)}');

print(f'y: {len(y)}');
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)

print(y_train.shape)
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(stop_words='english')

vec.fit(X_train)
vec.vocabulary_
len(vec.vocabulary_)
X_train_transformed = vec.transform(X_train)

X_test_transformed = vec.transform(X_test)
from sklearn.naive_bayes import BernoulliNB

ber = BernoulliNB()

ber.fit(X_train_transformed, y_train)
y_test_pred = ber.predict(X_test_transformed)
from sklearn import metrics



confusion = metrics.confusion_matrix(y_test, y_test_pred)

metrics.accuracy_score(y_test, y_test_pred)
confusion