import numpy as np

import pandas as pd
yelp = pd.read_csv('yelp.csv')
yelp.head()
yelp.info()
yelp['text_length'] = yelp['text'].apply(len)

yelp.head()
yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]

yelp_class.head(10)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
g = sns.FacetGrid(yelp,col='stars')

g.map(plt.hist,'text_length')
sns.countplot(x='stars',data=yelp,palette='rainbow')
X = yelp_class['text']

y = yelp_class['stars']
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X = cv.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train,y_train)
predictions = nb.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))