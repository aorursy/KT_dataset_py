import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

import seaborn as sns
df=pd.read_csv('../input/textdb3/fake_or_real_news.csv')

df.head()
df.shape
df['label'].value_counts(normalize=True)
sns.barplot(x=df['label'].value_counts().index,y=df['label'].value_counts().values,data=df)

plt.show()
x=df['text']

y=df['label']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)

train=vectorizer.fit_transform(x_train)

test=vectorizer.transform(x_test)
## Logistic Regression

lr=LogisticRegression()

lr.fit(train,y_train)

y_pred=lr.predict(test)
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
## Naive -Bayes Classifier (Used widely for text classification)

nb=BernoulliNB()

nb.fit(train,y_train)

y_pred=nb.predict(test)
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
### Passive Aggressive Classifier 

pac= PassiveAggressiveClassifier()

pac.fit(train,y_train)

y_pred= pac.predict(test)
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)