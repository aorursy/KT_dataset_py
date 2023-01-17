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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import plotly.graph_objects as go
review = pd.read_csv('../input/yelp-reviews-dataset/yelp.csv')

review.head()
review['length']=review['text'].apply(len)

review.head()
trace=go.Box(x=review['stars'],y=review['length'])

layout=go.Layout(title="Length according to number of stars",template="plotly_dark")

fig=go.Figure([trace],layout)

fig.show()
(sns.FacetGrid(review,col="stars")).map(plt.hist,'length',bins=50)

sns.countplot(x='stars',data=review,palette="rainbow")
star=review.groupby('stars').mean()

star
sns.heatmap(star.corr(),annot=True,cmap='coolwarm')
review_class=review[(review['stars']==1) | (review['stars']==5)]

X=review_class['text']

y=review_class['stars']

def text_process(text):

    nopunc = [char for char in text if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
from sklearn.feature_extraction.text import CountVectorizer



cv=CountVectorizer()
X=cv.fit_transform(X)
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=101)
from sklearn.naive_bayes import MultinomialNB



nb=MultinomialNB()

nb.fit(X_train,y_train)

predict=nb.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predict))

print(classification_report(y_test,predict))
from sklearn.pipeline import Pipeline
pipe_multi=Pipeline([('b',CountVectorizer()),('model',MultinomialNB())])
X=review_class['text']

y=review_class['stars']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=101)
pipe_multi.fit(X_train,y_train)

predict_multi=pipe_multi.predict(X_test)

print(confusion_matrix(y_test,predict_multi))

print(classification_report(y_test,predict_multi))
from sklearn.ensemble import GradientBoostingClassifier

pipe_Grad=Pipeline([('c',CountVectorizer()),('gb',GradientBoostingClassifier(max_depth=5,max_features=0.5,random_state=999999))])

pipe_Grad.fit(X_train,y_train)

predict_grad=pipe_Grad.predict(X_test)

print(confusion_matrix(y_test,predict_grad))

print(classification_report(y_test,predict_grad))
