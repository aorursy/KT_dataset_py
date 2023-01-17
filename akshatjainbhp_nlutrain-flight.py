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
import numpy as np
import pandas as pd
df = pd.read_csv("../input/training_nlu_smaller.csv")
df
df.shape
sns.heatmap(df.isnull(),cmap='YlGnBu')
df["previous_intent"].value_counts()
df["current_intent"].value_counts()
df=df.apply(lambda x: x.astype(str).str.lower())
df.head(50)
df.drop(['template'],axis=1,inplace=True)
df

df=df[df.previous_intent =='no']
df.head(50)
df.drop(['previous_intent'],axis=1,inplace=True)
df.reset_index(drop=True, inplace=True)
df.head(50)

df.shape
df["current_intent"].value_counts()
book = df[df['current_intent'] == 'book'].shape[0]
cancel = df[df['current_intent'] == 'cancel'].shape[0]
neg = df[df['current_intent'] == 'negation'].shape[0]
check = df[df['current_intent'] == 'check-in'].shape[0]
status = df[df['current_intent'] == 'status'].shape[0]

plt.bar(10,book,3, label="book")
plt.bar(15,neg,3, label="negation")
plt.bar(20,check,3, label="check-in")
plt.bar(25,status,3, label="status")
plt.bar(30,cancel,3, label="cancel")

plt.legend()
plt.ylabel('Number of examples')
plt.title('count of current intent')
plt.show
df['query'] = df['query'].str.replace(r'\W', ' ')
df.head(50)
import nltk
from collections import Counter

words = nltk.word_tokenize(" ".join(df['query'].values.tolist()))
counter = Counter(words)
print(counter.most_common(100))

# def query():
#     for i in df['query']:
#         print(i)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

df['current_intentt'] = labelencoder.fit_transform(df['current_intent'])
df.head(50)
df.drop(['current_intent'],axis=1,inplace=True)
df

from sklearn.model_selection import train_test_split

X=df['query']
y=df['current_intentt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)



from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(stop_words='english')

cv_train= cv.fit_transform(X_train)
cv_train

cv.get_feature_names()

d=cv_train.toarray()
d
d.shape
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(cv_train,y_train.astype("int"))
cv_train.shape
pred = model.predict(cv.transform(X_test))
pred
model.score(cv.transform(X_test),y_test.astype("int"))

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

results = confusion_matrix(y_test, pred) 
  
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',accuracy_score(y_test, pred) )
print('Report : ')
print(classification_report(y_test, pred) )
