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
import pandas as pd
df=pd.read_csv('/kaggle/input/fake-news/train.csv')
df.head()
x=df.drop('label',axis=1)
x.head()
y=df['label']
y.head()
df.shape
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
df=df.dropna()
messages=df.copy() #making a copy of dataset
messages.reset_index(inplace=True)  #to reset indices to 0,1,2 and df.set_index is used to set our own index,inplace=true means make it work as method, don't create new object. we do this because after dropping some rows,many indices will also get drop
messages['title'][6]
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['title'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
    
print(corpus)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,ngram_range=(1,3))
x=cv.fit_transform(corpus).toarray()
y=messages['label']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
cv.get_feature_names()[:20]
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
from sklearn import metrics
import numpy as np
classifier.fit(x_train,y_train)
pred=classifier.predict(x_test)
score=metrics.accuracy_score(y_test,pred)
print("Accuracy is ",score)
print("Confusion matrix is ",metrics.confusion_matrix(y_test,pred))

