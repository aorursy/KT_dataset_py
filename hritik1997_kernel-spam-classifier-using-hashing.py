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
dataset2=pd.read_csv("../input/spam-filter/emails.csv") 
dataset2.shape
dataset2.head()
dataset2.text.iloc[0]
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,5728):
  review1=re.sub('[^a-zA-Z]',' ',dataset2.text.iloc[i])
  review1=review1.lower()
  review1=review1.split()
  ps=PorterStemmer()
  review1=[ps.stem(word) for word in review1 if not word in set(stopwords.words('english'))]
  review1=' '.join(review1)
  corpus.append(review1)
corpus[0]
x=[[] for i in range(5728)]
table_size=5500

def word_to_index( word):
    word=word.lower()
    l=len(word)
    index=0
    for j in range(l):
        index=index+(ord(word[j]))*(l-j)*(l-j)
    index=index-100*(l-1)*(l-1)
     
    return (index%table_size)
    
print(word_to_index("b"))
def hashed_dictionary(review):
    arr=[0]*table_size
    review=review.split()
    for i in range(len(review)):
        index=word_to_index(review[i])
        arr[index]=arr[index]+1
    return arr
print(hashed_dictionary("abc a abx b d"))
arr=hashed_dictionary("abc a abx b d  abd bca xba down gown frown slown")
count=0
for i in range(len(arr)):
    if arr[i]>0:
        count=count+1
print(count)
    

for i in range(5728):
    x[i]=hashed_dictionary(corpus[i])
    
x=np.array(x)
x.shape
y=dataset2.iloc[:,1].values
y.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
x_train.shape
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(solver='lbfgs')
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
(1306+390)/(1306+14+9+390)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

(1299+396)/(1306+14+9+390)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(metric='euclidean', n_neighbors=5)
knn.fit(x_train, y_train)
y_pred=knn.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train, y_train)
y_pred=svclassifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
