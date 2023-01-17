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
data= pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='ISO-8859-1')
data.head()
data.columns
data.drop(columns=['Unnamed: 2', 'Unnamed: 3','Unnamed: 4'],inplace=True)
data.head()
data.rename({'v1': 'labels', 'v2': 'messages'}, axis=1, inplace=True)
data.head()
data.describe()
data.info()
data.groupby('labels').describe().T
data.isnull().sum()
len(data)
data['length']=data['messages'].apply(len)
data.head()
data['labels'].unique()
data['labels'].value_counts()
import matplotlib.pyplot as plt 

import seaborn as sns

#plt.style.use('fivethirtyeight')
data['length'].plot(bins=50,kind='hist')

plt.ioff()
plt.xscale('log')

bins=1.15**(np.arange(0,50))

plt.hist(data[data['labels']=='ham']['length'],bins=bins,alpha=0.8)

plt.hist(data[data['labels']=='spam']['length'],bins=bins,alpha=0.8)

plt.legend('ham','spam')

plt.show()
data.hist(column='length',by='labels',bins=50,figsize=(10,4))

plt.ioff()
data['length'].describe()
data[data['length']==910]['messages'].iloc[0]
from sklearn.model_selection import train_test_split
X=data['length'].values[:,None]

#X=data['length'].values

y=data['labels']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
X_train.shape
#y_test
from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression(solver='lbfgs')
lr_model.fit(X_train,y_train)
from sklearn import metrics
predictions=lr_model.predict(X_test)
predictions
#y_test
print(metrics.confusion_matrix(y_test,predictions))
df=pd.DataFrame(metrics.confusion_matrix(y_test,predictions),index=['ham','spam'],columns=['ham','spam'])

df
print(metrics.classification_report(y_test,predictions))
print(metrics.accuracy_score(y_test,predictions))
from sklearn.naive_bayes import MultinomialNB

nb_model=MultinomialNB()

nb_model.fit(X_train,y_train)

predictions=nb_model.predict(X_test)

print(metrics.confusion_matrix(y_test,predictions))
print(metrics.classification_report(y_test,predictions))
from sklearn.svm import SVC

svc_model=SVC(gamma='auto')

svc_model.fit(X_train,y_train)

predictions=svc_model.predict(X_test)

print(metrics.confusion_matrix(y_test,predictions))
print(metrics.classification_report(y_test,predictions))
data.head()
data.isnull().sum()
data['labels'].value_counts()
from sklearn.model_selection import train_test_split
X=data['messages']
y=data['labels']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)
from sklearn.feature_extraction.text import CountVectorizer 
count_vect=CountVectorizer()
# FIT Vectorizer to the data (build a vocab,count the number of words)

#count_vect.fit(X_train)

# Transform the original text to message --> Vector 

#X_train_counts=count_vect.transform(X_train)



X_train_counts=count_vect.fit_transform(X_train) # One step Fit and Transform
X_train_counts
X_train.shape
X_train_counts.shape
from sklearn.feature_extraction.text import TfidfTransformer 
tfidf_transformer=TfidfTransformer()
X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
from sklearn.feature_extraction.text import TfidfVectorizer 
vectorizer=TfidfVectorizer()
X_train_tfidf=vectorizer.fit_transform(X_train)
from sklearn.svm import LinearSVC
clf=LinearSVC()
clf.fit(X_train_tfidf,y_train)
from sklearn.pipeline import Pipeline
text_clf=Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])
text_clf.fit(X_train,y_train)
predictions=text_clf.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
from sklearn import metrics 
metrics.accuracy_score(y_test,predictions)
text_clf.predict(["Hi how are you doing today"])
text_clf.predict(["COngraluations you are lucky winner of bummer prize money"])