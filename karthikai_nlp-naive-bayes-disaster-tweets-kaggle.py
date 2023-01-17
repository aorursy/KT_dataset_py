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
train=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train.head()
test.head()
train.isnull().sum()
import seaborn as sns
sns.heatmap(train.isnull(),yticklabels=False)
train.drop(["keyword","location"],axis=1,inplace=True)
test.drop(["keyword","location"],axis=1,inplace=True)
target=train.target
sns.countplot(target)
train.drop(["target"],inplace=True,axis=1)
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
lemmatize=PorterStemmer()

corpus=[]
for i in range(len(train.text)):
    word=re.sub("[^a-zA-Z]"," ",train.text[i])
    word=word.lower()
    word=word.split()
    word=[lemmatize.stem(words) for words in word if words not in set(stopwords.words("english"))]
    word=" ".join(word)
    corpus.append(word)
corpus[1]
from sklearn.feature_extraction.text import CountVectorizer
count_vector=CountVectorizer(max_features=5000,ngram_range=(1, 2))
train_data=count_vector.fit_transform(corpus)
count_vector.get_feature_names()[:5]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train_data,target,test_size=0.3)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.ensemble import RandomForestClassifier
cross_val_score(LogisticRegression(),train_data,target)
cross_val_score(SVC(),train_data,target)
cross_val_score(MultinomialNB(alpha=1),train_data,target)
cross_val_score(RandomForestClassifier(n_estimators=100),train_data,target)
cross_val_score(BernoulliNB(alpha=1),train_data,target)
dist=MultinomialNB()
model=dist.fit(X_train,y_train)
y_pred=model.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
c=confusion_matrix(y_test,y_pred)
sns.heatmap(c,annot=True)
accuracy_score(y_test,y_pred)*100
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
test_data=[]
for i in range(len(test.text)):
    word=re.sub("[^a-zA-Z]"," ",test.text[i])
    word=word.lower()
    word=word.split()
    word=[lemmatize.stem(words) for words in word if words not in set(stopwords.words("english"))]
    word=" ".join(word)
    test_data.append(word)
test_for_pred=count_vector.transform(test_data)
test_for_pred.data
prediction=model.predict(test_for_pred)
submission=pd.DataFrame()
submission['id']=test.id
submission['target']=prediction
submission.target.value_counts()
final_submission=submission.to_csv("Result",index=False)
