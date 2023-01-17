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
train_path = '/kaggle/input/spam-sms-classification/TrainDataset.csv'

test_path = '/kaggle/input/spam-sms-classification/TestDataset.csv'
train = pd.read_csv(train_path)

train.head()
import re

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
porter = PorterStemmer()

lamma = WordNetLemmatizer()

tf = TfidfVectorizer()

countVector = CountVectorizer()
train.isnull().sum()
train.loc[0,'v2']
sentences = [train.loc[i,'v2'] for i in range(len(train['v2']))]
len(sentences)
len(train)
corpus = []



for i in range(len(sentences)):

    review = re.sub('[^a-zA-Z]',' ',sentences[i])

    review = review.lower()

    review = review.split()

    

    review = [porter.stem(word) for word in review if word not in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus.append(review)

    
corpus
tf = TfidfVectorizer(max_features=4000)

X = tf.fit_transform(corpus).toarray()
X.shape
X
train['v1'].head()
train['v1'].value_counts()
train['v1'] = train['v1'].map({'ham':0,'spam':1})
train['v1'].value_counts()
y = train['v1']
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,roc_auc_score

from xgboost import XGBClassifier

from catboost import CatBoostClassifier
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
classifiers = [['DecisionTree :',DecisionTreeClassifier()],

               ['RandomForest :',RandomForestClassifier()], 

               ['Naive Bayes :', GaussianNB()],

               ['KNeighbours :', KNeighborsClassifier()],

               ['SVM :', SVC()],

               ['Neural Network :', MLPClassifier()],

               ['LogisticRegression :', LogisticRegression()],

               ['ExtraTreesClassifier :', ExtraTreesClassifier()],

               ['AdaBoostClassifier :', AdaBoostClassifier()],

               ['GradientBoostingClassifier: ', GradientBoostingClassifier()],

               ['XGB :', XGBClassifier()],

               ['CatBoost :', CatBoostClassifier(logging_level='Silent')]]
for name,classifier in classifiers:

    classifier = classifier

    classifier.fit(X_train,y_train)

    predication = classifier.predict(X_test)

    accuracy = accuracy_score(y_test,predication)

    roc_score = roc_auc_score(y_test,predication)

    print('classifier name:',name,end='')

    print('accuracy:',accuracy,end='')

    print('roc auc score:',roc_score)

    print('----------------------')
best_classifier = MLPClassifier()
best_classifier.fit(X_train,y_train)
ypred = best_classifier.predict(X_test)
accuracy_score(y_test,ypred)
test = pd.read_csv(test_path)

test.head()
sentence_test = [test.loc[i,'v2'] for i in range(len(test['v2']))]
corpus_test = []



for i in range(len(sentence_test)):

    review = re.sub('[^a-zA-Z]',' ',sentence_test[i])

    review = review.lower()

    review = review.split()

    

    review = [porter.stem(word) for word in review if word not in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus_test.append(review)

    
X_test = tf.transform(corpus_test).toarray()
X_test.shape
X_test
ypred_test = best_classifier.predict(X_test)
y.shape
ypred_test.shape
print('so the accuracy on the test set is:',accuracy_score(y[:1115],ypred_test))