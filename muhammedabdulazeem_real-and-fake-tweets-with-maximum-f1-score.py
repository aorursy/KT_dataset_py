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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sample = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
train.head()
test.head()
train.shape,test.shape
train.isnull().sum()
test.isnull().sum()
train.drop(['keyword','location'],axis=1,inplace=True)

test.drop(['keyword','location'],axis=1,inplace=True)
train.head()
train = train.sample(frac=1)



# amount of fraud classes 492 rows.

train_1 = train.loc[train['target'] == 1]

train_0 = train.loc[train['target'] == 0][:3271]



normal_distributed_df = pd.concat([train_1, train_0])



# Shuffle dataframe rows

train = normal_distributed_df.sample(frac=1, random_state=42)



train.head()
train['target'].value_counts()
train_labels = train['target']
train_labels.value_counts()
import nltk

nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords

import re
tf_V = TfidfVectorizer(max_features=None,ngram_range=(1,3),analyzer='word')

bow = CountVectorizer(max_features=None,ngram_range=(1,3))

porter = PorterStemmer()

lemma = WordNetLemmatizer()
def clean_text_with_porter(para):

    corpus = []

    for i in range(len(para)):

        review = re.sub('^[a-zA-Z]',' ',para[i])

        review = review.lower()

        review = review.split()

        review = [porter.stem(word) for word in review if word not in set(stopwords.words('english'))]

        review = ' '.join(review)

        corpus.append(review)

    return corpus    
def clean_text_with_lemma(para):

    corpus = []

    for i in range(len(para)):

        review = re.sub('^[a-zA-Z]',' ',para[i])

        review = review.lower()

        review = review.split()

        review = [lemma.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]

        review = ' '.join(review)

        corpus.append(review)

    return corpus  
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid = train_test_split(train['text'],train_labels,test_size=0.15,random_state = 42,shuffle=True,stratify=train_labels)
X_train.shape,X_valid.shape
X_train = pd.DataFrame(X_train)

X_train.reset_index(inplace=True)
X_valid = pd.DataFrame(X_valid)

X_valid.reset_index(inplace=True)
X_train.head()
sentences_train = [X_train.loc[i,'text'] for i in range(len(X_train['text']))]



sentences_test = [X_valid.loc[i,'text'] for i in range(len(X_valid['text']))]
sentences_train = clean_text_with_porter(sentences_train)

sentences_test = clean_text_with_porter(sentences_test)
len(sentences_train),len(sentences_test)
X_train_matrix = tf_V.fit_transform(sentences_train).toarray()
X_train_matrix.shape
X_train_matrix
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB
LogReg = LogisticRegression()
LogReg.fit(X_train_matrix,y_train)
X_valid_matrix = tf_V.transform(sentences_test).toarray()
X_valid_matrix.shape
ypred_Logreg = LogReg.predict(X_valid_matrix)
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
print('----Logistic Regression-----')

print('accuracy:',accuracy_score(ypred_Logreg,y_valid))

print('precision_score:',precision_score(ypred_Logreg,y_valid))

print('recall:',recall_score(ypred_Logreg,y_valid))

print('f1_score:',f1_score(ypred_Logreg,y_valid))
rfc = RandomForestClassifier()

rfc.fit(X_train_matrix,y_train)
ypred_rfc = rfc.predict(X_valid_matrix)
print('----Random Forest Classifier-----')

print('accuracy:',accuracy_score(ypred_rfc,y_valid))

print('precision_score:',precision_score(ypred_rfc,y_valid))

print('recall:',recall_score(ypred_rfc,y_valid))

print('f1_score:',f1_score(ypred_rfc,y_valid))
naive = MultinomialNB()

naive.fit(X_train_matrix,y_train)
ypred_naive = naive.predict(X_valid_matrix)
print('----Naive Bayes MultiNomial-----')

print('accuracy:',accuracy_score(ypred_naive,y_valid))

print('precision_score:',precision_score(ypred_naive,y_valid))

print('recall:',recall_score(ypred_naive,y_valid))

print('f1_score:',f1_score(ypred_naive,y_valid))