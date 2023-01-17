from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from wordcloud import WordCloud as wc   # not needed
from nltk.corpus import stopwords
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import string
import scipy
import numpy
import nltk
import json
import sys
import csv
import os
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

train_large = pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
test_large = pd.read_csv('../input/quora-insincere-questions-classification/test.csv')

train = train_large[:100000]
test = test_large[:100]

train.head()
test.head()
print('Shape of train:',train.shape)
print('Shape of test:',test.shape)
train['num_words']=train['question_text'].apply(lambda x : len(str(x).split()))
test['num_words']=test['question_text'].apply(lambda x : len(str(x).split()))
print('Max number of words in a question train dataset:',np.mean(train['num_words']))
print('Max number of words in a question test dataset:',test['num_words'].mean())
train['num_unique_words']=train['question_text'].apply(lambda x : len(set(str(x).split())))
test['num_unique_words']=test['question_text'].apply(lambda x : len(set(str(x).split())))


print('maximum of num_unique_words in train',train["num_unique_words"].max())

print("maximum of num_unique_words in test",test["num_unique_words"].max())
from nltk.corpus import stopwords
eng_stopwords=set(stopwords.words('english'))
train['num_stopwords']=train['question_text'].apply(lambda x : len([i for i in str(x).split() if i in eng_stopwords]))
test['num_stopwords']=test['question_text'].apply(lambda x : len([i for i in str(x).split() if i in eng_stopwords]))
train
train['num_punctuations']=train['question_text'].apply(lambda x : len([i for i in str(x).split() if i in string.punctuation]))
test['num_punctuations']=test['question_text'].apply(lambda x : len([i for i in str(x).split() if i in string.punctuation]))
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score



train['question_text'] = [entry.lower() for entry in train['question_text']]

test['question_text'] = [entry.lower() for entry in test['question_text']]

train['question_text']= [word_tokenize(entry) for entry in train['question_text']]

test['question_text']= [word_tokenize(entry) for entry in test['question_text']]

train.head()
train
np.random.seed(500)
# or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
# the tag_map would map any tag to 'N' (Noun) except
# Adjective to J, Verb -> v, Adverb -> R
# that means if you get a Pronoun then it would still be mapped to Noun


for index,entry in enumerate(train['question_text']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    
    # pos_tag function below will provide the 'tag' 
    # i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only 
        # alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
            
    # The final processed set of words for each iteration will be stored 
    # in 'question_text_final'
    train.loc[index,'question_text_final'] = str(Final_words)  
    
for index,entry in enumerate(test['question_text']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words_test = []
    
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    
    # pos_tag function below will provide the 'tag' 
    # i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only 
        # alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words_test.append(word_Final)
            
    # The final processed set of words for each iteration will be stored 
    # in 'question_text_final'
    test.loc[index,'question_text_final'] = str(Final_words_test) 
Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(train['question_text_final'])

Train_X_Tfidf = Tfidf_vect.transform(train['question_text_final'])

Test_X_Tfidf = Tfidf_vect.transform(test['question_text_final'])
train['target'].value_counts()

train.head()
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
lr.fit(x_train,y_train)
ypred=lr.predict(x_test)
print(accuracy_score(ypred,y_test))
x_train,x_test,y_train,y_test=train_test_split(Train_X_Tfidf,train['target'],test_size=0.1,random_state=5)
from sklearn.neighbors import KNeighborsClassifier
max=count=0
for i in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    ypred=knn.predict(x_test)
    acc=accuracy_score(ypred,y_test)
    if acc>max:
        max=acc
        count=i
    print('For i = ', i ,":" ,acc )
knn=KNeighborsClassifier(n_neighbors=count)
knn.fit(x_train,y_train)
ypred=knn.predict(x_test)
acc=accuracy_score(ypred,y_test)
print(acc)
#count= 6
max=id=0
max1=id1=0
for i in range(1,20):
    dt = DecisionTreeClassifier(criterion='gini',max_depth=i)
    dt.fit(x_train,y_train )
    ypred=dt.predict(x_test )
    print(accuracy_score(ypred,y_test))
    if accuracy_score(ypred,y_test) > max:
        max=accuracy_score(ypred,y_test)
        id=i
    
    dt = DecisionTreeClassifier(criterion='entropy',max_depth=i)
    dt.fit(x_train,y_train )
    ypred=dt.predict(x_test )
    print(accuracy_score(ypred,y_test))
    if accuracy_score(ypred,y_test) > max:
        max1=accuracy_score(ypred,y_test)
        id1=i
        
print("----------------")
print(max ,":", id)
print(max1 , ":", id1)
from sklearn.svm import SVC 
svm=SVC(kernel='linear')
svm.fit(x_train,y_train)
ypred=svm.predict(x_test)
print(accuracy_score(y_test,ypred))
for i in range(10,201,10):
    dt = RandomForestClassifier(n_estimators=i)
    dt.fit(x_train,y_train )
    ypred=dt.predict(x_test )
    print("For n = ", i , " :",accuracy_score(ypred,y_test))