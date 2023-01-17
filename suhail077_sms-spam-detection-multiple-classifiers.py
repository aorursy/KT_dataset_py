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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import string

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from nltk.stem import SnowballStemmer

from nltk.corpus import stopwords

%matplotlib inline
sms =pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')

sms.head()
sms = sms[['v1','v2']]

sms.rename(columns={'v1':'label','v2':'message'},inplace=True)

sms.groupby('label').describe()
sms['length'] = sms['message'].apply(len)

sms.head()
sms.hist(column='length',by='label',bins=50)
text = sms['message'].copy()



def pre_process(text):

    # Remove Punctuations

    text = text.translate(str.maketrans('','',string.punctuation))

    # Remove Stopwords 

    text = [word for word in text.split()if word.lower() not in stopwords.words('english')]

    return ' '.join(text)



text_cleaned = text.apply(pre_process)



# Tfidf Vectorizer

vectorizer = TfidfVectorizer('english')

features = vectorizer.fit_transform(text_cleaned)

print(features)
# Split into Train and Test



features_train, features_test, labels_train, labels_test = train_test_split(features,sms['label'],test_size=0.3,random_state=111)



# Import various classifiers

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import accuracy_score
svc = SVC(kernel='sigmoid', gamma=1.0)

knc = KNeighborsClassifier(n_neighbors=49)

mnb = MultinomialNB(alpha=0.2)

dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)

lrc = LogisticRegression(solver='liblinear', penalty='l1')

rfc = RandomForestClassifier(n_estimators=31, random_state=111)

abc = AdaBoostClassifier(n_estimators=62, random_state=111)

bc = BaggingClassifier(n_estimators=9, random_state=111)

etc = ExtraTreesClassifier(n_estimators=9, random_state=111)



# Make Dictionary to iterate through the classifiers

clfs = {'SVC' : svc,'KN' : knc, 'NB': mnb, 'DT': dtc, 'LR': lrc, 'RF': rfc, 'AdaBoost': abc, 'BgC': bc, 'ETC': etc}
def train_predict_classifier(classifier, feature_train, labels_train, features_test):

    classifier.fit(feature_train,labels_train)

    return classifier.predict(features_test)

predicted_scores=[]

for i,classifier in clfs.items():

    predicted = train_predict_classifier(classifier,features_train,labels_train,features_test)

    predicted_scores.append((i,[accuracy_score(labels_test,predicted)]))

    

df = pd.DataFrame.from_items(predicted_scores,orient='index',columns=['Score'])

df
df.plot(kind='bar',ylim=(0.9,1.0),figsize=(15,10),align='center')

plt.title('Classifier Distribution')
def stemmer(text):

    text = text.split()

    words=''

    for word in text:

        stemmer = SnowballStemmer('english')

        words+=(stemmer.stem(word))+' '

    return words



text_cleaned = text_cleaned.apply(stemmer)

features = vectorizer.fit_transform(text_cleaned)



features_train, features_test, labels_train, labels_test = train_test_split(features,sms['label'],test_size=0.3,random_state=111)



predicted_scores=[]

for i,classifier in clfs.items():

    predicted = train_predict_classifier(classifier,features_train,labels_train,features_test)

    predicted_scores.append((i,[accuracy_score(labels_test,predicted)]))

    

df1 = pd.DataFrame.from_items(predicted_scores,orient='index',columns=['Score after Stemming'])

df = pd.concat([df,df1],axis=1)

df
df.plot(kind='bar',ylim=(0.85,1.0),figsize=(15,10),align='center')

plt.title('Classifier Distribution')
lf = sms['length'].as_matrix()

newfeature = np.hstack((features.todense(),lf[:, None]))



features_train, features_test, labels_train, labels_test = train_test_split(newfeature,sms['label'],test_size=0.3,random_state=111)



predicted_scores=[]

for i,classifier in clfs.items():

    predicted = train_predict_classifier(classifier,features_train,labels_train,features_test)

    predicted_scores.append((i,[accuracy_score(labels_test,predicted)]))



df2 = pd.DataFrame.from_items(predicted_scores,orient='index',columns=['Score after Stemming with Length'])

df = pd.concat([df,df2],axis=1)

df
df.plot(kind='bar',ylim=(0.85,1.0),figsize=(15,10),align='center')

plt.title('Distribution by Classifier')
from sklearn.ensemble import VotingClassifier



eclf = VotingClassifier(estimators=[('BgC', bc), ('ETC', etc), ('RF', rfc), ('Ada', abc)], voting='soft')

eclf.fit(features_train,labels_train)

pred = eclf.predict(features_test)

print(accuracy_score(labels_test,pred))