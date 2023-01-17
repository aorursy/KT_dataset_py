import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

from ggplot import *

import nltk

df = pd.read_csv('../input/Sharktankpitchesdeals.csv')

## Check whether dataset is loaded

df.head()
def data_cleaning(corpus):

    letters_only = re.sub("[^a-zA-Z]", " ", corpus) 

    words = letters_only.lower().split()                            

    return( " ".join( words ))     
df['Pitched_Business_Desc'] = df['Pitched_Business_Desc'].apply(lambda x:data_cleaning(x))

df = df[['Deal_Status','Pitched_Business_Desc']]

for i in range(5):

    print(df['Pitched_Business_Desc'][i])
## Split into train/test sets

from sklearn.cross_validation import train_test_split

train, test = train_test_split(df,test_size=0.2)
## Vectorize

train_corpus = []

test_corpus = []

for each in train['Pitched_Business_Desc']:

    train_corpus.append(each)

for each in test['Pitched_Business_Desc']:

    test_corpus.append(each)

## Start creating them

from sklearn.feature_extraction.text import CountVectorizer

v = CountVectorizer(ngram_range=(2,2))

train_features = v.fit_transform(train_corpus)

test_features=v.transform(test_corpus)
print(train_features.shape)

print(test_features.shape)
# Import ML models from sklearn

from sklearn.linear_model import LogisticRegression # Regression classifier

from sklearn.tree import DecisionTreeClassifier # Decision Tree classifier

from sklearn import svm # Support Vector Machine

from sklearn.linear_model import SGDClassifier # Stochastic Gradient Descent Classifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # Random Forest and Gradient Boosting Classifier

from sklearn.naive_bayes import MultinomialNB # Naive Bayes Classifier 

from sklearn.metrics import accuracy_score, recall_score, confusion_matrix # Some metrics to check the performance of the models
# Setting parameters for each algorithm - these are tunable to achieve max accuracy



Classifiers = {'LR':LogisticRegression(random_state=10,C=5,max_iter=200),

               'DTC':DecisionTreeClassifier(random_state=10,min_samples_leaf=2),

               'RF':RandomForestClassifier(random_state=10,n_estimators=100,n_jobs=-1),

               'GBC':GradientBoostingClassifier(random_state=10,n_estimators=400,learning_rate=0.2),

               'SGD':SGDClassifier(loss="hinge", penalty="l2"),

               'SVM':svm.SVC(kernel='linear', C=0.1),

               'NB':MultinomialNB(alpha=.05)}

# Create a pipeline so you can reuse the code

def ML_Pipeline(clf_name):

    clf = Classifiers[clf_name]

    fit = clf.fit(train_features,train['Deal_Status'])

    pred = clf.predict(test_features)

    Accuracy = accuracy_score(test['Deal_Status'],pred)

    Confusion_matrix = confusion_matrix(test['Deal_Status'],pred)

    print('==='*20)

    print('Accuracy = '+str(Accuracy))

    print('==='*20) 

    print(Confusion_matrix)
ML_Pipeline('LR')
ML_Pipeline('DTC')
ML_Pipeline('RF')
ML_Pipeline('GBC')
ML_Pipeline('NB')
ML_Pipeline('SVM')
ML_Pipeline('SGD')
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import fbeta_score, make_scorer

ftwo_scorer = make_scorer(fbeta_score, beta=2)

ftwo_scorer

make_scorer(fbeta_score, beta=2)

parameters = {'kernel':('linear', 'rbf'), 'C':[0.01, 0.1, 1, 10, 100]}

svc = svm.SVC()

clf = GridSearchCV(svc, parameters, scoring=ftwo_scorer)

clf.fit(train_features,train['Deal_Status'])

print(clf.best_params_)