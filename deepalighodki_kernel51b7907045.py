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
drugs = pd.read_csv("/kaggle/input/medicine-data/train.csv")
drugs.drop(['Id','drugName','condition','date','rating','usefulCount'], axis=1, inplace=True)

drugs
import re

def cleaning_text(i):

    i = re.sub("[^A-Za-z" "]+"," ",i).lower()

    i = re.sub("[0-9" "]+"," ",i)

    i= re.sub("[\W+""]", " ",i)        

    w = []

    for word in i.split(" "):

        if len(word)>3:

            w.append(word)

    return (" ".join(w))
drugs.review= drugs.review.apply(cleaning_text)

drugs = drugs.loc[drugs.review != " ",:]

drugs.review
test = pd.read_csv('../input/medicine-data/test.csv')
test.drop(['Id','drugName','condition','date','usefulCount'], axis=1, inplace=True)

test
test.review= test.review.apply(cleaning_text)

test = test.loc[test.review != " ",:]

test
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
def split_into_words(i):

    return [word for word in i.split(" ")]
# Preparing email texts into word count matrix format 

drugs_review1 = CountVectorizer(analyzer=split_into_words).fit(drugs.review)

drugs_review1
X=drugs.review

y=drugs['output'].map({'Yes': 0, 'No': 1})
# splitting data into train and test data sets 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=888)

X_train
# For all messages

all_matrix = drugs_review1.fit_transform(drugs.review)

all_matrix.shape # (40324, 26199)

# For training messages

train_matrix = drugs_review1.transform(X_train)

train_matrix.shape # (27017, 26199)

X_test_matrix = drugs_review1.transform(X_test)

X_test_matrix.shape

test_matrix = drugs_review1.transform(test.review)

test_matrix.shape # (13442, 26199)
tfidf_transformer = TfidfTransformer().fit(all_matrix)



train_tfidf = tfidf_transformer.transform(train_matrix)

train_tfidf.shape # (27017, 26199)



# Preparing TFIDF for Y_test 

test_tfidf = tfidf_transformer.transform(X_test_matrix)

test_tfidf.shape #  (13307, 26199)



# Preparing TFIDF for test data

testdata_tfidf = tfidf_transformer.transform(test_matrix)

testdata_tfidf.shape #  (13442, 26199)
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 777) 

X_train_oversampled,y_train_oversampled = sm.fit_sample(train_matrix, y_train)

X_train_oversampled.shape
from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes

classifier_mb = MB()

NB=classifier_mb.fit(X_train_oversampled,y_train_oversampled)

train_pred_m = NB.predict(train_matrix)

accuracy_train_m = np.mean(train_pred_m==y_train)

accuracy_train_m 
train_pred = NB.predict(X_test_matrix)

accuracy_train_m = np.mean(train_pred==y_test)

accuracy_train_m
all_pred=NB.predict(all_matrix)

accuracy_train_m = np.mean(all_pred==y)

drugs['predoutput']=all_pred

drugs.to_csv('drugs.csv',header=True)
test_pred_m = NB.predict(test_matrix)
result= pd.DataFrame(data=test_pred_m,columns=["output"])

result['output']=result.iloc[:,0].map({1:'No',0:'Yes'})

result.index.names = ['Id']

result.to_csv('MN.csv',header=True)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, random_state=0)

RF=classifier.fit(X_train_oversampled,y_train_oversampled)

RF_train_pred = RF.predict(train_matrix)

accuracy_train_m = np.mean(RF_train_pred==y_train)

accuracy_train_m
RF_test_pred = RF.predict(X_test_matrix)

accuracy_test_m = np.mean(RF_test_pred==y_test)

accuracy_test_m 
RF_test_pred = RF.predict(test_matrix)
RF_test_pred = classifier_mb.predict(test_matrix)

result= pd.DataFrame(data=RF_test_pred,columns=["output"])

result['output']=result.iloc[:,0].map({1:'No',0:'Yes'})

result.index.names = ['Id']

result.to_csv('RF.csv',header=True)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
seed=4

dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=seed)

bc = BaggingClassifier(base_estimator=dt, n_estimators=100, n_jobs=-1)

bc.fit(X_train_oversampled,y_train_oversampled) 

y_pred_train = bc.predict(train_matrix)

accuracy = accuracy_score(y_train, y_pred_train)

print(accuracy)
y_pred = bc.predict(X_test_matrix)

BDT_accuracy = accuracy_score(y_test, y_pred)

print(BDT_accuracy)
y_test_pred = bc.predict(test_matrix)
result= pd.DataFrame(data=y_test_pred ,columns=["output"])

result['output']=result.iloc[:,0].map({1:'No',0:'Yes'})

result.index.names = ['Id']

result.to_csv('BDT.csv',header=True)
from sklearn.neural_network import MLPClassifier

NN_mlp=MLPClassifier(hidden_layer_sizes=(5,5))

NN_mlp.fit(X_train_oversampled,y_train_oversampled)

pred_train=NN_mlp.predict(train_matrix)

accuracy = accuracy_score(y_train, pred_train)

print(accuracy)

pred_test=NN_mlp.predict(X_test_matrix)

accuracy = accuracy_score(y_test, pred_test)

print(accuracy)
NN_test_pred=NN_mlp.predict(test_matrix)
result= pd.DataFrame(data=NN_test_pred,columns=["output"])

result['output']=result.iloc[:,0].map({1:'No',0:'Yes'})

result.index.names = ['Id']

result.to_csv('NN.csv',header=True)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train_oversampled,y_train_oversampled)

Y_train_pred = knn.predict(train_matrix)

from sklearn import metrics

KNN_train=metrics.accuracy_score(y_train,Y_train_pred)#86%

KNN_train
Y_test_pred=knn.predict(X_test_matrix)

KNN_test=metrics.accuracy_score(y_test,Y_test_pred)

KNN_test