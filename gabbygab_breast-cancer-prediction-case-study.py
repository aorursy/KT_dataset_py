import os

import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import plotly 

import warnings

warnings.filterwarnings('ignore')





from sklearn.feature_selection import RFE

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import Binarizer, StandardScaler, Normalizer

from sklearn.model_selection import train_test_split, cross_val_score, KFold

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

%matplotlib inline

sns.set(style='whitegrid')

plt.style.use('seaborn-darkgrid')



df = pd.read_csv('../input/data.csv')

df.head()
print('Dataset size:', df.shape)

print('---'*48)

print('Dataset columns:',df.columns)
df.dtypes
df.isnull().sum()
lst = ['id', 'Unnamed: 32']

df.drop(lst,axis=1, inplace=True)
df.describe()
diagnosis = df.diagnosis.value_counts()

ax = sns.countplot(x='diagnosis', data=df)

plt.title('Class Distribution');

print(diagnosis)
corr = df.corr(method='pearson')

corr
f, ax = plt.subplots(figsize=(20, 12))

sns.heatmap(corr, fmt="f",cmap=plt.cm.Blues);
df.skew()
df.hist(figsize=(20,15));
df.plot(kind= 'density' , subplots=True, layout=(10,5), sharex=False, figsize=(20,30));
X = df.drop('diagnosis',axis=1)

y = df.diagnosis





X_train, X_test, y_train, y_test = train_test_split(X,

                                                   y,

                                                   test_size=0.3,

                                                   random_state=0)

print('X_train: ',X_train.shape)

print('X_test: ',X_test.shape)

print('y_train: ',y_train.shape)

print('y_test: ',y_test.shape)
models = []

models.append(( ' LR ' , LogisticRegression()))

models.append(( ' LDA ' , LinearDiscriminantAnalysis()))

models.append(( ' KNN ' , KNeighborsClassifier()))

models.append(( ' RF ' , RandomForestClassifier()))

models.append(( ' NB ' , GaussianNB()))

models.append(( ' SVM ' , SVC()))



results = []

names = []



for name, model in models:

    Kfold = KFold(n_splits=10, random_state=0)

    cv_results = cross_val_score(model, X_train, y_train, cv=Kfold, scoring= 'accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std());

    print(msg)


pipelines = []

pipelines.append(( ' ScaledLR ' , Pipeline([( 'Scaler' , StandardScaler()),( ' LR ' ,

LogisticRegression())])))

pipelines.append(( ' ScaledLDA ' , Pipeline([( 'Scaler' , StandardScaler()),( ' LDA ' ,

LinearDiscriminantAnalysis())])))

pipelines.append(( ' ScaledKNN ' , Pipeline([( ' Scaler ' , StandardScaler()),( ' KNN ' ,

KNeighborsClassifier())])))

pipelines.append(( ' ScaledRF ' , Pipeline([( ' Scaler ' , StandardScaler()),( ' RandomForest ' ,

RandomForestClassifier())])))

pipelines.append(( ' ScaledNB ' , Pipeline([( ' Scaler ' , StandardScaler()),( ' NB ' ,

GaussianNB())])))

pipelines.append(( ' ScaledSVM ' , Pipeline([( ' Scaler' , StandardScaler()),( ' SVM ' , SVC())])))



results = []

names = []



for name, model in pipelines:

    kfold = KFold(n_splits=10, random_state=0)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)


pipelines = []

pipelines.append(( ' NormLR ' , Pipeline([( 'Normalizer' , Normalizer()),( ' LR ' ,

LogisticRegression())])))

pipelines.append(( ' NormLDA ' , Pipeline([( 'Normalizer' , Normalizer()),( ' LDA ' ,

LinearDiscriminantAnalysis())])))

pipelines.append(( ' NormLKNN ' , Pipeline([( ' Normalizer ' , Normalizer()),( ' KNN ' ,

KNeighborsClassifier())])))

pipelines.append(( ' NormRandomForest ' , Pipeline([( ' Normalizer ' , Normalizer()),( ' RandomForest ' ,

RandomForestClassifier())])))

pipelines.append(( ' NormNB ' , Pipeline([( ' Normalizer ' , Normalizer()),( ' NB ' ,

GaussianNB())])))

pipelines.append(( ' NormSVM ' , Pipeline([( ' Normalizer' , Normalizer()),( ' SVM ' , SVC())])))



results = []

names = []



for name, model in pipelines:

    kfold = KFold(n_splits=10, random_state=0)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
pipelines = []

pipelines.append(( ' BinLR ' , Pipeline([( 'Binarizer' , Binarizer(threshold=0.0)),( ' LR ' ,

LogisticRegression())])))

pipelines.append(( ' BinLDA ' , Pipeline([( 'Binarizer' , Binarizer(threshold=0.0)),( ' LDA ' ,

LinearDiscriminantAnalysis())])))

pipelines.append(( ' BinKNN ' , Pipeline([( 'Binarizer ' , Binarizer(threshold=0.0)),( ' KNN ' ,

KNeighborsClassifier())])))

pipelines.append(( ' BinRandomForest ' , Pipeline([( 'Binarizer ' , Binarizer(threshold=0.0)),( ' RandomForest ' ,

RandomForestClassifier())])))

pipelines.append(( ' BinNB ' , Pipeline([( ' Binarizer ' , Binarizer(threshold=0.0)),( ' NB ' ,

GaussianNB())])))

pipelines.append(( 'BinSVM ' , Pipeline([( ' Binarizer' , Binarizer(threshold=0.0)),( ' SVM ' , SVC())])))



results = []

names = []



for name, model in pipelines:

    kfold = KFold(n_splits=10, random_state=0)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test =scaler.transform(X_test)

log = LogisticRegression().fit(X_train,y_train)



y_pred = log.predict(X_test)

print('Accuracy: ', accuracy_score(y_test,y_pred))
plt.figure(figsize=(10,5))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='2.0f');
print(classification_report(y_test,y_pred))