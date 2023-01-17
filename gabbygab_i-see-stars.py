import pandas as pd

import pandas_profiling as pp

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')



# Metrics

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score



# Tuning

from sklearn.model_selection import GridSearchCV



# validation

from sklearn.model_selection import train_test_split, cross_val_score,KFold

from sklearn.pipeline import Pipeline



# Prerocessing

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer



# Machine learning models

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier



# Ensembles

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier



sns.set(style='whitegrid')

plt.style.use('seaborn-darkgrid')

%matplotlib inline









df = pd.read_csv('../input/pulsar_stars.csv')

df.head()

pp.ProfileReport(df)
sns.countplot(x='target_class', data=df)

plt.title('Class Distribution');
X = df.drop('target_class', axis=1)

y = df.target_class



X_train, X_test, y_train, y_test = train_test_split(X,

                                                   y,

                                                   test_size=0.3,

                                                   random_state=0)

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
scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

logis = LogisticRegression().fit(X_train,y_train)

y_pred = logis.predict(X_test)

print(accuracy_score(y_pred, y_test))
print(np.unique(y_pred))
print(classification_report(y_pred, y_test))
plt.figure(figsize=(10,5))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='2.0f');


scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)





penalty = ['l1', 'l2']

C = np.logspace(0, 4, 10)

hyperparameters = dict(C=C, penalty=penalty)



grid = GridSearchCV(LogisticRegression(), hyperparameters, cv=10, verbose=0)

grid_result = grid.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))



ensembles = []

ensembles.append(( ' ScaledAB ' , Pipeline([( ' Scaler ' , StandardScaler()),( ' AB ' ,

AdaBoostClassifier())])))

ensembles.append(( ' ScaledBG ' , Pipeline([( ' Scaler ' , StandardScaler()),( ' AB ' ,

BaggingClassifier())])))

ensembles.append(( ' ScaledGBM ' , Pipeline([( ' Scaler ' , StandardScaler()),( ' GBM ' ,

GradientBoostingClassifier())])))

ensembles.append(( ' ScaledRF ' , Pipeline([( ' Scaler ' , StandardScaler()),( ' RF ' ,

RandomForestClassifier())])))



                 

results = []

names = []

for name, model in ensembles:

    kfold = KFold(n_splits=10, random_state=0)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

bagging = BaggingClassifier().fit(X_train,y_train)

y_pred = bagging.predict(X_test)

print(accuracy_score(y_pred, y_test))
print(classification_report(y_pred, y_test))
plt.figure(figsize=(10,5))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='2.0f');