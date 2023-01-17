import pandas as pd

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

%matplotlib inline 
data = pd.read_csv('../input/pokemon/Pokemon.csv')
data.head(10)
data.shape
# Removing irrelevant features(# and Name) and features with Nan values(Type 2)

data = data.drop(['#','Type 2','Name'],axis='columns')
data.head(10)
data.Legendary.value_counts()
legendaryPokemon = data.loc[data['Legendary']==True]

legendaryPokemon = legendaryPokemon.append(legendaryPokemon.append(legendaryPokemon))

bal_data = data.append(legendaryPokemon.append(legendaryPokemon.append(legendaryPokemon)))
# mapping true and false to 1 and 0 respectively

bal_data['Legendary'] = bal_data.Legendary.map({False:0,True:1})
from sklearn.compose import make_column_transformer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

col_trans = make_column_transformer(

            (OneHotEncoder(),['Type 1','Generation']),

            (StandardScaler(),['Total','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']),

            remainder = 'passthrough')
df = bal_data
from sklearn.model_selection import train_test_split

X = df.drop(['Legendary'], axis = 1)

y = df['Legendary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
col_trans.fit_transform(X_train)
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline

logreg = LogisticRegression(solver='lbfgs')

pipe = make_pipeline(col_trans,logreg)
from sklearn.model_selection import cross_val_score

print('Accuracy score on Train data: {}'.format(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()*100))
pipe = make_pipeline(col_trans,logreg)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

from sklearn import metrics

print('Accuracy score on Test data: {}'.format(metrics.accuracy_score(y_test,y_pred)*100))
from sklearn.neighbors import KNeighborsClassifier

knn_scores = []

for k in range(1,31):

    knn_classifier = KNeighborsClassifier(n_neighbors = k)

    pipe = make_pipeline(col_trans,knn_classifier)

    knn_scores.append(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())
plt.figure(figsize=(12,12))

plt.plot([k for k in range(1, 31)], knn_scores, color = 'red')

for i in range(1,31):

    plt.text(i, knn_scores[i-1], (i, round(knn_scores[i-1]*100,2)))

plt.xticks([i for i in range(1, 31)])

plt.xlabel('Number of Neighbors (K)')

plt.ylabel('Scores')

plt.title('K Neighbors Classifier scores for different K values')
print('Accuracy score on Train data: {}'.format(knn_scores[1]*100))
knn_classifier = KNeighborsClassifier(n_neighbors = 2)

pipe = make_pipeline(col_trans,knn_classifier)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('Accuracy score on Test Data: {}'.format(metrics.accuracy_score(y_test,y_pred)*100))
from sklearn.svm import SVC

svc_scores = []

kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for i in range(len(kernels)):

    svc_classifier = SVC(kernel = kernels[i])

    pipe = make_pipeline(col_trans,svc_classifier)

    svc_scores.append(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())
from matplotlib.cm import rainbow

import numpy as np

colors = rainbow(np.linspace(0, 1, len(kernels)))

plt.figure(figsize=(10,7))

plt.bar(kernels, svc_scores, color = colors)

for i in range(len(kernels)):

    plt.text(i, svc_scores[i], svc_scores[i])

plt.xlabel('Kernels')

plt.ylabel('Scores')

plt.title('Support Vector Classifier scores for different kernels')
print('Accuracy score on Train data: {}'.format(svc_scores[0]*100))
svc_classifier = SVC(kernel = 'linear')

pipe = make_pipeline(col_trans,svc_classifier)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('Accuracy score on Test data: {}'.format(metrics.accuracy_score(y_test,y_pred)*100))
from sklearn.tree import DecisionTreeClassifier

dt_scores = []

for i in range(1, len(X.columns) + 1):

    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)

    pipe = make_pipeline(col_trans,dt_classifier)

    dt_scores.append(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())
plt.figure(figsize=(10,10))

plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'green')

for i in range(1, len(X.columns) + 1):

    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))

plt.xticks([i for i in range(1, len(X.columns) + 1)])

plt.xlabel('Max features')

plt.ylabel('Scores')

plt.title('Decision Tree Classifier scores for different number of maximum features')
print('Accuracy score on Train data: {}'.format(dt_scores[5]*100))
dt_classifier = DecisionTreeClassifier(max_features = 6, random_state = 0)

pipe = make_pipeline(col_trans,dt_classifier)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('Accuracy  score on Test data: {}'.format(metrics.accuracy_score(y_test,y_pred)*100))
from sklearn.ensemble import RandomForestClassifier

rf_scores = []

estimators = [10, 100, 200, 500, 1000]

for i in estimators:

    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)

    pipe = make_pipeline(col_trans,rf_classifier)

    rf_scores.append(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())
plt.figure(figsize=(10,7))

colors = rainbow(np.linspace(0, 1, len(estimators)))

plt.bar([i for i in range(len(estimators))], rf_scores, color = colors, width = 0.8)

for i in range(len(estimators)):

    plt.text(i, rf_scores[i], round(rf_scores[i],5))

plt.xticks(ticks = [i for i in range(len(estimators))], labels = [str(estimator) for estimator in estimators])

plt.xlabel('Number of estimators')

plt.ylabel('Scores')

plt.title('Random Forest Classifier scores for different number of estimators')
print('Accuracy score on Train data: {}'.format(rf_scores[0]*100))
rf_classifier = RandomForestClassifier(n_estimators = 10, random_state = 0)

pipe = make_pipeline(col_trans,rf_classifier)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('Accuracy score on Test data: {}'.format(metrics.accuracy_score(y_test,y_pred)*100))