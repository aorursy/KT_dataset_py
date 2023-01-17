import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/heart-disease-uci/heart.csv')
df.head()
df.dtypes
df.hist(figsize=(15,15))
plt.figure(figsize=(15,15))

sns.heatmap(df.corr(),annot=True,cmap='RdYlGn')
plt.bar(df.target.unique(),df.target.value_counts(),color=['red','green'])

plt.xticks([0,1])

print('No disease:{}%\nDisease:{}%'.format(round(df.target.value_counts(normalize=True)[0],2)*100,

                                           round(df.target.value_counts(normalize=True)[1],2)*100))
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import make_column_transformer

from sklearn.preprocessing import StandardScaler

column_trans = make_column_transformer(

                (OneHotEncoder(),['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']),

                (StandardScaler(),['age', 'trestbps', 'chol', 'thalach', 'oldpeak']),

                remainder = 'passthrough')
from sklearn.model_selection import train_test_split

X = df.drop(['target'], axis = 1)

y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
column_trans.fit_transform(X_train)
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline

logreg = LogisticRegression(solver='lbfgs')

pipe = make_pipeline(column_trans,logreg)
from sklearn.model_selection import cross_val_score

cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()
from sklearn.neighbors import KNeighborsClassifier

knn_scores = []

for k in range(1,31):

    knn_classifier = KNeighborsClassifier(n_neighbors = k)

    pipe = make_pipeline(column_trans,knn_classifier)

    knn_scores.append(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())
plt.figure(figsize=(12,12))

plt.plot([k for k in range(1, 31)], knn_scores, color = 'red')

for i in range(1,31):

    plt.text(i, knn_scores[i-1], (i, round(knn_scores[i-1]*100,2)))

plt.xticks([i for i in range(1, 31)])

plt.xlabel('Number of Neighbors (K)')

plt.ylabel('Scores')

plt.title('K Neighbors Classifier scores for different K values')
knn_scores[25]
from sklearn.svm import SVC

svc_scores = []

kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for i in range(len(kernels)):

    svc_classifier = SVC(kernel = kernels[i])

    pipe = make_pipeline(column_trans,svc_classifier)

    svc_scores.append(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())
from matplotlib.cm import rainbow

colors = rainbow(np.linspace(0, 1, len(kernels)))

plt.figure(figsize=(10,10))

plt.bar(kernels, svc_scores, color = colors)

for i in range(len(kernels)):

    plt.text(i, svc_scores[i], svc_scores[i])

plt.xlabel('Kernels')

plt.ylabel('Scores')

plt.title('Support Vector Classifier scores for different kernels')
svc_scores[0] #linear
from sklearn.tree import DecisionTreeClassifier

dt_scores = []

for i in range(1, len(X.columns) + 1):

    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)

    pipe = make_pipeline(column_trans,dt_classifier)

    dt_scores.append(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())
plt.figure(figsize=(10,10))

plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'green')

for i in range(1, len(X.columns) + 1):

    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))

plt.xticks([i for i in range(1, len(X.columns) + 1)])

plt.xlabel('Max features')

plt.ylabel('Scores')

plt.title('Decision Tree Classifier scores for different number of maximum features')
dt_scores[4]
from sklearn.ensemble import RandomForestClassifier

rf_scores = []

estimators = [10, 100, 200, 500, 1000]

for i in estimators:

    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)

    pipe = make_pipeline(column_trans,rf_classifier)

    rf_scores.append(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())
plt.figure(figsize=(10,10))

colors = rainbow(np.linspace(0, 1, len(estimators)))

plt.bar([i for i in range(len(estimators))], rf_scores, color = colors, width = 0.8)

for i in range(len(estimators)):

    plt.text(i, rf_scores[i], round(rf_scores[i],5))

plt.xticks(ticks = [i for i in range(len(estimators))], labels = [str(estimator) for estimator in estimators])

plt.xlabel('Number of estimators')

plt.ylabel('Scores')

plt.title('Random Forest Classifier scores for different number of estimators')
rf_scores[1]
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

pipe = make_pipeline(column_trans,nb)

cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()
pipe = make_pipeline(column_trans,logreg)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
from sklearn import metrics

metrics.accuracy_score(y_test,y_pred)*100
svc_classifier = SVC(kernel = 'linear')

pipe = make_pipeline(column_trans,svc_classifier)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
from sklearn import metrics

metrics.accuracy_score(y_test,y_pred)*100