import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
heart = pd.read_csv('../input/heart-disease-uci/heart.csv')
heart.info()
heart.describe()
heart.sample(5)
heart.isnull().sum()
heart.isnull().any(axis=1).sum()
heart.groupby('target').mean()
plt.figure(figsize=(15,10))

sns.heatmap(heart.corr(), cmap='Spectral', annot=True)

plt.show()
abs(heart[heart['sex']==0].corr()['target'].drop(labels=['sex'])) - abs(heart[heart['sex']==1].corr()['target'].drop(labels=['sex']))
fig, ax = plt.subplots(figsize=(15,7))

ax.plot(heart[heart['sex']==0].corr()['target'].drop(labels=['sex', 'target']), label='Female')

ax.plot(heart[heart['sex']==1].corr()['target'].drop(labels=['sex', 'target']), label='Male')

ax.plot(heart.corr()['target'].drop(labels=['sex', 'target']), label='Both')

plt.title('Correlation of heart disease with various parameters')

plt.legend()
heart.groupby(['sex','target']).count()
heart['age range'] = pd.cut(heart['age'], bins=[0, 40, 50, 60, 70, 100])
sns.countplot(heart['age range'], hue='target', data=heart)
heart.groupby(['age range', 'sex', 'target'])['age'].count()
heart.groupby(['age range', 'target', 'sex'])['age'].count()
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for i in numerical_features:

    g = sns.FacetGrid(heart, col='sex', hue='target', height=5)

    g.map(sns.distplot, i)
fig, axes = plt.subplots(2, 4, figsize=(16,8))

for i, ax in enumerate(axes.ravel()):

    sns.countplot(heart[categorical_features[i]], ax=ax, hue=heart['target'])

    ax.set_xlabel(categorical_features[i])

plt.tight_layout()
pp = numerical_features

pp.append('target')

pp
sns.pairplot(heart.loc[:, pp], hue='target')
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from sklearn.compose import make_column_transformer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
heart.head(3)
score_mean = {}

score_max = {}
def process(dataframe, rand):

    y = dataframe['target']

    X = dataframe.drop(['target', 'age range'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = rand)

    mct = make_column_transformer(

            (OneHotEncoder(categories='auto', handle_unknown='ignore',sparse=False), ['cp', 'slope', 'thal']), 

            remainder=MinMaxScaler())

    X_train = mct.fit_transform(X_train)

    X_test = mct.transform(X_test)

    return X_train, X_test, y_train, y_test
def regression(dataframe, rand):

    y = dataframe['target']

    X = dataframe.drop(['target', 'age range'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = rand)

    mct = make_column_transformer(

            (OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore'), ['cp', 'slope', 'thal']), 

            remainder=MinMaxScaler())

    X_train = mct.fit_transform(X_train)

    logreg = LogisticRegression(solver='liblinear')

    logreg.fit(X_train, y_train)

    X_test = mct.transform(X_test)

    return logreg, X_test, y_test
scores = []

for i in range(0, 200):

    logreg, X_test, y_test = regression(heart, i)

    scores.append(logreg.score(X_test, y_test))
plt.figure(figsize=(15,5))

plt.plot(scores)

plt.xlabel('random state')

plt.ylabel('regression score')
np.array(scores).mean()
score_mean['Logistic Regression'] = np.round(np.array(scores).mean(), 2)
logreg, X_test, y_test = regression(heart, 153)
logreg.score(X_test, y_test)
score_max['Logistic Regression'] = np.round(logreg.score(X_test, y_test), 2)
predictions = logreg.predict(X_test)
confusion_matrix(y_test, predictions)
print(classification_report(y_test, predictions))
prob = logreg.predict_proba(X_test)
roc_score = roc_auc_score(y_test, predictions)
fpr, tpr, thresholds = roc_curve(y_test, prob[:, 1])
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_score)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([-0.05, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")
sns.heatmap(prob[np.argsort(prob[:, 0])])
plt.figure(figsize=(15,5))

plt.plot(prob[np.argsort(prob[:, 0])])

plt.xlabel('test case number')

plt.ylabel('probability')

plt.legend(['disease', 'no disease'])
heart_f = heart[heart['sex']==0].drop(['sex'], axis=1)
logreg_f, X_test, y_test = regression(heart_f, 0)
logreg_f.score(X_test, y_test)
heart_m = heart[heart['sex']==1].drop(['sex'], axis=1)
logreg_m, X_test, y_test = regression(heart_m, 33)
logreg_m.score(X_test, y_test)
scores = []

for i in range(0, 200):

    logreg, X_test, y_test = regression(heart_f, i)

    scores.append(logreg.score(X_test, y_test))
np.array(scores).mean()
plt.figure(figsize=(15,5))

plt.plot(scores)

plt.xlabel('random state')

plt.ylabel('regression score')
scores = []

for i in range(0, 200):

    logreg, X_test, y_test = regression(heart_m, i)

    scores.append(logreg.score(X_test, y_test))
np.array(scores).mean()
plt.figure(figsize=(15,5))

plt.plot(scores)

plt.xlabel('random state')

plt.ylabel('regression score')
from sklearn.neighbors import KNeighborsClassifier
max_score = []

for i in range(200):

    X_train, X_test, y_train, y_test = process(heart, i)

    score_list = []

    for i in range(1, 20):

        knn = KNeighborsClassifier(n_neighbors = i)

        knn.fit(X_train, y_train)

        score_list.append(knn.score(X_test, y_test))

    max_score.append(max(score_list))
plt.figure(figsize=(15,5))

plt.plot(max_score)
np.array(max_score).mean()
score_mean['K Nearest Neighbors'] = np.round(np.array(max_score).mean(), 2)
X_train, X_test, y_train, y_test = process(heart, 153)
score_list = []

for i in range(1, 20):

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(X_train, y_train)

    score_list.append(knn.score(X_test, y_test))
max(score_list)
plt.plot(range(1,20), score_list)

plt.xticks(range(1,20))

plt.xlabel("K value")

plt.ylabel("Score")

plt.show()
knn = KNeighborsClassifier(n_neighbors = 12)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
knn.score(X_test, y_test)
score_max['K Nearest Neighbors'] = np.round(knn.score(X_test, y_test), 2)
score_mean
from sklearn.svm import SVC
score_list = []

for i in range(200):

    X_train, X_test, y_train, y_test = process(heart, i)

    svm = SVC(1, gamma='scale')

    svm.fit(X_train, y_train)

    score_list.append(svm.score(X_test, y_test))
plt.figure(figsize=(15,5))

plt.plot(score_list)
np.array(score_list).mean()
score_mean['Support Vector Machines'] = np.round(np.array(score_list).mean(), 2)
X_train, X_test, y_train, y_test = process(heart, 153)
svm = SVC(1, gamma='scale')
svm.fit(X_train, y_train)
svm.score(X_test, y_test)
score_max['Support Vector Machines'] = np.round(svm.score(X_test, y_test), 2)
from sklearn.tree import DecisionTreeClassifier
score_list = []

for i in range(200):

    X_train, X_test, y_train, y_test = process(heart, i)

    dtc = DecisionTreeClassifier()

    dtc.fit(X_train, y_train)

    score_list.append(dtc.score(X_test, y_test))
plt.figure(figsize=(15,5))

plt.plot(score_list)
np.array(score_list).mean()
score_mean['Decision Tree Classifier'] = np.round(np.array(score_list).mean(), 2)
X_train, X_test, y_train, y_test = process(heart, 5)
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc.score(X_test, y_test)
score_max['Decision Tree Classifier'] = np.round(dtc.score(X_test, y_test), 2)
from sklearn.ensemble import RandomForestClassifier
score_list = []

for i in range(200):

    X_train, X_test, y_train, y_test = process(heart, i)

    rfc = RandomForestClassifier(n_estimators=100)

    rfc.fit(X_train, y_train)

    score_list.append(rfc.score(X_test, y_test))
plt.figure(figsize=(15,5))

plt.plot(score_list)
np.array(score_list).mean()
score_mean['Random Forest Classifier'] = np.round(np.array(score_list).mean(), 2)
X_train, X_test, y_train, y_test = process(heart, 153)
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc.score(X_test, y_test)
score_max['Random Forest Classifier'] = np.round(rfc.score(X_test, y_test), 2)
from sklearn.naive_bayes import GaussianNB
score_list = []

for i in range(200):

    X_train, X_test, y_train, y_test = process(heart, i)

    gnb = GaussianNB()

    gnb.fit(X_train, y_train)

    score_list.append(gnb.score(X_test, y_test))
plt.figure(figsize=(15,5))

plt.plot(score_list)
np.array(score_list).mean()
score_mean['Naive Bayes'] = np.round(np.array(score_list).mean(), 2)
X_train, X_test, y_train, y_test = process(heart, 153)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb.score(X_test, y_test)
score_max['Naive Bayes'] = np.round(gnb.score(X_test, y_test), 2)
from sklearn.neural_network import MLPClassifier
# To filter ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.

import warnings

warnings.filterwarnings('ignore') 
score_list = []

for i in range(200):

    X_train, X_test, y_train, y_test = process(heart, i)

    mlp = MLPClassifier(10, max_iter=200)

    mlp.fit(X_train, y_train)

    score_list.append(mlp.score(X_test, y_test))
plt.figure(figsize=(15,5))

plt.plot(score_list)
np.array(score_list).mean()
score_mean['Neural Network'] = np.round(np.array(score_list).mean(), 2)
X_train, X_test, y_train, y_test = process(heart, 153)
mlp = MLPClassifier(10, max_iter=200)
mlp.fit(X_train, y_train)
mlp.score(X_test, y_test)
score_max['Neural Network'] = np.round(mlp.score(X_test, y_test), 2)
score_max
score_mean
plt.figure(figsize=(10,5))

plt.plot(list(score_mean.keys()), list(score_mean.values()), 'b-o', label = 'mean score')

plt.plot(list(score_max.keys()), list(score_max.values()), 'r-*', label = 'max score')

for i, v in enumerate(score_mean.values()):

    plt.text(i, v+.01, v)

for i, v in enumerate(score_max.values()):

    plt.text(i, v+.01, v)

plt.xticks(rotation=45)

plt.ylim(0.7, 1)

plt.legend()