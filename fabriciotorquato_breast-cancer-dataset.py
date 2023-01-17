from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score
breast_cancer = load_breast_cancer()

data = pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)

data['target'] = pd.Series(breast_cancer.target)

data.head()
data.info()
data.describe()
col = data.columns       # .columns gives columns names in data 

print(col)
ax = sns.countplot(data.target,label="Count")

B, M = data.target.value_counts()

print('Number of Benign: ',B)

print('Number of Malignant : ',M)
print(data.columns)
featureMeans = list(data.columns[:10])

featureMeans
correlationData = data[featureMeans].corr()

sns.pairplot(data[featureMeans].corr(), diag_kind='kde', size=2);
plt.figure(figsize=(10,10))

sns.heatmap(data[featureMeans].corr(), annot=True, square=True, cmap='coolwarm')

plt.show()
bins = 12

plt.figure(figsize=(15,15))

for idx,atr in enumerate(featureMeans):

    plt.subplot(5, 2, idx+1)

    sns.distplot(data[data['target']==1][atr], bins=bins, color='green', label='M')

    sns.distplot(data[data['target']==0][atr], bins=bins, color='red', label='B')

    plt.legend(loc='upper right')

plt.tight_layout()

plt.show()
from sklearn.decomposition import PCA



import matplotlib.pyplot as plt



pca = PCA(n_components=2)

X_r = pca.fit_transform(data.loc[:,featureMeans])



colors = ['navy', 'turquoise']

for color, i, target_name in zip(colors, [0, 1], data.loc[:, 'target']):

    plt.scatter(X_r[data.loc[:, 'target'] == i, 0], X_r[data.loc[:, 'target'] == i, 1], color=color, alpha=.8, label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)

plt.title('PCA')

plt.plot()
X = data.loc[:,featureMeans]

y = data.loc[:, 'target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.95, random_state = 42)
pipeline = Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors = 5))])

scores = cross_val_score(pipeline, X, y, cv=5) 



print('Acurácia - %.2f +- %.2f' % (scores.mean() * 100, scores.std() * 100))
data_gs, data_cv, target_gs, target_cv = train_test_split(X, y, test_size=0.95, random_state=42)
pipeline = Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())])

parameters = {'clf__n_neighbors': [3,4,5,6,7], 'clf__p': [1, 2, 5]}

clf = GridSearchCV(pipeline, 

                    parameters,

                    scoring='accuracy',

                    cv=5)

clf.fit(data_gs, target_gs)

scores = cross_val_score(clf.best_estimator_, data_cv, target_cv, cv=5)



print(clf.best_params_)

print('Accuracy - %.2f +- %.2f' % (scores.mean() * 100, scores.std() * 100))
clf = clf.best_estimator_

kf = StratifiedKFold(n_splits = 5)

acc = []

for train_index, test_index in kf.split(data_cv, target_cv):

    X_train,X_test = data_cv.iloc[train_index],data_cv.iloc[test_index]

    y_train,y_test = target_cv.iloc[train_index],target_cv.iloc[test_index]

    

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    acc.append(accuracy_score(y_pred,y_test))



acc = np.array(acc)

print('Accuracy - %.2f +- %.2f' % (acc.mean() * 100, acc.std() * 100))