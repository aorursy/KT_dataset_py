import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



data = pd.read_csv("../input/data.csv",header=0)

data.head()
data.info()
data.drop(['id','Unnamed: 32'], axis = 1, inplace = True)
data.head()
data['diagnosis'].value_counts()
diagnosis_dict = {'B': 0, 'M':1}

data['diagnosis'] = data['diagnosis'].map(diagnosis_dict)

data['diagnosis'].value_counts()
data.describe()
sns.pairplot(data, vars = list(data)[1:7], hue = 'diagnosis', palette="husl")

plt.figure(figsize = (20,20))

sns.heatmap(data.corr(), annot = True, cmap="coolwarm")

plt.show()
X = data.drop(['diagnosis'], axis =1)

y = data['diagnosis']
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix



min_X = X.min()

range_X = (X - min_X).max()

X_scaled = (X - min_X)/range_X



X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size = 0.3, random_state = 33)





from sklearn.svm import SVC

model_svc = SVC()

model_svc.fit(X_train, y_train)

y_pred = model_svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot = True, fmt='g', cmap="coolwarm")

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
tn, fp, fn, tp = cm.ravel()

print('False negatives:', fn)
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}



grid = GridSearchCV(SVC(), param_grid, refit = 'recall_score', scoring = 'recall')

grid.fit(X_train, y_train)



y_pred = grid.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
grid.best_params_
sns.heatmap(cm, annot = True, fmt='g', cmap="coolwarm")

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
tn, fp, fn, tp = cm.ravel()

print('False negatives:', fn)
from sklearn.decomposition import PCA



pca = PCA()

pca.fit(X_train)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.show()
for dim in [10, 8, 6, 4]:

    X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size = 0.3, random_state = 33)

    pca = PCA(n_components = dim)

    pca.fit(X_train)

    X_train = pca.transform(X_train)

    X_test = pca.transform(X_test)

    model_svc = SVC(C = 1, gamma = 1, kernel = 'rbf')

    model_svc.fit(X_train, y_train)

    y_pred = model_svc.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot = True, fmt='g', cmap="coolwarm")

    plt.title('%i dimmensions' %dim)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()