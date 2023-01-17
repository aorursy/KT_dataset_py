import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt

from sklearn import model_selection

import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')
dataset = pd.read_csv(r"../input/baseball.csv")

dataset.head()


imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)

imputer = imputer.fit(dataset[['RankSeason', 'RankPlayoffs', 'OOBP', 'OSLG']])

dataset[['RankSeason', 'RankPlayoffs', 'OOBP', 'OSLG']] = imputer.transform(dataset[['RankSeason', 'RankPlayoffs', 'OOBP', 'OSLG']])

# replacing League with int

dataset.League.replace(['NL', 'AL'], [1, 0], inplace=True)
dataset = dataset[dataset.columns.difference(['RankPlayoffs', 'Team'])]

y = dataset.iloc[:, 9].values

y = np.ravel(y)

X = dataset[dataset.columns.difference(['Playoffs'])]
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling 

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

warnings.filterwarnings(action='once')
seed = 7

scoring = 'accuracy'

# Spot Check Algorithms

models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))
results = []

names = []

for name, model in models:

	kfold = model_selection.KFold(n_splits=10, random_state=seed)

	cv_results = model_selection.cross_val_score(estimator = model, X = X_train, y = y_train, cv =kfold, scoring=scoring)

	results.append(cv_results)

	names.append(name)

	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

	print(msg)
classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_train, y_train)
# Applying PCA

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_
classifier = GaussianNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue','yellow')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('red', 'green', 'blue','yellow','gray','pink','black'))(i), label = j)

plt.title('GaussianNB (Training set)')

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.legend()

plt.show()

# Visualising the Test set results

from matplotlib.colors import ListedColormap

X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue','yellow')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('red', 'green', 'blue','yellow','gray','pink','black'))(i), label = j)

plt.title('Logistic Regression (Test set)')

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.legend()

plt.show()