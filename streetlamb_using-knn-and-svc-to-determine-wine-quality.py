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
import pandas as pd

import sklearn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from scipy.stats import randint

from sklearn.preprocessing import StandardScaler

import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
wine_csv = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

wine_data = pd.DataFrame(wine_csv)

wine_data.head()
wine_data.describe()
wine_data.info()
corr_matrix = wine_data.corr()

corr_matrix["quality"].sort_values(ascending=False)
wine_data["quality"].hist()
wine_data_array = np.array(wine_data)

X = wine_data_array[:, :-1]

Y = wine_data_array[:, -1:]

Y = Y.ravel()

X, Y
#Splitting the data into train and test sets

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(X, Y):

  X_train, X_test = X[train_index], X[test_index]

  y_train, y_test = Y[train_index], Y[test_index]
#Confirm that the sets are representative of population

pd.DataFrame(y_train, columns=["quality"]).hist()
#Standardizing the features

standard_scalar = StandardScaler()

X_train_scaled = standard_scalar.fit_transform(X_train)

X_test_scaled = standard_scalar.transform(X_test)
param_distribs = {

    'n_neighbors': randint(low=1, high=200),

    'weights': ['uniform', 'distance'],

}

knn_clf = KNeighborsClassifier()

rnd_search_cv = RandomizedSearchCV(knn_clf, param_distributions=param_distribs,  n_iter=100, random_state=42, verbose=0 )

rnd_search_cv.fit(X_train_scaled, y_train)
best_knn_model = rnd_search_cv.best_estimator_

best_knn_model
from sklearn.metrics import classification_report, plot_confusion_matrix



y_pred = best_knn_model.predict(X_test_scaled)

print(classification_report(y_test, y_pred))
fig, ax = plt.subplots(figsize=(20, 10))

plot_confusion_matrix(best_knn_model, X_test_scaled, y_test,cmap=plt.cm.Blues, ax=ax)

plt.show()
from sklearn.svm import LinearSVC
param_distribs = {

    "C": np.arange(1,3,.1)

}

lin_svc = LinearSVC(dual=False)

rnd_search_cv = RandomizedSearchCV(lin_svc, param_distributions=param_distribs, random_state=42, verbose=0 )

rnd_search_cv.fit(X_train_scaled, y_train)
best_svc_model = rnd_search_cv.best_estimator_

best_svc_model
rnd_search_cv.best_score_
y_pred = best_svc_model.predict(X_test_scaled)

print(classification_report(y_test, y_pred))
fig, ax = plt.subplots(figsize=(20, 10))

plot_confusion_matrix(best_svc_model, X_test_scaled, y_test,cmap=plt.cm.Blues, ax=ax)

plt.show()