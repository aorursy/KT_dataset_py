# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from sklearn import datasets

%pylab inline

pylab.rcParams['figure.figsize'] = (10, 6)



iris = datasets.load_iris()



X = iris.data[:, [2, 3]]

y = iris.target
iris_df = pd.DataFrame(iris.data[:, [2, 3]], columns=iris.feature_names[2:])
print(iris_df.head())
print('\n' + 'The unique labels in this data are ' + str(np.unique(y)))
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)



print('There are {} samples in the training set and {} samples in the test set'.format(

X_train.shape[0], X_test.shape[0]))

print()
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()



sc.fit(X_train)



X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)



print('After standardizing our features, the first 5 rows of our data now look like this:\n')

print(pd.DataFrame(X_train_std, columns=iris_df.columns).head())
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt



markers = ('s', 'x', 'o')

colors = ('red', 'blue', 'lightgreen')

cmap = ListedColormap(colors[:len(np.unique(y_test))])

for idx, cl in enumerate(np.unique(y)):

    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],

               c=cmap(idx), marker=markers[idx], label=cl)
from sklearn.svm import SVC



svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)

svm.fit(X_train_std, y_train)



print('The accuracy of the svm classifier on training data is {:.2f} out of 1'.format(svm.score(X_train_std, y_train)))



print('The accuracy of the svm classifier on test data is {:.2f} out of 1'.format(svm.score(X_test_std, y_test)))
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

knn.fit(X_train_std, y_train)



print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(knn.score(X_train_std, y_train)))

print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(knn.score(X_test_std, y_test)))
import xgboost as xgb



xgb_clf = xgb.XGBClassifier()

xgb_clf = xgb_clf.fit(X_train_std, y_train)



print('The accuracy of the xgb classifier is {:.2f} out of 1 on training data'.format(xgb_clf.score(X_train_std, y_train)))

print('The accuracy of the xgb classifier is {:.2f} out of 1 on test data'.format(xgb_clf.score(X_test_std, y_test)))