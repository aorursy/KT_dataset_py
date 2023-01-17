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

dataset = pd.read_csv('/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv')[:15000]

X = np.array([np.array(dataset).T[0], np.array(dataset).T[4]])

y = np.array(dataset).T[8]

X = X.astype(np.float64)

y = y.astype(int)

dataset.head()
import matplotlib.pyplot as plt

import collections

plt.scatter(X[0], X[1], c=y)

plt.title('Stars', fontsize=15)

plt.xlabel('Mean of the integrated profile', fontsize=13)

plt.ylabel('Mean of the DM-SNR curve', fontsize=13)

plt.show()



plt.bar(range(0, 2), [collections.Counter(y)[0], collections.Counter(y)[1]])

plt.title('Stars', fontsize=15)

plt.xlabel('Regular                            Pulsar', fontsize=13)

plt.ylabel('Number of stars', fontsize=13)

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.2, random_state=1)



from sklearn.neighbors import KNeighborsClassifier as knn

from sklearn.tree import DecisionTreeClassifier as dtc

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier as rfc



classifiers = {

    'Random Forest Classifier': rfc(n_estimators=100),

    'K Nearest Neighbors' : knn(),

    'Support Vector Classifier': SVC(gamma='scale'),

    'Decision Tree': dtc()

}



for i in classifiers.items():

    model = i[1]

    model.fit(X_train, y_train)
from mlxtend.plotting import plot_decision_regions

from sklearn.model_selection import cross_val_score as cvs

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score



for i in classifiers.items():

    model=i[1]

    scores = cvs(model, X_test, y_test, cv=5).mean()

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)

    print(i[0])

    print('Cross validation score: ' + str(scores))

    print('Model score:            ' + str(model.score(X_test, y_test)))

    print('Mean absolute error:    ' + str(mae))

    plot_decision_regions(X.T, y, clf=model)

    plt.title(i[0], fontsize=15)

    plt.xlabel('Mean of the integrated profile', fontsize=13)

    plt.ylabel('Mean of the DM-SNR curve', fontsize=13)

    plt.show()



model = list(classifiers.items())[1][1]

pred = model.predict(X_test)

submis = pd.DataFrame({'Real': y_test, 'Prediction': pred})

submis.to_csv('submission.csv', index=False)