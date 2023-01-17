import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = pd.read_csv('../input/Iris.csv', index_col="Id")

data.head()
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
SEED = 256



le = LabelEncoder()

target = le.fit_transform(data['Species'])
X_train, X_test, y_train, y_test = train_test_split(data[data.columns[:-1]], target,

                                                    test_size=0.25, random_state=SEED)
clf = MLPClassifier((2, 3, 4), max_iter=1000)
clf.fit(X_train, y_train)

print("Точность классификации: ", accuracy_score(clf.predict(X_test), y_test))