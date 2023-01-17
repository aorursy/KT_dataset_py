PATH_DATASET = "/kaggle/input/mnist-in-csv/"

TRAINING_DATASET = "mnist_train.csv"

TESTING_DATASET = "mnist_test.csv"
import pandas as pd

import os



def load_dataset(filename, path_dataset=PATH_DATASET):

    print("LOADED: " + filename)

    return pd.read_csv(os.path.join(path_dataset, filename))

    

train = load_dataset(TRAINING_DATASET)

test = load_dataset(TESTING_DATASET)
train.info()
test.info()
train.head()
test.head()
x_train, y_train, x_test, y_test = train.drop(['label'], axis=1), train.label, test.drop(['label'], axis=1), test.label
from sklearn.neighbors import KNeighborsClassifier



knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=3)

knn_clf.fit(x_train, y_train)
# from sklearn.model_selection import GridSearchCV

# from sklearn.neighbors import KNeighborsClassifier



# param_grid = [{"weights": ["uniform", "distance"], "n_neighbors": [3, 4, 5]}]



# knn_clf = KNeighborsClassifier()

# grid_search = GridSearchCV(knn_clf, param_grid, cv=2, verbose=3)

# grid_search.fit(x_train, y_train)
y_test_pred = knn_clf.predict(x_test)
y_test_pred
y_test
from sklearn.metrics import accuracy_score



accuracy_score(y_test, y_test_pred)
from sklearn.metrics import precision_score



precision_score(y_test, y_test_pred, average="micro")
from sklearn.metrics import recall_score



recall_score(y_test, y_test_pred, average="micro")
from sklearn.metrics import f1_score



f1_score(y_test, y_test_pred, average="micro")