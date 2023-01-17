from sklearn.datasets import load_iris

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier,GradientBoostingClassifier,BaggingClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings('ignore')
np.random.seed(20)
class clust():

    def _load_data(self, sklearn_load_ds):

        data = sklearn_load_ds

        X = pd.DataFrame(data.data)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, data.target, test_size=0.3,

                                                                                random_state=42)



    def __init__(self, sklearn_load_ds):

        self._load_data(sklearn_load_ds)



    def classify(self, model=LogisticRegression(random_state=42)):#if no classifier is given then Logistic as default

        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)

        print('Accuracy: {}'.format(accuracy_score(self.y_test, y_pred)))



    def Kmeans(self, output='add'):

        n_clusters = len(np.unique(self.y_train))

        clf = KMeans(n_clusters=n_clusters, random_state=42)

        clf.fit(self.X_train)

        y_labels_train = clf.labels_

        y_labels_test = clf.predict(self.X_test)

        if output == 'add':

            self.X_train['km_clust'] = y_labels_train

            self.X_test['km_clust'] = y_labels_test

            print('Train :',self.X_train)

            print('Test :',self.X_test)

            

        elif output == 'replace':

            self.X_train = y_labels_train[:, np.newaxis]

            self.X_test = y_labels_test[:, np.newaxis]

            print('Train :',self.X_train)

            print('Test :',self.X_test)

        else:

            raise ValueError('output should be either add or replace')

        return self
data=load_iris()



clust(data).Kmeans(output='add').classify(model=RandomForestClassifier())

# clust(data).Kmeans(output='add').classify(model=AdaBoostClassifier())

# clust(data).Kmeans(output='add').classify()
