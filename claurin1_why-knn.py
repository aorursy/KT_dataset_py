import numpy as np # linear algebra

from scipy.stats import pearsonr

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.api.types import union_categoricals

from matplotlib import pyplot as plt 

import seaborn as sns

from pprint import pprint

from time import time



from os import listdir

from os import path



## Much code copied from the Faces example taken from scikit 

## (https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html)

from time import time

import logging

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from sklearn.neighbors import (KNeighborsClassifier,

                               NeighborhoodComponentsAnalysis)

from sklearn.linear_model import LogisticRegression



print(__doc__)

## Display progress logs on stdout

logging.basicConfig(level = logging.INFO,

                   format = '%(asctime)s %(message)s')

random_state = 3412
train_data = pd.read_csv(path.join("..", "input", "learn-together", "train.csv"))

test_data = pd.read_csv(path.join("..", "input", "learn-together", "test.csv"))
X = train_data.drop(['Cover_Type', 'Id'], axis = 1)

X_test = test_data.drop(['Id'], axis = 1)



y = train_data['Cover_Type']

test_Id = test_data.Id

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25, random_state = 124127)



target_names = np.array(['Spruce/Fir',

                             'Lodgepole Pine',

                             'Ponderosa Pine',

                             'Cottonwood/Willow',

                             'Aspen',

                             'Douglas-fir',

                             'Krummholz'])

# Quantitative evaluation of the predictions using matplotlib

d = [1,2,4]

K = [1,3,5,9, 17]



d_by_K = [[d0, k0] for d0 in d for k0 in K]

knn_f1_results = []

logistic_f1_results = []

nca_coeffs = {}

nca_knn_f1_list = []

nca_logit_f1_list = []



for i, (dim, n_nb) in enumerate(d_by_K):



    print("Fitting the classifier to the training set")

    t0 = time()

    nca = NeighborhoodComponentsAnalysis(max_iter = 50, random_state = random_state,

                                        n_components = dim)

    nca = nca.fit(X_train, y_train)

    nca_coeffs[str(i)] = nca.components_

    clf = KNeighborsClassifier(n_neighbors = n_nb)

    clf = clf.fit(nca.transform(X_train), y_train)

    if i % 5 == 0:

        logit_clf = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs' )

        logit_clf = logit_clf.fit(nca.transform(X_train), y_train)

    

    print("done in {:3f}s".format(time() - t0))

    print("Predicting forest cover on the validation set to work on featurizing.")

    t0 = time()

    y_pred = clf.predict(nca.transform(X_valid))

    if i % 5 == 0:

        y_logit_pred = logit_clf.predict(nca.transform(X_valid))

    

    print("Done in {:3f}s".format(time() - t0))

    nca_knn_f1 = f1_score(y_valid, y_pred, average = 'weighted')

    nca_knn_f1_list.append(nca_knn_f1)

    

    if i % 5 == 0:

        nca_logit_f1 = f1_score(y_valid, y_logit_pred, average = 'weighted')

        nca_logit_f1_list.append(nca_logit_f1)

        print('i: {}'.format(i))
