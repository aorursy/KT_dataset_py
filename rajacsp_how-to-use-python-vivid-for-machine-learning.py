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
!pip install python-vivid
# Dataset tools

from sklearn.datasets import make_classification



# Sklearn Metrics 

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.model_selection import train_test_split



# SKModels

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB



from vivid.utils import timer, get_logger
logger = get_logger(__name__)
def calculate_score(y_true, y_pred):

    

    _functions = [

        accuracy_score,

        f1_score,

        precision_score,

        recall_score

    ]

    

    score_map = {}

    

    for func in _functions:

        score_map[func.__name__] = func(y_true, y_pred)

        

    return score_map
X, y = make_classification(n_samples = int(2e2), n_features = 10,)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.5, shuffle = True)
logger.info(f'train: {X_train.shape[0]}, features: {X_train.shape[1]}')
# SVC

with timer(logger, prefix = '\ntime taken for svc : '):

    

    svc_model = SVC()

    

    svc_model.fit(X_train, y_train)

    y_pred = svc_model.predict(X_test)

    

    score = calculate_score(y_test, y_pred)

    

    logger.info(score)
# Random Classifier

with timer(logger, prefix = '\ntime taken for RandomClassifier: '):

    

    r_forest_model = RandomForestClassifier()

    

    r_forest_model.fit(X_train, y_train)

    y_pred = r_forest_model.predict(X_test)

    

    score = calculate_score(y_test, y_pred)

    

    logger.info(score)
with timer(logger, prefix = '\ntime taken for KNN: '):

    

    knn_model = KNeighborsClassifier()

    

    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)

    

    score = calculate_score(y_test, y_pred)

    

    logger.info(score)
# Logistic Regression



with timer(logger, prefix = '\n time taken for Logistic Regression: '):

    

    logreg_model = LogisticRegression()

    

    logreg_model.fit(X_train, y_train)

    y_pred = logreg_model.predict(X_test)

    

    score = calculate_score(y_test, y_pred)

    

    logger.info(score)
# Gaussian NB



with timer(logger, prefix = '\n time taken for Gaussian NB: '):

    

    gnb_model = GaussianNB()

    

    gnb_model.fit(X_train, y_train)

    y_pred = gnb_model.predict(X_test)

    

    score = calculate_score(y_test, y_pred)

    

    logger.info(score)