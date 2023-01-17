import pandas as pd

import numpy as np

np.set_printoptions(precision=4,suppress=True)

import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score, f1_score

import warnings

warnings.filterwarnings("ignore",category=FutureWarning)

warnings.filterwarnings("ignore",category=DeprecationWarning)

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning) 
heart = pd.read_csv("../input/heart.csv")

heart.head()
heart.info()
heart.describe()
checkNANs = heart.isnull().any(axis=1)  # axis=1 checks every row

sum(checkNANs)
heart['target'].value_counts()
heart.hist(bins = 50, figsize=(18,10)) 

plt.show()
heart['cp'] = heart['cp'].astype('category')

heart['sex'] = heart['sex'].astype('category')

heart['exang'] = heart['exang'].astype('category')



heart = pd.get_dummies(heart)
corr_matrix = heart.corr()

print (corr_matrix['target'].sort_values(ascending = False))
X = heart.drop('target',axis=1)

y = heart['target']
SVC_train_set_accuracy = []

SVC_test_set_accuracy = []

RFC_train_set_accuracy = []

RFC_test_set_accuracy = []



for _ in range(0,20):

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25, random_state = np.random.randint(100), stratify=y)

    

    scaler = MinMaxScaler()

    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    

    param_grid = [{'kernel': ['rbf'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},

              {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}]

    grid_search = GridSearchCV(SVC(), param_grid, cv=10)

    grid_search.fit(X_train_scaled, y_train)

    SVC_train_set_accuracy.append(grid_search.best_score_)

    SVC_test_set_accuracy.append(grid_search.score(X_test_scaled,y_test))

    

    forest = RandomForestClassifier(max_features = 3)

    forest.fit(X_train_scaled, y_train)

    RFC_train_set_accuracy.append(forest.score(X_train_scaled, y_train))

    RFC_test_set_accuracy.append(forest.score(X_test_scaled, y_test))





print("SVC training set accuracy: {:.2f} ± {:.2f} ".format(np.array(SVC_train_set_accuracy).mean(),np.array(SVC_train_set_accuracy).std()))

print("SVC test set accuracy: {:.2f} ± {:.2f}".format(np.array(SVC_test_set_accuracy).mean(),np.array(SVC_test_set_accuracy).std())) 

print("RFC training set accuracy: {:.2f} ± {:.2f}".format(np.array(RFC_train_set_accuracy).mean(),np.array(RFC_train_set_accuracy).std()))

print("RFC test set accuracy: {:.2f} ± {:.2f}".format(np.array(RFC_test_set_accuracy).mean(), np.array(RFC_test_set_accuracy).std()))