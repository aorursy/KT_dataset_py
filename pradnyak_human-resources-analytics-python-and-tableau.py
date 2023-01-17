import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns



# Importing the dataset

dataset = pd.read_csv('../input/HR_comma_sep.csv')

X = dataset.iloc[:, dataset.columns != 'left'].values

y = dataset.iloc[:, 6].values



#quick view

dataset.isnull().any()

dataset.head()

dataset.shape

dataset.dtypes

dataset.describe
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

X[:, 8] = labelencoder_X.fit_transform(X[:, 8])

X[:, 7] = labelencoder_X.fit_transform(X[:, 7])

onehotencoder = OneHotEncoder(categorical_features = [8, 7])

X = onehotencoder.fit_transform(X).toarray()



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
corr = dataset.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
                    #3#

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0, C = 10, gamma = 1.1)

classifier.fit(X_train, y_train)

                    #4#

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)

classifier.feature_importances_

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

from sklearn.model_selection import cross_val_score

cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()