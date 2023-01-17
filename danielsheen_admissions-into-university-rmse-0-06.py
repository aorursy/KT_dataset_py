from pandas import read_csv

import pandas as pd

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

from matplotlib import pyplot

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

import numpy as np
df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')

df.head()
# indexing on Serial No.

df.set_index('Serial No.', inplace = True)
df.head()
# luckily we're only dealing with numbers and clean up of NaN values/invalid values is unecessary

df.info()
df.hist(bins = 50, figsize = (20, 15))

plt.show()
scatter_matrix(df)

plt.show()
corr_matrix = df.corr()

corr_matrix["Chance of Admit "].sort_values(ascending = False)
corr_matrix
train_set, test_set = train_test_split(df, test_size = 0.25, random_state = 42)

# training set

X_train = train_set.values[:,0:7]

y_train = train_set.values[:,7]



# test set

X_test = test_set.values[:,0:7]

y_test = test_set.values[:,7]
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler



minmax_scaler = MinMaxScaler(feature_range = (0, 1)).fit(X_train)

standard_scaler = StandardScaler().fit(X_train)



# Min-Max 

X_train_MM = minmax_scaler.transform(X_train)

X_test_MM = minmax_scaler.transform(X_test)



# Standard Scaler

X_train_ST = standard_scaler.transform(X_train)

X_test_ST = standard_scaler.transform(X_test)
from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.svm import LinearSVR

from sklearn.tree import DecisionTreeRegressor 



models = []

models += [['Ridge', Ridge(alpha = 0.9, solver = "cholesky")]]

models += [['Lasso', Lasso(alpha = 1)]]

models += [['Elastic Net', ElasticNet(alpha = 0.1, l1_ratio = 0.25)]]

models += [['SVM', LinearSVR()]]

models += [['Tree', DecisionTreeRegressor()]]
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error



kfold = KFold(n_splits = 5, random_state = 42)

result_MM =[]

names = []



for name, model in models:

    cv_score = -1 * cross_val_score(model, X_train_MM, y_train, cv = kfold, scoring = 'neg_root_mean_squared_error')

    result_MM +=[cv_score]

    names += [name]

    print('%s: %f (%f)' % (name,cv_score.mean(), cv_score.std()))
result_ST =[]

for name, model in models:

    cv_score = -1 * cross_val_score(model, X_train_ST, y_train, cv = kfold, scoring = 'neg_root_mean_squared_error')

    result_MM +=[cv_score]

    print('%s: %f (%f)' % (name,cv_score.mean(), cv_score.std()))
# training the models

Ridge_model_MM = Ridge(alpha = 0.9, solver = "cholesky").fit(X_train_MM, y_train)

Ridge_model_ST = Ridge(alpha = 0.9, solver = "cholesky").fit(X_train_ST, y_train)



# getting predictions

predictions_MM = Ridge_model_MM.predict(X_test_MM)

predictions_ST = Ridge_model_ST.predict(X_test_ST)
from sklearn.metrics import mean_squared_error

print("Ridge, Min Max: " + str(np.sqrt(mean_squared_error(y_test, predictions_MM))))

print("Ridge, Standard Scaler: " + str(np.sqrt(mean_squared_error(y_test, predictions_ST))))