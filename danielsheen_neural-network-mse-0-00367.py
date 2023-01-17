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
# All of our given features

columns = df.columns.values

columns
col_fix = []

for col in columns:

    col_fix += [col.strip()]

df.columns = col_fix

df.columns.values
scatter_matrix(df)

plt.show()
df.plot(x = 'Research', y = 'Chance of Admit', kind = 'scatter')
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif_calc(features):

    vif = pd.DataFrame()

    vif['features'] = features.columns

    vif['VIF'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]

    return vif
X_temp = df.copy()

X_temp.insert(0,'Intercept', 1)

X_temp.drop(columns = ['Chance of Admit'], inplace = True)

vif = vif_calc(X_temp)

vif
temp = df[['GRE Score', 'TOEFL Score', 'CGPA', 'Chance of Admit']]

temp.corr(method = 'pearson')
df.drop(columns = ['TOEFL Score'], inplace = True)

df.head()
X_temp = df.copy()

X_temp.insert(0,'Intercept', 1)

X_temp.drop(columns = ['Chance of Admit'], inplace = True)

vif = vif_calc(X_temp)

vif
df['GRE Score'] = df['GRE Score'].divide(340)

df['CGPA'] = df['CGPA'].divide(10)

df.head()
df['Academics'] = df['GRE Score'] * df['CGPA']

df.head()
df.drop(columns = ['GRE Score', 'CGPA'], inplace = True)

df.head()
X_temp = df.copy()

X_temp.insert(0,'Intercept', 1)

X_temp.drop(columns = ['Chance of Admit'], inplace = True)

vif = vif_calc(X_temp)

vif
from sklearn.model_selection import train_test_split

X = df.copy()

X.drop(columns = ['Chance of Admit'], inplace = True)

y = df['Chance of Admit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# scaling the features

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from keras.models import Sequential

from keras.layers import Dense

from keras import regularizers

def model_create():

    model = Sequential()

    model.add(Dense(20, kernel_initializer='normal', activation = 'relu', input_dim = 5))

    model.add(Dense(20, activation = 'relu'))

    model.add(Dense(1, kernel_initializer='normal', activation = 'sigmoid'))

    model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mae'])

    return model
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold



estimator = KerasRegressor(build_fn = model_create, epochs = 500, batch_size = 500, verbose=0)

kfold = KFold(n_splits=15)

results = cross_val_score(estimator, X_train, y_train, cv=kfold)

print("Baseline: %f (%f) MSE" % (results.mean(), results.std()))
estimator.fit(X_train, y_train)

predictions = estimator.predict(X_test)
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

print("Final mean squared error: %f      R^2 value: %f" %(mean_squared_error(y_test, predictions), r2_score(y_test, predictions)))