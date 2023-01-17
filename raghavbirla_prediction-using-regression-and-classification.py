import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
#from sklearn.neural_network import MLPRegressor
from math import sqrt
data = pd.read_csv('../input/winequality-red.csv')
data.head()
data.shape
features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
            'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
target = ['quality']
data.isnull().any()
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=200)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_prediction = regressor.predict(X_test)
print(y_prediction[:5])
print('*'*40)
print(y_test[:5])
y_test.describe()
RMSE = sqrt(mean_squared_error(y_true=y_test, y_pred=y_prediction))
print(RMSE)
regressor = DecisionTreeRegressor(max_depth=50)
regressor.fit(X_train, y_train)
y_prediction = regressor.predict(X_test)
y_prediction[:5]
y_test[:5]
RMSE = sqrt(mean_squared_error(y_true=y_test, y_pred=y_prediction))
print(RMSE)
from sklearn.tree import DecisionTreeClassifier
data_classifier = data.copy()
data_classifier.head()
data_classifier['quality'].dtype
data_classifier['quality_label'] = (data_classifier['quality'] > 6.5)*1
data_classifier['quality_label']
features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
            'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
target_classifier = ['quality_label']
X = data_classifier[features]
y = data_classifier[target_classifier]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)
wine_quality_classifier = DecisionTreeClassifier(max_leaf_nodes=20, random_state=0)
wine_quality_classifier.fit(X_train, y_train)
prediction = wine_quality_classifier.predict(X_test)
print(prediction[:5])
print('*'*10)
print(y_test['quality_label'][:5])
accuracy_score(y_true=y_test, y_pred=prediction)
from sklearn.linear_model import LogisticRegression
data_classifier.head()
features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
            'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
target_classifier = ['quality_label']
X = data_classifier[features]
y = data_classifier[target_classifier]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
prediction = logistic_regression.predict(X_test)
print(prediction[:5])
print(y_test[:5])
accuracy_score(y_true=y_test, y_pred=prediction)
