import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.metrics import mean_absolute_error as mae

from sklearn.metrics import mean_squared_error as mse

from sklearn.model_selection import train_test_split
def test_model(model):

    global X_train, X_test, y_train, y_test

    y_pred_test = model.predict(X_test.reshape(-1,1))

    y_pred_train = model.predict(X_train.reshape(-1,1))

    print('-' * 20)

    print('Root Mean Squared Error on test data:\t', round( np.sqrt(mse(y_test, y_pred_test)), 2))

    print('Root Mean Squared Error on training data:\t', round( np.sqrt(mse(y_train, y_pred_train)), 2))

    print('Mean Absolute on test data:\t', round( mae(y_test, y_pred_test), 2))

    print('Mean Squared Error on training data:\t', round( mae(y_train, y_pred_train), 2))

    print('-' * 20)
sns.set({ 'figure.figsize': (9, 6) })



# Dataset is from: https://www.kaggle.com/mohansacharya/graduate-admissions

df = pd.read_csv('../input/Admission_Predict.csv').dropna()

del df['Serial No.']

sns.heatmap(df.corr().round(2), annot=True)

plt.show()
X = df['CGPA'].values

y = df['Chance of Admit '].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, shuffle=True)
from sklearn.linear_model import LinearRegression

model = LinearRegression(n_jobs=-1, copy_X=True).fit(X_train.reshape(-1,1), y_train)

test_model(model)
plt.scatter(X, y, color='black')

plt.plot(X, model.predict(X.reshape(-1,1)), color='red')

plt.xticks(())

plt.yticks(())

plt.xlabel('Chance')

plt.ylabel('CGPA')

plt.show()
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(learning_rate_init=.3,

                     max_iter=1000,

                     random_state=9,

                     hidden_layer_sizes=(500,))

model.fit(X_train.reshape(-1,1), y_train)

test_model(model)
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(min_samples_split=2,

                             min_weight_fraction_leaf=0.0,

                             min_samples_leaf=1,

                             min_impurity_decrease=0)

model.fit(X_train.reshape(-1,1), y_train)

test_model(model)