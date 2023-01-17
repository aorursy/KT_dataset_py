import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import model_selection

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from xgboost import XGBRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.model_selection import GridSearchCV



%matplotlib inline

import os

print(os.listdir("../input"))
file = '../input/airfoil.csv'

airfoil_data = pd.read_csv(file, header=None)

airfoil_data.columns = ['frequency', 'angle_of_attack', 'chord', 'velocity', 'suc_displacement', 'sound_pressure']

sns.pairplot(data=airfoil_data)
y = airfoil_data.sound_pressure

features = ['frequency', 'angle_of_attack', 'chord', 'velocity', 'suc_displacement']

X = airfoil_data[features]

y = y.to_frame()
from sklearn import preprocessing

scaler_x = preprocessing.MinMaxScaler()

X_scaled = scaler_x.fit_transform(X)

scaler_y = preprocessing.MinMaxScaler()

y_scaled = scaler_y.fit_transform(np.array(y).reshape(-1,1))

y_scaled = y_scaled.reshape(-1)

train_X, val_X, train_y, val_y = train_test_split(X_scaled, y_scaled, random_state=1)

val_y = scaler_y.inverse_transform(np.array(val_y).reshape(-1,1))
def compare_models(a,b,c,d):

    print('\nCompare Multiple Classifiers:')

    print('\nK-Fold Cross-Validation Accuracy:\n')

    models = []

    models.append(('LR', LinearRegression()))

    models.append(('RF', RandomForestRegressor(n_estimators=10)))

    models.append(('DT', DecisionTreeRegressor()))

    models.append(('XGB', XGBRegressor()))

    models.append(('SVR RBF', SVR(gamma='auto')))

    models.append(('SVR Lin', SVR(kernel='linear', gamma='auto')))

    resultsAccuracy = []

    names = []

    for name, model in models:

        model.fit(a,b)

        kfold = model_selection.KFold(n_splits=10, random_state=7)

        accuracy_results = model_selection.cross_val_score(model, a,b, cv=kfold)

        resultsAccuracy.append(accuracy_results)

        names.append(name)

        accuracyMessage = "%s: %f (%f)" % (name, accuracy_results.mean(), accuracy_results.std())

        print(accuracyMessage)

        

    # boxplot algorithm comparison

    fig = plt.figure()

    fig.suptitle('Algorithm Comparison: Accuracy')

    ax = fig.add_subplot(111)

    plt.boxplot(resultsAccuracy)

    ax.set_xticklabels(names)

    ax.set_ylabel('Cross-Validation: Accuracy Score')

    plt.show()

    return



compare_models(train_X, train_y, val_X, val_y)
rf_model = RandomForestRegressor(n_estimators=100, random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_predictions = scaler_y.inverse_transform(np.array(rf_val_predictions).reshape(-1,1))



rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE when using Random Forest {}".format(rf_val_mae))

val_r2 = r2_score(val_y, rf_val_predictions)

print("Validation R^2: {}".format(val_r2))

kfold = model_selection.KFold(n_splits=10, random_state=7)

accuracy_results = model_selection.cross_val_score(rf_model, train_X,train_y, cv=kfold)

accuracyMessage = "%s: %f (%f)" % ('RF Cross Validation', accuracy_results.mean(), accuracy_results.std())

print(accuracyMessage)
fig, ax = plt.subplots()

ax.scatter(val_y, rf_val_predictions)

ax.plot([val_y.min(), val_y.max()], [rf_val_predictions.min(), rf_val_predictions.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()
xg_model = xgb.XGBRegressor(learning_rate=0.2, max_depth=6, n_estimators=200, random_state=1)

xg_model.fit(train_X, train_y)

xg_preds = xg_model.predict(val_X)

xg_preds = scaler_y.inverse_transform(np.array(xg_preds).reshape(-1,1))

xg_mae = mean_squared_error(xg_preds, val_y)

print("Validation MSE when using Gradient Boost {}".format(xg_mae))

val_r2 = r2_score(val_y, xg_preds)

print("Validation R^2: {}".format(val_r2))



kfold = model_selection.KFold(n_splits=10, random_state=7)

accuracy_results = model_selection.cross_val_score(xg_model, train_X,train_y, cv=kfold)

accuracyMessage = "%s: %f (%f)" % ('XGB Cross Validation', accuracy_results.mean(), accuracy_results.std())

print(accuracyMessage)
fig, ax = plt.subplots()

ax.scatter(val_y, xg_preds)

ax.plot([val_y.min(), val_y.max()], [xg_preds.min(), xg_preds.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()