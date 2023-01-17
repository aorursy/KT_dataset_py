# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load data from csv
mssm = pd.read_csv("/kaggle/input/predicting-finetuning-in-mssm/FineTuningMSSM100.csv")
# Data copies
data = mssm
data.head()
data.isnull().sum()
# Reassign signmu (categorical)
data["signmu"] = data["signmu"].replace({-1:0})
# Remove spurious data where FT = 0
data = data[data['Fine-Tuning'].between(0,10000,inclusive=False)]
data.describe()
# Assign training, target sets
train_X = data.drop(['Fine-Tuning'],axis=1)
train_Y = data['Fine-Tuning']
# Remove features
#lst = ['M1','M2','M3','MHu2','MHd2','signmu','Mq233','Mu233','Tu33','tanb']
#train_X = train_X[lst]
# Plot of fine-tuning
plt.figure(figsize=(15,15))
sns.distplot(train_Y)
plt.show()
# Correlation matrix
plt.figure(figsize=(15,15))
cor = train_X.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train_X,train_Y,test_size=0.3,random_state=0)
# Standardized data set
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

# Stochastic gradient descent
sgd = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', SGDRegressor(max_iter=1000))
])
sgd.fit(X_train,y_train)

# SGD metrics
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
y_pred = sgd.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("Root Mean Squared Error : ", rmse)
print("R^2 : ",r2)
# Extreme gradient boosting

import xgboost as xgb
xgbr = xgb.XGBRegressor(n_estimators=1000,
                        learning_rate=0.01,
                        max_depth=6,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        gamma=1,
                        random_state=0,
                        verbosity=0)
xgbr.fit(X_train, y_train)

# XGBR metrics
y_pred = xgbr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("Root Mean Squared Error : ", rmse)
print("R^2 : ",r2)
# Random Forests

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)

# RF metrics
y_pred = regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("Root Mean Squared Error : ", rmse)
print("R^2 : ",r2)
# NN design

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

def build_model():
    n_nodes = 2*len(train_X.columns)
    n_hidden = 5
    model = keras.Sequential()
    model.add(layers.Dense(n_nodes, activation='relu', input_shape=[len(train_X.keys())]))
    for i in range(n_hidden):
        model.add(layers.Dense(n_nodes, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(loss='mse',
                optimizer=Adam(learning_rate=0.01),
                metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.losses.MeanAbsolutePercentageError()])
    return model
model = build_model()
model.summary()
# Model Fitting
model.fit(
  X_train_scaled, y_train,
  epochs=100,
  validation_split = 0.2,
  verbose=1)
mse, rmse, mape = model.evaluate(X_test_scaled, y_test)

## NN metrics
y_pred = model.predict(X_test_scaled)
mape = tf.keras.losses.MeanAbsolutePercentageError()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("Root Mean Squared Error : ", rmse)
print("Mean Absolute Percentage Error : ", mape(y_test, y_pred).numpy())
print("R^2 : ",r2)
final_data = pd.DataFrame({'Fine-Tuning':y_test,
                          'Predicted Fine-Tuning (SGD)':sgd.predict(X_test),
                          'Predicted Fine-Tuning (XGBoost)':xgbr.predict(X_test),
                          'Predicted Fine-Tuning (Random Forest)':regressor.predict(X_test),
                          'Predicted Fine-Tuning (NN)':model.predict(X_test_scaled).ravel()
                          })
final_data = final_data.sort_values(by='Fine-Tuning').reset_index()
plt.figure(figsize=(10,10))
sns.lineplot(final_data['Fine-Tuning'],((final_data['Fine-Tuning']-final_data['Predicted Fine-Tuning (XGBoost)'])/final_data['Fine-Tuning']).abs()*100)
plt.axis([0, 250, 0, 300])
plt.xlabel('Actual Fine-Tuning')
plt.ylabel('MAPE')
plt.show()