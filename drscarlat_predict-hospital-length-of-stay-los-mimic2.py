# IMPORT MODULES
# TURN ON the GPU !

import os
from operator import itemgetter    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder, LabelBinarizer
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, cross_val_predict, StratifiedKFold, train_test_split, learning_curve, ShuffleSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor

import tensorflow as tf

from keras import models, regularizers, layers, optimizers, losses, metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

print(os.getcwd())
print("Modules imported \n")
import os
print(os.listdir("../input"))
# Load MIMIC2 data 

data = pd.read_csv('../input/mldata.csv')
print("With id", data.shape)
#print(data.head())

data_full = data.drop('hadm_id', 1)

print("No id",data_full.shape)
#print(data_full.head())

# Label = LOS
y = data_full['los_days']
X = data_full.drop('los_days', 1)
X = X.drop('expired_icu', 1)

print("y - Labels", y.shape)
print("X - No Label No id No expired", X.shape)
print(X.columns)
print(data_full.shape)
data_full.info()
data_full.describe()
data_full.head(10)
data_full.hist(bins=50, figsize=(20,15))
plt.show()
age_histogram = data_full.hist(column='age', bins=20, range=[0, 100])
for ax in age_histogram.flatten():
    ax.set_xlabel("Age")
    ax.set_ylabel("Num. of Patients")
    
age_LOS = data_full.hist(column='los_days', bins=20, range=[0, 100])
for ax in age_LOS.flatten():
    ax.set_xlabel("LOS")
    ax.set_ylabel("Num. of Patients")
data_full.groupby('insurance').size().plot.bar()
plt.show()
data_full.groupby('admission_type').size().plot.bar()
plt.show()
data_full.groupby('admission_source').size().plot.bar()
plt.show()
# Pearson linear correlation

corr_matrix = data_full.corr()
corr_matrix["los_days"].sort_values(ascending=False)
# Check that all X columns have no missing values
X.info()
X.describe()
from pandas.plotting import scatter_matrix

scatter_matrix(data_full.loc[:, data_full.columns] ,  figsize  = [15, 15], diagonal = "kde")
plt.show()
#data_full.plot(kind="scatter", x="sofa_max", xlim=(0,25), y="LOS", alpha=0.1, ylim=(0,50))
data_full.plot(kind="scatter", x="age", xlim=(0,80), y="los_days", alpha=0.1, ylim=(0,50))
# MAP Text to Numerical Data
# Use one-hot-encoding to convert categorical features to numerical

print(X.shape)
categorical_columns = ['expired_icu', 
                       'gender',
                      'marital_status',
                      'ethnicity',
                      'admission_type',
                      'admission_source',
                      'insurance',
                      'religion',
                      'DiagnosisFirst',
                      'ProcedureFirst'
                      ]

for col in categorical_columns:
    #if the original column is present replace it with a one-hot
    if col in X.columns:
        one_hot_encoded = pd.get_dummies(X[col])
        X = X.drop(col, axis=1)
        X = X.join(one_hot_encoded, lsuffix='_left', rsuffix='_right')
        
print(X.shape)
print(X.columns)
print(X['VENTRICULOSTOMY          '])

print(data_full.shape)
print(X.shape)

#XnotNorm = np.array(X.copy())
XnotNorm = X.copy()
print('XnotNorm ', XnotNorm.shape)

yFI = data_full.los_days
ynotNorm = yFI.copy()
print('ynotNorm ', ynotNorm.shape)
# Normalize X

x = XnotNorm.values #returns a numpy array
scaler = preprocessing.StandardScaler()
x_scaled = scaler.fit_transform(x)
XNorm = pd.DataFrame(x_scaled, columns=XnotNorm.columns)
print(XNorm)
# Normalize y

y = ynotNorm.values #returns a numpy array
y = y.reshape(-1, 1)
y_scaled = scaler.fit_transform(y)
ynorm=pd.DataFrame(y_scaled)
print(ynorm)
# FEATURE IMPORTANCE Data NOT normalized using Lasso model - NOT the best one ...

trainFinalFI = XnotNorm
yFinalFI = ynotNorm

lasso=Lasso(alpha=0.001, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
lasso.fit(trainFinalFI,yFinalFI)

FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=trainFinalFI.columns)

# Focus on those with 0 importance
#print(FI_lasso.sort_values("Feature Importance",ascending=False).to_string())
#print("_"*80)
FI_lasso[FI_lasso["Feature Importance"] >40].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))
plt.xticks(rotation=90)
FI_lasso[FI_lasso["Feature Importance"] <-10].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))
plt.xticks(rotation=90)
plt.show()
# CROSS VALIDATION

def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse
# Lin reg ALL models HYPERPARAMS NOT optimized

models = [RandomForestRegressor(),GradientBoostingRegressor(), ExtraTreesRegressor()]
names = ["RandomForestRegressor", "GradientBoostingRegressor", "ExtraTreesRegressor"]
# Run the models and compare

ModScores = {}

for name, model in zip(names, models):
    score = rmse_cv(model, XNorm, ynorm)
    ModScores[name] = score.mean()
    print("{}: {:.2f}".format(name,score.mean()))

print("_"*100)
for key, value in sorted(ModScores.items(), key = itemgetter(1), reverse = False):
    print(key, round(value,3))
# Optimize hyper params for one model

model = GradientBoostingRegressor()

param_grid = [{},]

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(XNorm, ynorm)

print(grid_search.best_estimator_)
model = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100, presort='auto', random_state=None,
             subsample=1.0, verbose=0, warm_start=False)
# FEATURE IMPORTANCE - NORMALIZED - GBR model
# NOTE GBR - has NO Negative feature importance

trainFinalFI = XNorm
yFinalFI = ynorm

GBR = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100, presort='auto', random_state=None,
             subsample=1.0, verbose=0, warm_start=False)

GBR.fit(trainFinalFI,yFinalFI)

FI_GBR = pd.DataFrame({"Feature Importance":GBR.feature_importances_,}, index=trainFinalFI.columns)
FI_GBR[FI_GBR["Feature Importance"] > 0.009469].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))
plt.xticks(rotation=90)
plt.xticks(rotation=90)
plt.show()
# List of important features for GBR
FI_GBR = pd.DataFrame({"Feature Importance":GBR.feature_importances_,}, index=trainFinalFI.columns)
FI_GBR=FI_GBR.sort_values('Feature Importance', ascending = False)
print(FI_GBR[FI_GBR["Feature Importance"] > 0.0025])
#print(FI_GBR)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Error")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = 1-np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = 1-np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
# LEARNING CURVES Train / Validation
# Note - ynotNorm !!


title = "Learning Curves "
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plot_learning_curve(model, title, XNorm, ynotNorm, cv=cv, n_jobs=5)
#plot_learning_curve(model, title, XNorm, y, ylim=(0.01, 0.99), cv=cv, n_jobs=4)
# Split into Train & Test

#   NOTE - For ed purposes ynotNorm was USED !!!

X_train, X_test, y_train, y_test = train_test_split(XNorm, ynotNorm, test_size=0.2, random_state=42)
print ('X_train: ', X_train.shape)
print ('X_test: ', X_test.shape)
print ('y_train: ', y_train.shape)
print ('y_test: ', y_test.shape)
# Model FINAL fit and evaluation on test

model.fit(X_train, y_train)

final_predictions = model.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) 
print("rmse on test ", round(final_rmse, 4))

final_mae = mean_absolute_error(y_test, final_predictions)
print("mae on test ", round(final_mae, 4))
# PLOT True vs Predicted

xChart = [np.array(y_test)]
yChart = [np.array(final_predictions)]

plt.scatter(xChart,yChart, alpha=0.3)
plt.xlim(0,60)
plt.ylim(0,60)
plt.plot( [0,60],[0,60], 'b')
plt.show()

plt.scatter(xChart,yChart, alpha=0.3)
plt.xlim(0,30)
plt.ylim(0,30)
plt.plot( [0,30],[0,30], 'b')
plt.show()
# Transfer data to NN format

x_val = X_test
partial_x_train = X_train
y_val = y_test
partial_y_train = y_train

print("partial_x_train ", partial_x_train.shape)
print("partial_y_train ", partial_y_train.shape)

print("x_val ", x_val.shape)
print("y_val ", y_val.shape)
# NN MODEL
from keras import models
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.001),
                       input_shape=(partial_x_train.shape[1],)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.001)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l1(0.001)))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
print("model compiled")
history = model.fit(partial_x_train, partial_y_train,
                    validation_data=(x_val, y_val), 
                    verbose=1,
                   epochs=100)
acc = history.history['mean_absolute_error']
val_acc = history.history['val_mean_absolute_error']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training error')
plt.plot(epochs, val_acc, 'r', label='Validation error')
plt.title('Training and validation ERROR')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation LOSS')
plt.legend()
plt.show()
# Model fit and evaluation on test
# Set the num of Epochs and Batch Size according to learning curves

model.fit(partial_x_train, partial_y_train, epochs=100)
test_mse_score, test_mae_score = model.evaluate(x_val, y_val)
print("test_mae_score on test ", test_mae_score)
print("test_mse_score on test ", test_mse_score)
# PREDICT
final_predictions = model.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) 
print("rmse on test ", round(final_rmse, 4))

final_mae = mean_absolute_error(y_test, final_predictions)
print("mae on test ", round(final_mae, 4))