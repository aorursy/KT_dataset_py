# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sb
pd.set_option('display.max_columns', None)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#final_dateframe = pd.read_csv('/kaggle/input/final-dataframe/final_dataframe.csv',sep=',', thousands=',' )
data_version1 = pd.read_csv('/kaggle/input/final-dataframe-death-may4/final_dataframe_death_may4.csv' ).drop(columns='Unnamed: 0')
data_version1.iloc[:, 0:21].hist(figsize = (24,20))
plt.show()
df = data_version1
final_dateframe = data_version1
## Create a function that append different models performance, parameter, and their data version to a dataframe
master_performance_table = pd.DataFrame(columns = ['Model', 'dataset version','MAE Train',
                                                   'MAE Test','MSE Train','MSE Test','R^2 test','cv_mae','cv_mse','Parameters'])
def master_performance(master_performance_table, Model_name,dataset_version, pred_test,pred_train,para,cv_mae_score,cv_mse_score):
    MAE_train = metrics.mean_absolute_error(y_train, pred_train)
    MAE_test = metrics.mean_absolute_error(y_test, pred_test)
    MSE_train = metrics.mean_squared_error(y_train, pred_train)
    MSE_test = metrics.mean_squared_error(y_test, pred_test)
    R2_test = metrics.r2_score(y_test, pred_test)
    master_performance_table = master_performance_table.append([{'Model':Model_name,'dataset version':dataset_version,'MAE Train':MAE_train,
                                                                 'MAE Test':MAE_test,'MSE Train':MSE_train, 'MSE Test': MSE_test,
                                                                 'R^2 test':R2_test,'cv_mae':cv_mae_score,'cv_mse':cv_mse_score,'Parameters':para}], ignore_index=True)
    return master_performance_table
    

C_mat = data_version1.corr().iloc[0:20, 0:20]
fig = plt.figure(figsize = (10,10))

sb.heatmap(C_mat, vmax = .8, square = True)
plt.show()
data_version1 = data_version1.drop(columns='deaths')
### data version 1 train/test split
X = data_version1.drop(['cases'],axis=1)
y = data_version1['cases']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
RF = RandomForestRegressor(random_state=43)
RF.fit(X_train, y_train)
predictions = RF.predict(X_test)
pred_train = RF.predict(X_train)
cv_mae = cross_val_score(RF,X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mse = cross_val_score(RF,X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mae_score = -np.mean(cv_mae)
cv_mse_score = -np.mean(cv_mse)
cv_mse_score
master_performance_table = master_performance(master_performance_table,'Base Line Random Forest','data version 1',
                                              predictions,pred_train,'NA',cv_mae_score,cv_mse_score)
master_performance_table
#final_dateframe = pd.read_csv('/kaggle/input/final-dataframe/final_dataframe.csv',sep=',', thousands=',' )
data_version1 = pd.read_csv('/kaggle/input/final-dataframe-death-may4/final_dataframe_death_may4.csv' ).drop(columns='Unnamed: 0')
data_version1 = data_version1.drop(columns='cases')

data_version1[data_version1['state_New York']==1]
### data version 1 train/test split
X = data_version1.drop(['deaths'],axis=1)
y = data_version1['deaths']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
RF = RandomForestRegressor(random_state=43)
RF.fit(X_train, y_train)
predictions = RF.predict(X_test)
pred_train = RF.predict(X_train)
cv_mae = cross_val_score(RF,X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mse = cross_val_score(RF,X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mae_score = -np.mean(cv_mae)
cv_mse_score = -np.mean(cv_mse)
master_performance_table = master_performance(master_performance_table,'Base Line Random Forest','data version 1',
                                              predictions,pred_train,'NA',cv_mae_score,cv_mse_score)
master_performance_table
feature_list = data_version1.columns[1:]
# Get numerical feature importances
importances = list(RF.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 300, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)

X_train
train_stats = X_train.describe()
train_stats = train_stats.transpose()
train_stats
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
X_train_norm = norm(X_train)
X_test_norm = norm(X_test)

def build_model():
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(X_train_norm.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model
model = build_model()

model.summary()

#!pip install git+https://github.com/tensorflow/docs
#import tensorflow_docs as tfdocs
EPOCHS = 1000

history = model.fit(
  X_train_norm, y_train,
  epochs=EPOCHS, validation_split = 0.2, verbose=0)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist

# plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

# plotter.plot({'Basic': history}, metric = "mae")
# plt.ylim([0, 10])
# plt.ylabel('MAE [MPG]')

test_predictions = model.predict(X_test_norm).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 2000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

error = test_predictions - y_test
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

#final_dateframe = pd.read_csv('/kaggle/input/final-dataframe/final_dataframe.csv',sep=',', thousands=',' )
df_future= pd.read_csv('/kaggle/input/future/future_afterMay4.csv' ).iloc[:, 1:]
df_future
X_test
df_future
future_norm = norm(df_future)
df_future['death_predicted'] = model.predict(future_norm).flatten()
df_future
plt.scatter(df_future.date_num, df_future.death_predicted)




from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
from keras.layers import Dense, Activation, Flatten
from keras.callbacks import ModelCheckpoint
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]
NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()
NN_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)















rf_random = RandomizedSearchCV(estimator = RF, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X_train, y_train)
str(rf_random.best_params_)
RF_para = RandomForestRegressor(n_estimators = 144,min_samples_split=2, 
                                min_samples_leaf=1,max_features='auto',max_depth=110,
                                bootstrap=True, random_state = 42)
RF_para.fit(X_train, y_train)
predictions_para = RF_para.predict(X_test)
pred_para_train = RF_para.predict(X_train)
cv_mae_pa = cross_val_score(RF_para,X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mse_pa = cross_val_score(RF_para,X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mae_score_pa = -np.mean(cv_mae_pa)
cv_mse_score_pa = -np.mean(cv_mse_pa)
master_performance_table = master_performance(master_performance_table,'Tuning Random Forest','data version 1',
                                              predictions_para,pred_para_train,str(rf_random.best_params_),cv_mae_score_pa,
                                             cv_mse_score_pa)
master_performance_table
X2 = final_dateframe.drop('cases',axis=1)
y2 = final_dateframe['cases']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size = 0.2, random_state = 42)
RF2 = RandomForestRegressor(random_state=45)
RF2.fit(X_train2, y_train2)
predictions3 = RF2.predict(X_test2)
pred_train3 = RF2.predict(X_train2)
cv_mae3 = cross_val_score(RF2,X_train2, y_train2, cv=5, scoring='neg_mean_absolute_error')
cv_mse3 = cross_val_score(RF2,X_train2, y_train2, cv=5, scoring='neg_mean_squared_error')
cv_mae_score3 = -np.mean(cv_mae3)
cv_mse_score3 = -np.mean(cv_mse3)
master_performance_table = master_performance(master_performance_table,'Base Line Random Forest','data version 2',
                                              predictions3,pred_train3,'NA',cv_mae_score2,cv_mse_score2)
master_performance_table
feature_list = final_dateframe.columns[1:]
# Get numerical feature importances
importances = list(RF2.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 300, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
rf_random2 = RandomizedSearchCV(estimator = RF2, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=45, n_jobs = -1)

rf_random2.fit(X_train2, y_train2)
rf_random.best_params_
RF_para2 = RandomForestRegressor(n_estimators = 144,min_samples_split=2, 
                                min_samples_leaf=1,max_features='auto',max_depth=110,
                                bootstrap=True, random_state = 45)
RF_para2.fit(X_train2, y_train2)
predictions_para2 = RF_para2.predict(X_test2)
pred_para_train2 = RF_para2.predict(X_train2)
cv_mae4 = cross_val_score(RF_para2,X_train2, y_train2, cv=5, scoring='neg_mean_absolute_error')
cv_mse4 = cross_val_score(RF_para2,X_train2, y_train2, cv=5, scoring='neg_mean_squared_error')
cv_mae_score4 = -np.mean(cv_mae4)
cv_mse_score4 = -np.mean(cv_mse4)
master_performance_table = master_performance(master_performance_table,'Tuning Random Forest','data version 2',
                                              predictions_para2,pred_para_train2,str(rf_random.best_params_),cv_mae_score4,cv_mse_score4)
master_performance_table
feature_list = final_dateframe.columns[1:]
# Get numerical feature importances
importances = list(RF_para.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
from sklearn import datasets
import xgboost as xgb
XGB = xgb.XGBRegressor()
XGB.fit(X_train,y_train, eval_metric='mae')
preds_test = XGB.predict(X_test)
preds_train = XGB.predict(X_train)
# print('\nMAE train:', metrics.mean_absolute_error(y_train,XGB.predict(X_train)))
# print('\nMAE test:', metrics.mean_absolute_error(y_test,preds))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test,preds))
cv_mae_xg = cross_val_score(XGB,X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mse4_xg = cross_val_score(XGB,X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mae_score4_xg = -np.mean(cv_mae_xg)
cv_mse_score4_xg = -np.mean(cv_mse4_xg)
master_performance_table = master_performance(master_performance_table,'Simple Xgboost','data version 1',
                                              preds_test,preds_train,'NA',cv_mae_score4_xg,cv_mse_score4_xg)
master_performance_table
XGB1 = xgb.XGBRegressor(max_depth = 4)
XGB1.fit(X_train,y_train, eval_metric='mae')
preds_test2 = XGB1.predict(X_test)
preds_train2 = XGB1.predict(X_train)
cv_mae_xg2 = cross_val_score(XGB1,X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mse4_xg2 = cross_val_score(XGB1,X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mae_score4_xg2 = -np.mean(cv_mae_xg2)
cv_mse_score4_xg2 = -np.mean(cv_mse4_xg2)
master_performance_table = master_performance(master_performance_table,'Simple Tuning Xgboost','data version 1',
                                              preds_test2,preds_train2,'max_depth = 4',cv_mae_score4_xg2,cv_mse_score4_xg2)
master_performance_table
## use random search to do large scale parameter tuning
param_grid = {
        'silent': [False],
        'max_depth': [6, 10, 15, 20],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'n_estimators': [100]}
xgb_para = RandomizedSearchCV(XGB, param_grid, n_iter=20,
                              n_jobs=1, verbose=2, cv=3,
                              scoring='neg_mean_absolute_error', random_state=42)
xgb_para.fit(X_train, y_train)
xgb_para.best_params_
XGB_para = xgb.XGBRegressor(subsample=0.6,silent = False,
                           reg_lambda = 1, n_estimators=100,
                           min_child_weight = 1, max_depth = 20,
                           learning_rate = 0.1, gamma = 0.25,
                           colsample_bytree = 0.8, colsample_bylevel = 0.8 )
XGB_para.fit(X_train, y_train)
predictions_para3 = XGB_para.predict(X_test)
pred_para_train3 = XGB_para.predict(X_train)
cv_mae_xg3 = cross_val_score(XGB_para,X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mse4_xg3 = cross_val_score(XGB_para,X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mae_score4_xg3 = -np.mean(cv_mae_xg3)
cv_mse_score4_xg3 = -np.mean(cv_mse4_xg3)
master_performance_table = master_performance(master_performance_table,'Tuning Xgboost','data version 1',
                                              predictions_para3,pred_para_train3,str(xgb_para.best_params_) ,cv_mae_score4_xg3,cv_mse_score4_xg3)
master_performance_table
master_performance_table.to_csv('performance_table.csv')
fig, ax = plt.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(XGB_para, max_num_features=20, height=0.5, ax=ax)
