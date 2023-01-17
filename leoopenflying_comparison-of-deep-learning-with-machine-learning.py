# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split#split the data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score#R square
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from keras import models
from keras import layers
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasRegressor


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/aqsoldb-a-curated-aqueous-solubility-dataset/curated-solubility-dataset.csv')
dataSet = data.drop(['ID','InChI','InChIKey','SD','Solubility','Ocurrences','SMILES','Name'],axis = 1)
labels = data[['Solubility','Group']]
def splitData(dataSet,labels):
    GroupData_train = []
    GroupLabels_train = []
    GroupData_test = []
    GroupLabels_test = []
    
    for i in range(1,6):
        X_train, X_test, y_train, y_test = eachGroup(dataSet,labels,i)
        GroupData_train.append(X_train)
        GroupLabels_train.append(y_train)
        GroupData_test.append(X_test)
        GroupLabels_test.append(y_test)
    
    trainSet = pd.concat([GroupData_train[0],GroupData_train[1],GroupData_train[2],GroupData_train[3],GroupData_train[4]])
    testSet = pd.concat([GroupData_test[0],GroupData_test[1],GroupData_test[2],GroupData_test[3],GroupData_test[4]])
    yTrainSet = pd.concat([GroupLabels_train[0],GroupLabels_train[1],GroupLabels_train[2],GroupLabels_train[3],GroupLabels_train[4]])
    yTestSet = pd.concat([GroupLabels_test[0],GroupLabels_test[1],GroupLabels_test[2],GroupLabels_test[3],GroupLabels_test[4]])
    
    trainSet = trainSet.drop(['Group'],axis = 1)
    testSet = testSet.drop(['Group'],axis = 1)
    yTrainSet = yTrainSet.drop(['Group'],axis = 1)
    yTestSet = yTestSet.drop(['Group'],axis = 1)
    
    return trainSet,testSet,yTrainSet,yTestSet
    

def eachGroup(dataSet,labels,groupNumber):
    Data = dataSet.loc[dataSet['Group'] == 'G{0}'.format(groupNumber)]
    Labels = labels.loc[labels['Group'] == 'G{0}'.format(groupNumber)]
    X_train, X_test, y_train, y_test = train_test_split(Data, Labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test



X_train, X_test, Y_train, Y_test = splitData(dataSet,labels)
# Y_train = Y_train.values.ravel()
# Y_test = Y_test.values.ravel()
# Y_train = np.array(Y).astype(int)
# Y_test = np.array(Y).astype(int)
Y_train = np.array(Y_train).reshape(-1,1)
Y_test = np.array(Y_test).reshape(-1,1)
scoring = ['precision_macro', 'recall_macro']
std_x = StandardScaler()
x_train = std_x.fit_transform(X_train)
x_test = std_x.transform(X_test)

std_y = StandardScaler()
y_train = std_y.fit_transform(Y_train)  
y_test = std_y.transform(Y_test)
modelSVR = SVR(gamma='auto',kernel = 'linear')
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
kf = KFold(n_splits=10, random_state=42,shuffle=True)
gridSVR=GridSearchCV(modelSVR,param_grid=parameters,
                    cv=kf,scoring='neg_mean_squared_error')
gridSVR.fit(x_train, y_train)




newModelSVR =gridSVR.best_estimator_
newModelSVR.fit(x_train, y_train)
result = newModelSVR.predict(x_test)
y_true = np.array(y_test)
print("MSE:{0}".format(mean_squared_error(y_true, result)))
print("RMSE:{0}".format(mean_squared_error(y_true, result,squared=False)))
print("MAE:{0}".format(mean_absolute_error(y_true, result)))
print("R2 SQUARE:{0}".format(r2_score(y_true, result)))

gridSVR.best_params_
modelLR = Lasso()
alpha_param={'alpha':list(np.logspace(-4,-2,10))}
kf = KFold(n_splits=10, random_state=42,shuffle=True)
gridLR=GridSearchCV(modelLR,param_grid=alpha_param,
                    cv=kf,scoring='neg_mean_squared_error')
gridLR.fit(x_train, y_train)


# newModelLR = Lasso(alpha = bestLR['alpha'])
newModelLR = gridLR.best_estimator_
newModelLR.fit(x_train, y_train)
result = newModelLR.predict(x_test)
y_true = np.array(y_test)
print("MSE:{0}".format(mean_squared_error(y_true, result)))
print("RMSE:{0}".format(mean_squared_error(y_true, result,squared=False)))
print("MAE:{0}".format(mean_absolute_error(y_true, result)))
print("R2 SQUARE:{0}".format(r2_score(y_true, result)))
gridLR.best_params_
modelDT = DecisionTreeRegressor()
tuned_parameters= {'max_depth':[3, 5, 10, 15],'min_samples_leaf':[2,3,5,10],'min_impurity_decrease':[0.1,0.2,0.5]}
kf = KFold(n_splits=10, random_state=42,shuffle=True)
gridDT=GridSearchCV(modelDT,param_grid=tuned_parameters,
                    cv=kf,scoring='neg_mean_squared_error')

gridDT.fit(X_train, Y_train)

newModelDT = gridDT.best_estimator_
newModelDT.fit(X_train, Y_train)
result = newModelDT.predict(X_test)
y_true = np.array(Y_test)
print("MSE:{0}".format(mean_squared_error(y_true, result)))
print("RMSE:{0}".format(mean_squared_error(y_true, result,squared=False)))
print("MAE:{0}".format(mean_absolute_error(y_true, result)))
print("R2 SQUARE:{0}".format(r2_score(y_true, result)))
gridDT.best_params_
modelXGB = XGBRegressor(objective='reg:squarederror')
parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}
kf = KFold(n_splits=10, random_state=42,shuffle=True)
gridXGB=GridSearchCV(modelXGB,param_grid=parameters,
                    cv=kf,scoring='neg_mean_squared_error')
gridXGB.fit(X_train, Y_train)

newModelXGB = gridXGB.best_estimator_
result = newModelXGB.predict(X_test)
y_true = np.array(Y_test)
print("MSE:{0}".format(mean_squared_error(y_true, result)))
print("RMSE:{0}".format(mean_squared_error(y_true, result,squared=False)))
print("MAE:{0}".format(mean_absolute_error(y_true, result)))
print("R2 SQUARE:{0}".format(r2_score(y_true, result)))
gridXGB.best_params_

parameters = {'learning_rate':[0.06,0.07,0.08,0.09,0.1],
               'n_estimators':[100,150,200]}

kf = KFold(n_splits=10, random_state=42,shuffle=True)
gridGB = GridSearchCV(estimator=GradientBoostingRegressor(loss='ls',max_depth=9,max_features=9,
                                                             subsample=0.8,min_samples_leaf=4, min_samples_split=6),n_jobs=-1,
                            param_grid=parameters,scoring='neg_mean_squared_error',iid=False,cv=kf)
gridGB.fit(X_train,Y_train)


newModelGB = gridGB.best_estimator_
newModelGB.fit(X_train, Y_train)
result = newModelGB.predict(X_test)
y_true = np.array(Y_test)
print("MSE:{0}".format(mean_squared_error(y_true, result)))
print("RMSE:{0}".format(mean_squared_error(y_true, result,squared=False)))
print("MAE:{0}".format(mean_absolute_error(y_true, result)))
print("R2 SQUARE:{0}".format(r2_score(y_true, result)))
gridGB.best_params_
modelRF = RandomForestRegressor(max_depth=2, random_state=0)
tuned_parameters= {'max_depth':[3, 5, 10, 15],'min_samples_leaf':[2,3,5,10],'min_impurity_decrease':[0.1,0.2,0.5]}
kf = KFold(n_splits=10, random_state=42,shuffle=True)
gridRF=GridSearchCV(modelRF,param_grid=tuned_parameters,
                    cv=kf,scoring='neg_mean_squared_error')
gridRF.fit(X_train,Y_train)
newModelRF = gridRF.best_estimator_
newModelRF.fit(X_train, Y_train)
result = newModelRF.predict(X_test)
y_true = np.array(Y_test)
print("MSE:{0}".format(mean_squared_error(y_true, result)))
print("RMSE:{0}".format(mean_squared_error(y_true, result,squared=False)))
print("MAE:{0}".format(mean_absolute_error(y_true, result)))
print("R2 SQUARE:{0}".format(r2_score(y_true, result)))
gridRF.best_params_
def build_model():

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
# k = 4
# num_val_samples = len(x_train) // k
# num_epochs = 100
# all_scores = []
# for i in range(k):
#     print('processing fold #', i)
#     '''
#     Prepares the validation data: data from partition #k
#     '''
#     val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
#     val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]
    
#     '''
#     Prepares the training data: data from all other partitions
#     '''
#     partial_x_train = np.concatenate(
#         [x_train[:i * num_val_samples],
#          x_train[(i + 1) * num_val_samples:]], axis=0)
#     partial_y_train = np.concatenate(
#         [y_train[:i * num_val_samples],
#          y_train[(i + 1) * num_val_samples:]], axis=0)
    
#     '''
#     Builds the Keras model (already compiled)
#     '''
#     model = build_model()
#     '''
#     Trains the model (in silent mode, verbose = 0)
#     '''
    
    
#     model.fit(partial_x_train, partial_y_train, epochs=num_epochs,
#               batch_size=1, verbose=0)
#     '''
#     Evaluates the model on the validation data
#     '''
#     val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
#     all_scores.append(val_mae)
model_grid = KerasRegressor(build_fn=build_model)


parameters = {
              'epochs':[50, 100, 150,200],
              'batch_size':[5, 10, 20,30]
             }
grid = GridSearchCV(estimator=model_grid, param_grid=parameters)
grid.fit(x_train, y_train)
grid.best_params_
def build_model(optimizer='adam'):

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['mse'])
    return model
activation = ['softmax', 'softplus', 'softsign', 'relu', 
              'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid=dict(activation=activation)
grid.best_params_
from keras.layers import Dropout
from keras.constraints import maxnorm
def build_model(dropout_rate=0.0, weight_constraint=0):

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['mse'])
    return model
model_grid = KerasRegressor(build_fn=build_model,epochs=100, batch_size=10, verbose=0)


weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = dict(weight_constraint=weight_constraint, dropout_rate=dropout_rate)
grid = GridSearchCV(estimator=model_grid, param_grid=param_grid, n_jobs=1)
grid.fit(x_train, y_train)
