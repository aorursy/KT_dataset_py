import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))





from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import MinMaxScaler







# Metrics

from sklearn import metrics

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error



# Model Selection

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import RidgeCV, Lasso, BayesianRidge, Ridge, ElasticNet
train = pd.read_csv('../input/comprehensive-exploration-cleaning/train_.csv')

holdout = pd.read_csv('../input/comprehensive-exploration-cleaning/test_.csv')
train_Id = train['Id'] ; holdout_Id = holdout['Id']

train_data = train.drop('Id', axis=1) ; holdout_data = holdout.drop('Id', axis=1)
X = train_data.drop('SalePrice', axis=1)

y = train_data['SalePrice']

log_y = np.log(y)



X_train,  X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=101)
# Step 1: Fit Lasso

lasso = Lasso().fit(X_train, y_train)

# Step 2: Build parameters dictionary

parameters = {'alpha':np.logspace(-10, 6, 20)}





# Fit Lasso Model with Gridsearch

#====================================================================

# Step 1: Run Gridsearch

clf = GridSearchCV(lasso, parameters, cv=5, verbose=0)

# Step 2: Fit best model from Gridsearch 

best_model = clf.fit(X_train, y_train)

# Step 3: Predict

y_pred = clf.predict(X_test)





# Step 4: Calculate metrics

rsquare_lasso = round(r2_score(y_test, y_pred),4)

rmse_lasso = round(np.sqrt(metrics.mean_squared_error(np.log(y_test), np.log(y_pred))),4)

print('Lasso - RMSE : ', rmse_lasso ,'\nLasso - rsquare : ',rsquare_lasso)



# Lasso + RobustScaler Model with Gridsearch

#====================================================================

# Step 1: Run Gridsearch

clf = make_pipeline(RobustScaler(),GridSearchCV(lasso, parameters, cv=5, verbose=0))

# Step 2: Fit best model from Gridsearch 

best_model = clf.fit(X_train, y_train)

# Step 3: Predict

y_pred = clf.predict(X_test)





# Step 4: Calculate metrics

rsquare_lasso_RobustScaler = round(r2_score(y_test, y_pred),4)

rmse_lasso_RobustScaler = round(np.sqrt(metrics.mean_squared_error(np.log(y_test), np.log(y_pred))),4)

print('Lasso + RobustScaler - RMSE : ', rmse_lasso_RobustScaler ,'\nLasso + RobustScaler - rsquare : ',rsquare_lasso_RobustScaler)
n_samples, n_features = X_train.shape[0], X_train.shape[1]



# Step 1: Fit Lasso

ridge = KernelRidge().fit(X_train, y_train)

# Step 2: Build parameters dictionary

parameters = {'alpha':np.logspace(-10, 6, 20)}





# Kernel Ridge Regression Model with Gridsearch

#====================================================================

# Step 1: Run Gridsearch

clf = GridSearchCV(ridge, parameters, cv=5, verbose=0)

# Step 2: Fit Lasso again with the best parameters from Gridsearch 

best_model = clf.fit(X_train, y_train)

# Step 3: Predict

y_pred = clf.predict(X_test)



# Step 4: Calculate metrics

rsquare_ridge = round(r2_score(y_test, y_pred),4)

rmse_ridge = round(np.sqrt(metrics.mean_squared_error(np.log(y_test), np.log(y_pred))),4)

print('Kernel Ridge Regression - RMSE : ', rmse_ridge,'\nKernel Ridge Regression - rsquare : ',rsquare_ridge)
# Step 1: Fit Lasso

eNet = ElasticNet().fit(X_train, y_train)

# Step 2: Build parameters dictionary

parameters = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],"l1_ratio": np.arange(0.0, 1.0, 0.1)}





# elastic-net Regression Model with Gridsearch

#====================================================================

# Step 1: Run Gridsearch

clf = GridSearchCV(eNet, parameters, cv=5, verbose=0)

# Step 2: Fit Lasso again with the best parameters from Gridsearch 

best_model = clf.fit(X_train, y_train)

# Step 3: Predict

y_pred = clf.predict(X_test)



# Step 4: Calculate metrics

rsquare_enet = round(r2_score(y_test, y_pred),4)

rmse_enet = round(np.sqrt(metrics.mean_squared_error(np.log(y_test), np.log(y_pred))),4)

print('elastic-net - RMSE : ', rmse_enet ,'\nelastic-net - rsquare :',rsquare_enet)
'''



# Step 1. Scale the features and convert data into tensors

scaler = MinMaxScaler()

scaler.fit(X_train)



X_train = pd.DataFrame(data = scaler.transform(X_train),columns = X_train.columns, index = X_train.index)

X_test = pd.DataFrame(data = scaler.transform(X_test),  columns = X_test.columns, index = X_test.index)





# Step 2. Convert data to tensors

def make_features_cols():

    input_cols = [tf.feature_column.numeric_column(k) for k in X_train.columns]

    return input_cols

feature_columns = make_features_cols()  





# Step 3. Build input function, model and train the model

#-------------------------------------------------------------------------

# Build input function

input_func = tf.estimator.inputs.pandas_input_fn(x = X_train, y=y_train,batch_size=10,num_epochs = 1000,shuffle = True)

# Build model

model = tf.estimator.DNNRegressor(hidden_units=[100,100,100], feature_columns=feature_columns)

# Train the model

model.train(input_fn=input_func, steps = 10000)





# Step 4: Predict

predict_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test, batch_size=10, num_epochs=1,shuffle = False)

pred_gen = model.predict(predict_input_func)

predictions = list(pred_gen)



y_pred = []

for pred in predictions:

    y_pred.append(pred['predictions'])







# Step 5: Calculate metrics

rsquare_DNN_Regressor = round(r2_score(y_test, y_pred),4)

rmse_DNN_Regressor = round(np.sqrt(metrics.mean_squared_error(log(y_test), log(y_pred))),4)

print('DNN Regressor - RMSE : ', rmse_DNN_Regressor ,'\nDNN Regressor - rsquare : ',rsquare_DNN_Regressor)

'''



print('Lasso            RMSE : ', rmse_lasso ,'  rsquare : ',rsquare_lasso)

print('Lasso+RS         RMSE : ', rmse_lasso_RobustScaler , '  rsquare : ',rsquare_lasso_RobustScaler)

print('Kernel Ridge     RMSE : ', rmse_ridge,'   rsquare : ',rsquare_ridge)

print('Enet             RMSE : ', rmse_enet ,'  rsquare : ',rsquare_enet)

#print('DNN-Regressor    RMSE : ', rmse_DNN_Regressor ,'  rsquare : ',rsquare_DNN_Regressor)

X_holdout = pd.DataFrame(data = holdout.drop('Id', axis=1),columns = X_train.columns, index = holdout.index)





# Step 1: Fit ElasticNet

eNet = ElasticNet().fit(X_train, y_train)

# Step 2: Build parameters dictionary

parameters = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],"l1_ratio": np.arange(0.0, 1.0, 0.1)}





# elastic-net Regression Model with Gridsearch

#====================================================================



# Step 1: Run Gridsearch

clf = GridSearchCV(eNet, parameters, cv=5, verbose=0)

# Step 2: Fit Lasso again with the best parameters from Gridsearch 

best_model = clf.fit(X_train, y_train)

# Step 3: Predict

y_pred_holdout = clf.predict(X_holdout)





submission = pd.DataFrame({"Id": holdout["Id"],"SalePrice": y_pred_holdout})

submission.loc[submission['SalePrice'] <= 0, 'SalePrice'] = 0

fileName = "submission.csv"

submission.to_csv(fileName, index=False)




