# Need to install autokeras

!pip install autokeras

!pip install git+https://github.com/keras-team/keras-tuner.git@1.0.2rc1
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

test_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
print("Shape of training data",train_data.shape)

print("Shape of testing data",test_data.shape)
feature_cols = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',

       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',

       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',

       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',

       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',

       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',

       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',

       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',

       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',

       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',

       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',

       'SaleCondition']



target_cols = ['SalePrice']



X_train = train_data[feature_cols]

y_train = train_data[target_cols]

X_test = test_data[feature_cols]

# X_all = pd.concat([X_train,X_test],axis=1)
# Get list of categorical variables

s = (X_train.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)



s = (X_train.dtypes != 'object')

numerical_cols = list(s[s].index)



print("Non-Categorical variables:")

print(numerical_cols)
# Get names of columns with missing values

cols_with_missing = [col for col in train_data.columns

                     if train_data[col].isnull().any()]

# cols_with_missing



train_data_categorical_cols_imputed = train_data[object_cols].apply(lambda x: x.fillna(x.value_counts().index[0]))

test_data_categorical_cols_imputed = test_data[object_cols].apply(lambda x: x.fillna(x.value_counts().index[0]))



train_data_non_categorical_cols_imputed =train_data[numerical_cols].fillna(train_data[numerical_cols].mean())

test_data_non_categorical_cols_imputed =test_data[numerical_cols].fillna(test_data[numerical_cols].mean())

X_train = pd.concat([train_data_categorical_cols_imputed, train_data_non_categorical_cols_imputed], axis=1)

X_test = pd.concat([test_data_categorical_cols_imputed, test_data_non_categorical_cols_imputed], axis=1)
# This step for using prediction in xgboost

print(X_train.shape)

print(X_test.shape)

frames = [X_train, X_test]



X_all = pd.concat(frames)

print(X_all.shape)

print(y_train.shape)
# train_df_categorical_cols_imputed = train_df[categorical_columns].apply(lambda x: x.fillna(x.value_counts().index[0]))
def display_scores(scores):

    print("Scores: ",scores)

    print("Mean:",scores.mean())

    print("Standard deviation:",scores.std())
from sklearn.preprocessing import OneHotEncoder



# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))

OH_cols_test = pd.DataFrame(OH_encoder.fit_transform(X_test[object_cols]))

# OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

OH_cols_X_all = pd.DataFrame(OH_encoder.fit_transform(X_all[object_cols]))

# X_all



# One-hot encoding removed index; put it back

OH_cols_train.index = X_train.index

OH_cols_test.index = X_test.index

OH_cols_X_all.index = X_all.index





# OH_cols_valid.index = X_valid.index



# Remove categorical columns (will replace with one-hot encoding)

num_X_train = X_train.drop(object_cols, axis=1)

num_X_test = X_test.drop(object_cols, axis=1)

num_X_all = X_all.drop(object_cols, axis=1)

# num_X_valid = X_valid.drop(object_cols, axis=1)



# Add one-hot encoded columns to numerical features

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)

OH_X_all = pd.concat([num_X_all,OH_cols_X_all],axis=1)

# OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)



# print("MAE from Approach 3 (One-Hot Encoding):") 

# print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
training_features = OH_X_train.copy()

training_labels = y_train.copy()

test_features = OH_X_test.copy()

test_xgb = OH_X_all[1460:]

test_keras = OH_X_all[1460:]

test_bagging = OH_X_all[1460:]

print(test_keras.shape)

print(training_features.shape)

print(test_xgb.shape)

print(training_labels.shape)



# (1460, 79)

# (1459, 79)

# (2919, 79)
from sklearn.model_selection import cross_val_score

from sklearn.metrics import balanced_accuracy_score,make_scorer,mean_squared_error

rmse_accuracy = make_scorer(mean_squared_error,squared=False)
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

# from sklearn import neighbors

import xgboost as xgb



linr_reg = LinearRegression()

tree_reg = DecisionTreeRegressor()

random_forest_reg = RandomForestRegressor(max_features=2, max_leaf_nodes=5, n_estimators=100)



# knn_reg = neighbors.KNeighborsRegressor()





linr_reg.fit(training_features,training_labels)

tree_reg.fit(training_features,training_labels)

random_forest_reg.fit(training_features,training_labels.values.ravel())

# knn_reg.fit(training_features,training_labels)
from sklearn.model_selection import cross_val_score

scores_lin_reg = cross_val_score(linr_reg,training_features,training_labels,scoring='neg_mean_squared_error',cv=10)

linr_rmse_scores = np.sqrt(-scores_lin_reg)

display_scores(linr_rmse_scores)
scores_tree_reg = cross_val_score(tree_reg,training_features,training_labels,scoring='neg_mean_squared_error',cv=10)

tree_rmse_scores = np.sqrt(-scores_tree_reg)

display_scores(tree_rmse_scores)
scores_random_forest_reg = cross_val_score(random_forest_reg,training_features,training_labels.values.ravel(),scoring='neg_mean_squared_error',cv=10)

random_forest_rmse_scores = np.sqrt(-scores_random_forest_reg)

display_scores(random_forest_rmse_scores)
from sklearn.model_selection import GridSearchCV



param_grid = [{'n_estimators':[30,100, 200],'max_features':[2,4,6,8],'max_leaf_nodes':[5,10]},

             {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]}]



# forest_model = random_forest_reg()



grid_search = GridSearchCV(random_forest_reg,param_grid,cv=5,

                          scoring=rmse_accuracy,

                          return_train_score=True)



grid_search.fit(training_features,training_labels.values.ravel())
grid_search.best_estimator_
grid_search.best_params_
from sklearn.ensemble import BaggingRegressor

bag_reg = BaggingRegressor(DecisionTreeRegressor(),n_estimators=500,max_samples=100,bootstrap=True,

                          n_jobs=1)

bag_reg.fit(training_features,training_labels)

prediction_bagging = bag_reg.predict(test_bagging)

predictions = pd.DataFrame(prediction_bagging,columns=['SalePrice'])

# predictions.head()

final_submission = pd.concat([test_data['Id'],predictions['SalePrice']],axis=1)

final_submission.to_csv('Bagging_decision_tree.csv',index=False)
test_features.shape
training_features = OH_X_all[:1460]

test_features = OH_X_all[1460:]

training_labels = y_train.copy()

print(training_features.shape)

print(training_labels.shape)

print(test_features.shape)
import autokeras as ak

# Initialize the structured data regressor.

keras_reg = ak.StructuredDataRegressor(

    overwrite=True,

    max_trials=3,



) # It tries 10 different models.

# Feed the structured data regressor with training data.

keras_reg.fit(

    # The path to the train.csv file.

    training_features,

    # The name of the label column.

   training_labels,

    epochs=10)



# # Evaluate the best model with testing data.

# print(reg.evaluate(test_file_path, 'Price'))
# Predict with the best model.

predicted_keras = keras_reg.predict(test_features)

predictions = pd.DataFrame(predicted_keras,columns=['SalePrice'])

# predictions.head()

final_submission = pd.concat([test_data['Id'],predictions['SalePrice']],axis=1)

final_submission.to_csv('salesprice_keras.csv',index=False)
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

xgb_reg = xgb.XGBRegressor()

# dtrain = xgb.DMatrix(training_features, label=training_labels)

# train_test_split(training_features.as_matrix(), training_labels.as_matrix(), test_size=0.25)

X_train , X_val, y_train, y_val = train_test_split(training_features.to_numpy(), training_labels.to_numpy(), test_size=0.25)

# train_test_split(training_features,training_labels)

xgb_reg.fit(X_train,y_train,eval_set=[(X_val,y_val)],early_stopping_rounds=5)



test_features = test_xgb.to_numpy()

prediction_xgb=xgb_reg.predict(test_features)

#now we pass the testing data to the trained algorithm

predictions = pd.DataFrame(prediction_xgb,columns=['SalePrice'])

# predictions.head()

final_submission = pd.concat([test_data['Id'],predictions['SalePrice']],axis=1)

final_submission.to_csv('Prediction_xgb.csv',index=False)
from sklearn.ensemble import VotingRegressor



voting_reg = VotingRegressor(estimators=[('xgb_reg',xgb_reg),('rf',random_forest_reg),

                                        ('bagging',bag_reg)]

                             )



#  ('keras',keras_reg),

# print(type(voting_reg))



voting_reg.fit(training_features.to_numpy(),training_labels.to_numpy())



# Predict with the best model.

predicted_voting = voting_reg.predict(test_features)

predictions = pd.DataFrame(predicted_voting,columns=['SalePrice'])

# predictions.head()

final_submission = pd.concat([test_data['Id'],predictions['SalePrice']],axis=1)

final_submission.to_csv('salesprice_voting_reg.csv',index=False)


