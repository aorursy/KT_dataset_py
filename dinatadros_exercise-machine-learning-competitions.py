# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import LinearSVR

from learntools.core import *

from sklearn.preprocessing import *

from sklearn.feature_selection import *

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)



# Create target object and call it y

# Create X, 

features = ['YearBuilt', 'OverallQual', 'YearRemodAdd','GrLivArea', 'GarageCars', 'FullBath', 'TotalBsmtSF']

features_output = features + ['SalePrice']

#Using Pearson Correlation

# plt.figure(figsize=(12,10))

print(features_output)

print(home_data[features_output])

print(home_data[features_output].corr()['SalePrice'][:])

# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

# plt.show()



# home_data = home_data.dropna(subset=features)

y = home_data.SalePrice

X = home_data[features]



limitPer = len(X) * .7

X = X.dropna(thresh=limitPer, axis=1)



categorical_features = list(X.select_dtypes(include=['object']).columns)





# print(categorical_features)



###############################################################################################

X['label'] = 'train'



test_data = pd.read_csv("../input/home-data-for-ml-course/test.csv")

test_X = test_data[features]

test_X['label'] = 'test'

test_X.fillna(test_X.mean())



# Concat

concat_df = pd.concat([X , test_X])





# Create your dummies

features_df = pd.get_dummies(concat_df, columns=categorical_features)

# Split your data

X = features_df[features_df['label'] == 'train']



X = X.fillna(X.mean())

test_X = features_df[features_df['label'] == 'test']



# Drop your labels

X = X.drop('label', axis=1)

test_X = test_X.drop('label', axis=1)



scaler = StandardScaler()

# X = scaler.fit_transform( X )

# test_X = scaler.transform( test_X )



print(X.columns)
correlated_features = []

correlation_matrix = home_data.drop('SalePrice', axis=1).corr()

k = 0

for i in range(len(correlation_matrix.columns)):

    for j in range(i):

        if abs(correlation_matrix.iloc[i, j]) > 0.7:

            colname = correlation_matrix.columns[i]              

            rowname = correlation_matrix.columns[j]   

            correlated_features.append(k)

            k = k + 1

            correlated_features.append(colname)

            correlated_features.append(rowname)

            

print(correlated_features)
# Create a random forest classifier

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



clf = svm.SVR()

# Train the classifier

clf.fit(train_X, train_y)



print(len(clf.feature_importances_))

print(len(features_labels))

# Print the name and gini importance of each feature

print([zip(y,x) for (y, x) in sorted(zip(features_labels,clf.feature_importances_), key=lambda pair: pair[0])])

for feature in zip(features_labels, clf.feature_importances_):

    print(feature)

    

# features that have an importance of more than 0.15

sfm = SelectFromModel(clf, threshold=0.15)



# Train the selector

sfm.fit(train_X, train_y)
from sklearn.linear_model import LinearRegression

from sklearn import linear_model

# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=150, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))



svm_model = svm.LinearSVR(random_state=1)

# Train the classifier

svm_model.fit(train_X, train_y)

svm_val_predictions = svm_model.predict(val_X)

svm_mae = mean_absolute_error(svm_val_predictions, val_y)





print("Validation MAE for SVM Model: {:,.0f}".format(svm_mae))



LR_model = LinearRegression()

# Train the classifier

LR_model.fit(train_X, train_y)

LR_val_predictions = LR_model.predict(val_X)

LR_mae = mean_absolute_error(LR_val_predictions, val_y)





print("Validation MAE for SVM Model: {:,.0f}".format(LR_mae))



reg_model = linear_model.Ridge(alpha=.5)

# Train the classifier

reg_model.fit(train_X, train_y)

reg_val_predictions = reg_model.predict(val_X)

reg_mae = mean_absolute_error(reg_val_predictions, val_y)





print("Validation MAE for SVM Model: {:,.0f}".format(reg_mae))
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(train_X, train_y)

test_X = test_X.fillna(0)

# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)





# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)


# print(mutual_info_classif(X, y, discrete_features = True))

# Feature extraction

# test = SelectKBest(score_func=chi2, k=4)

# fit = test.fit(X, y)



# # Summarize scores

# np.set_printoptions(precision=3)

# print(fit.scores_)





# features = fit.transform(X)

# # Summarize selected features

# print(features[0:5,:])



# rfc = RandomForestRegressor(random_state=1)

# rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10))

# rfecv.fit(X, y)



# dset = pd.DataFrame()

# dset['attr'] = features

# print(rfecv.estimator_.feature_importances_)

# dset['importance'] = rfecv.estimator_.feature_importances_



# dset = dset.sort_values(by='importance', ascending=False)





# plt.figure(figsize=(16, 14))

# plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')

# plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)

# plt.xlabel('Importance', fontsize=14, labelpad=20)

# plt.show()



# categorical_features = ['Foundation',  'Fence', 'SaleCondition', 'LotFrontage', 'Neighborhood', 'Condition1', 'ExterQual', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'Functional','GarageType']

# features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']