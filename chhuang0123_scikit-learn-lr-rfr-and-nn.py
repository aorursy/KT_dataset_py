#%% 1
# import basic library
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
#%% 2
# configure pandas configuration
pd.options.display.max_rows = 100
pd.options.display.float_format = '{:.1f}'.format

#%% 3
# import dataset
dataframe = pd.read_csv("../input/train.csv", sep=",")
dataframe
#%% 4
# list distribution for int fields
for column_name in dataframe.columns:
    if dataframe[column_name].dtype not in ['int64']:
        continue

    plt.hist(dataframe[column_name])
    plt.title(column_name)
    plt.show()
#%% 5
# list distribution for object fields
for column_name in dataframe.columns:
    if dataframe[column_name].dtype not in ['object']:
        continue

    dataframe[column_name].value_counts().plot.bar()
    plt.title(column_name)
    plt.show()
#%% 6
# get the correlation between two columns
corr = dataframe.corr()
corr.style.background_gradient()

# 1. OverallQual   : 0.790981601
# 2. GrLivArea 	   : 0.708624478
# 3. GarageCars    : 0.640409197
# 4. GarageArea    : 0.623431439
# 5. TotalBsmtSF   : 0.613580552
# 6. 1stFlrSF      : 0.605852185
# 7. FullBath      : 0.560663763
# 8. TotRmsAbvGrd  : 0.533723156
# 9. YearBuilt     : 0.522897333
#%% 6-1
plt.scatter(x=dataframe['OverallQual'], y=dataframe['SalePrice'])
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')
plt.show()
#%% 6-2
# remove > 3000
plt.scatter(x=dataframe['GrLivArea'], y=dataframe['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show()
#%% 6-3
plt.scatter(x=dataframe['GarageCars'], y=dataframe['SalePrice'])
plt.xlabel('GarageCars')
plt.ylabel('SalePrice')
plt.show()
#%% 6-4
# remove > 1200
plt.scatter(x=dataframe['GarageArea'], y=dataframe['SalePrice'])
plt.ylabel('GarageArea')
plt.xlabel('OverallQual')
plt.show()
#%% 6-5
# remove > 2500
plt.scatter(x=dataframe['TotalBsmtSF'], y=dataframe['SalePrice'])
plt.ylabel('TotalBsmtSF')
plt.xlabel('OverallQual')
plt.show()
#%% 6-6
dataframe['SaleCondition'].value_counts().plot.bar()
plt.title('SaleCondition')
plt.show()

sale_conditions = dataframe['SaleCondition'].unique()
sale_conditions
#%% 7
# define how to pre-process feature field
def preprocess_features(dataframe):
    selected_features = dataframe[dataframe.columns]
    processed_features = selected_features.copy()

    # remove outlier
    column_name = 'GrLivArea'
    processed_features[column_name] = (
        dataframe[column_name]).apply(lambda x: min(x, 3000))

    column_name = 'GarageArea'
    processed_features[column_name] = (
        dataframe[column_name]).apply(lambda x: min(x, 1200))

    column_name = 'TotalBsmtSF'
    processed_features[column_name] = (
        dataframe[column_name]).apply(lambda x: min(x, 2500))

    # One Hot Encoder
    column_name = 'SaleCondition'
    processed_features = pd.concat(
        [
            dataframe,
            pd.get_dummies(dataframe[column_name], prefix=column_name)
        ],
        axis=1)

    return processed_features
#%% 8
# define how to pre-process target field
def preprocess_targets(dataframe, target_label='SalePrice'):
    selected_targets = dataframe[dataframe.columns]
    processed_targets = selected_targets.copy()

    if target_label in dataframe:
        # Scale the target to be in units of thousands of dollars.
        processed_targets[target_label] = dataframe[target_label] / 1000.0

    return processed_targets
#%% 9
# define minimal feature fields
minimal_features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
]

for sale_condition in sale_conditions:
    minimal_features.append("SaleCondition_{}".format(sale_condition))

minimal_features
#%% 10
# pre-process dataset
test_dataframe = preprocess_features(dataframe)
test_dataframe = preprocess_targets(test_dataframe)
test_dataframe
#%% 11
# define real dataset
X = test_dataframe[minimal_features]
y = test_dataframe['SalePrice']

# split into training and validation dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#%% 12
# start to train
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
#%% 13
# start to predict
y_pred = linreg.predict(X_test)

# print result
from sklearn import metrics
print("MAE: ", metrics.mean_absolute_error(y_test, y_pred))
print("MSE: ", metrics.mean_squared_error(y_test, y_pred))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df
#%% 14
# visualize result
actual_values = y_test
plt.scatter(y_pred, actual_values, alpha=.75, color='b')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()
#%% 15
# try to predict test dataset
submission_dataframe = pd.read_csv("../input/test.csv", sep=",")

# submission file
submission_csv = pd.DataFrame()
submission_csv['Id'] = submission_dataframe['Id']

# submission dataset
submission_dataframe = preprocess_features(submission_dataframe)
submission_dataframe = submission_dataframe[minimal_features]
#%% 16
# fix missing value
for column_name in minimal_features:
    if column_name in submission_dataframe:
        submission_dataframe[column_name] = submission_dataframe[
            column_name].fillna(submission_dataframe[column_name].median())
#%% 17
# start to predict
predictions = linreg.predict(submission_dataframe)
predictions = pd.DataFrame({'Predicted': predictions})
predictions['Predicted'] = predictions['Predicted'] * 1000.0

# submit file
submission_csv['SalePrice'] = predictions['Predicted']
submission_csv.to_csv('submission.csv', index=False)

#%% 18
# tree base
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)
y_pred = forest_model.predict(X_test)

# result
from sklearn import metrics
print("MAE: ", metrics.mean_absolute_error(y_test, y_pred))
print("MSE: ", metrics.mean_squared_error(y_test, y_pred))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df
#%% 19
predictions = forest_model.predict(submission_dataframe)
predictions = pd.DataFrame({'Predicted': predictions})
predictions['Predicted'] = predictions['Predicted'] * 1000.0

# submit file
submission_csv['SalePrice'] = predictions['Predicted']
submission_csv.to_csv('submission_tree.csv', index=False)

#%% 20
from sklearn.neural_network import MLPRegressor

mlpr_model = MLPRegressor()
mlpr_model.fit(X_train, y_train)
y_pred = mlpr_model.predict(X_test)

# result
from sklearn import metrics
print("MAE: ", metrics.mean_absolute_error(y_test, y_pred))
print("MSE: ", metrics.mean_squared_error(y_test, y_pred))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df
#%% 21
predictions = mlpr_model.predict(submission_dataframe)
predictions = pd.DataFrame({'Predicted': predictions})
predictions['Predicted'] = predictions['Predicted'] * 1000.0

# submit file
submission_csv['SalePrice'] = predictions['Predicted']
submission_csv.to_csv('submission_nn.csv', index=False)
