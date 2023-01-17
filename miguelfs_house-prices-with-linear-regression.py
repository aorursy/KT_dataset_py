import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_squared_log_error

print(os.listdir())
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")
pd.set_option('display.max_columns', None)
sample_submission.head()
test_data.head()
train_data.head()
train_data["SalePrice"].hist(bins = 50, figsize=(20,15))
plt.show()
corr_matrix = train_data.corr()
corr_matrix["SalePrice"].sort_values(ascending=False)
missing_train_values = train_data.columns[train_data.isnull().any()]
missing_train_values_count = train_data[missing_train_values].isnull().sum()
missing_train_values_count
missing_test_values = test_data.columns[test_data.isnull().any()]
missing_test_values_count = test_data[missing_test_values].isnull().sum()
missing_test_values_count
attributes_with_none_value = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
                              "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC",
                              "Fence", "MiscFeature"]
attributes_with_none_value.append("MasVnrType")
attributes_with_zero_value = ["LotFrontage", "GarageYrBlt", "GarageArea", "GarageCars", "MasVnrArea"]
train_data[attributes_with_none_value] = train_data[attributes_with_none_value].fillna(value="None")
test_data[attributes_with_none_value] = test_data[attributes_with_none_value].fillna(value="None")
train_data[attributes_with_zero_value] = train_data[attributes_with_zero_value].fillna(value=0)
test_data[attributes_with_zero_value] = test_data[attributes_with_zero_value].fillna(value=0)
missing_train_values = train_data.columns[train_data.isnull().any()]
missing_train_values_count = train_data[missing_train_values].isnull().sum()
missing_train_values_count
missing_test_values = test_data.columns[test_data.isnull().any()]
missing_test_values_count = test_data[missing_test_values].isnull().sum()
missing_test_values_count
train_data = train_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
test_data = test_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
missing_train_values = train_data.columns[train_data.isnull().any()]
missing_train_values_count = train_data[missing_train_values].isnull().sum()
missing_train_values_count
missing_test_values = test_data.columns[test_data.isnull().any()]
missing_test_values_count = test_data[missing_test_values].isnull().sum()
missing_test_values_count
train_data.shape
test_data.shape
numeric_attributes = train_data.select_dtypes(include=[np.number])
numeric_attributes = numeric_attributes.drop(["MSSubClass", "Id", "SalePrice"], axis=1)

linear_encoding_attributes = ["LotShape", "LandContour", "GarageCond", "Street","Utilities", "LandSlope", "ExterQual",  "ExterCond", "BsmtQual",
                              "BsmtCond",  "BsmtExposure", "BsmtFinType1", "BsmtFinType2",  "HeatingQC",
                              'CentralAir',  'KitchenQual',  'Functional',  'FireplaceQu', 'GarageFinish',
                              'GarageQual',  'PavedDrive',  'PoolQC' ]


onehot_encoding_attributes = [ "MSSubClass", "Alley", "MSZoning", "LotConfig", "Neighborhood", "Condition1", "Condition2",  'BldgType',
                              'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                              'Foundation', 'Heating', 'Electrical', 'GarageType',  'Fence', 'MiscFeature',
                              'SaleType',  'SaleCondition']
numeric_attributes = list(numeric_attributes.columns.values) + linear_encoding_attributes
labelEncoder = LabelEncoder()
for attribute in linear_encoding_attributes:
    linear_encoded_array = labelEncoder.fit_transform(train_data[attribute])
    train_data[attribute] = linear_encoded_array
    linear_encoded_array = labelEncoder.fit_transform(test_data[attribute])
    test_data[attribute] = linear_encoded_array
train_data.shape
test_data.shape
for attribute in onehot_encoding_attributes:
    train_data = pd.concat([train_data, pd.get_dummies(train_data[attribute], prefix=attribute)],axis=1)
    train_data.drop([attribute],axis=1, inplace=True)
    test_data = pd.concat([test_data, pd.get_dummies(test_data[attribute], prefix=attribute)],axis=1)
    test_data.drop([attribute],axis=1, inplace=True)
train_data.shape
test_data.shape
for col in train_data.columns:
    if col not in test_data.columns and col != "SalePrice":
        test_data[col] = 0
for col in test_data.columns:
    if col not in train_data.columns:
        train_data[col] = 0
test_data.shape
corr_matrix = train_data.corr()
corr_matrix["SalePrice"].sort_values(ascending=False)
attributes = ["SalePrice", "OverallQual", "GrLivArea", "GarageArea"]

attributes = ["SalePrice", "ExterQual", "BsmtQual"]
scatter_matrix(train_data[attributes], figsize = (12,8))
train_data.head()
train_regression_set, test_regression_set = train_test_split(train_data, test_size=0.2, random_state = 30)
regression = LinearRegression()
def plot_hist_values_x_predictions(values, predictions):
    %pylab inline
    pylab.rcParams['figure.figsize'] = (20, 15)
    plt.hist([values, predictions], label=['Sale Price', 'Prediction'], bins=50)
    plt.legend(loc='upper right')
    plt.figure(figsize=(20,15))
    plt.show()

train_regression_set, test_regression_set = train_test_split(train_data, test_size=0.2, random_state = 42)
regression = LinearRegression()
regression.fit(train_regression_set.drop(['SalePrice', 'Id'], axis=1), train_regression_set["SalePrice"])
predictions = regression.predict(test_regression_set.drop(['SalePrice', 'Id'], axis=1))

plot_hist_values_x_predictions(test_regression_set['SalePrice'], predictions)

ms_log_error = mean_squared_log_error(test_regression_set['SalePrice'], predictions)
rms_log_error = np.sqrt(ms_log_error)
rms_log_error
train_data = train_data.assign(log1p_SalePrice = lambda x: np.log1p(x["SalePrice"]))
print(train_data[["SalePrice", "log1p_SalePrice"]].head())
train_data["log1p_SalePrice"].hist(bins = 50, figsize=(20,15))
plt.show()
train_regression_set, test_regression_set = train_test_split(train_data, test_size=0.2, random_state = 40)
lin_regression = LinearRegression()
lin_regression.fit(train_regression_set.drop(['SalePrice', 'Id', "log1p_SalePrice"], axis=1), train_regression_set["log1p_SalePrice"])

log1p_predictions = lin_regression.predict(test_regression_set.drop(['SalePrice', 'Id', "log1p_SalePrice"], axis=1))
predictions = np.expm1(log1p_predictions)

plot_hist_values_x_predictions(test_regression_set['SalePrice'], predictions)

ms_log_error = mean_squared_log_error(test_regression_set['SalePrice'], predictions)
rms_log_error = np.sqrt(ms_log_error)
rms_log_error
lasso_regression = Lasso(alpha = 0.0005, random_state=1)
lasso_regression.fit(train_regression_set.drop(['SalePrice', 'Id', "log1p_SalePrice"], axis=1), train_regression_set["log1p_SalePrice"])

log1p_predictions = lasso_regression.predict(test_regression_set.drop(['SalePrice', 'Id', "log1p_SalePrice"], axis=1))
predictions = np.expm1(log1p_predictions)

plot_hist_values_x_predictions(test_regression_set['SalePrice'], predictions)
ms_log_error = mean_squared_log_error(test_regression_set['SalePrice'], predictions)
rms_log_error = np.sqrt(ms_log_error)
rms_log_error
scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(train_regression_set.drop(['SalePrice', 'Id', "log1p_SalePrice"], axis=1))
scaled_test_data = scaler.fit_transform(test_regression_set.drop(['SalePrice', 'Id', "log1p_SalePrice"], axis=1))

def get_rms_log_error_from_lasso_with(alpha):
    lasso_regression = Lasso(alpha, random_state=1)
    lasso_regression.fit(scaled_train_data, train_regression_set["log1p_SalePrice"])
    log1p_predictions = lasso_regression.predict(scaled_test_data)
    predictions = np.expm1(log1p_predictions)
    ms_log_error = mean_squared_log_error(test_regression_set['SalePrice'], predictions)
    rms_log_error = np.sqrt(ms_log_error)
    return (predictions, rms_log_error, lasso_regression)

(_, error, regressor) = get_rms_log_error_from_lasso_with(alpha=0.0005)
print(error)
alpha_iterator = 0.00010
best_rms_log_error = 99999
best_alpha = alpha_iterator
xablau = False

while alpha_iterator < 0.010:
    alpha_iterator =  alpha_iterator + 0.00010
    (_, rms_log_error, _ )= get_rms_log_error_from_lasso_with(alpha_iterator)
    if rms_log_error < best_rms_log_error:
        xablau = True
        best_alpha = alpha_iterator
        best_rms_log_error = rms_log_error
        
print("alpha = ", best_alpha, " and rms_log_error = ", best_rms_log_error)
(predictions, error, lasso_regressor) = get_rms_log_error_from_lasso_with(alpha = 0.0031)
print(error)
scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(train_data.drop(['SalePrice', 'Id', "log1p_SalePrice"], axis=1))
scaled_test_data = scaler.fit_transform(test_data.drop(['Id'], axis=1))

lasso_regression = Lasso(alpha = 0.0031, random_state=1)
lasso_regression.fit(scaled_train_data, train_data["log1p_SalePrice"])
log1p_predictions = lasso_regression.predict(scaled_test_data)
predictions = np.expm1(log1p_predictions)

predictions
sample_submission.head()
sample_submission["SalePrice"] = predictions
sample_submission.head()
sample_submission.to_csv("sample_submisson_on_the_go.csv", index=False)
print(os.listdir())
