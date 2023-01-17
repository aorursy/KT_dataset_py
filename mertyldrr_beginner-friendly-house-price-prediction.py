def get_categorical_features(data_df):

    return data_df.select_dtypes(include='object')





def get_numerical_features(data_df):

    return data_df.select_dtypes(exclude='object')





def read_train_test_data():

    train_df = pd.read_csv('../input/train.csv', index_col='Id')

    test_df = pd.read_csv('../input/test.csv', index_col='Id')

    

    print("Shape of Train Data: " + str(train_df.shape))

    print("Shape of Test Data: " + str(test_df.shape))



    return train_df, test_df



def inv_y(transformed_y):

    return np.exp(transformed_y)



def MissingandUniqueStats(df):

    total_entry_list = []

    total_missing_list = []

    missing_value_ratio_list = []

    datatype_list = []

    unique_values_list = []

    number_of_unique_values_list = []

    variable_names_list = []

    

    for col in df.columns:

        total_entry_list.append(df[col].shape[0] - df[col].isna().sum())

        total_missing_list.append(df[col].isna().sum())

        missing_value_ratio_list.append(round((df[col].isna().sum() / len(df[col])), 4))

        datatype_list.append(df[col].dtype)

        unique_values_list.append(df[col].unique())

        number_of_unique_values_list.append(len(df[col].unique()))

        variable_names_list.append(col)

    

    all_info_df = pd.DataFrame({'#TotalEntry':total_entry_list, '#TotalMissingValue':total_missing_list,\

                               '%MissingValueRatio':missing_value_ratio_list, 'DataType':datatype_list, 'UniqueValues':unique_values_list,\

                               '#UniqueValues':number_of_unique_values_list})

    

    all_info_df.index = variable_names_list

    all_info_df.index.name='Columns'

    

    return all_info_df.sort_values(by="#TotalMissingValue",ascending=False)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train_df, test_df = read_train_test_data()
target = train_df.SalePrice



plt.figure(figsize=(8, 6))

sns.distplot(target)

plt.title('Distribution of SalePrice')

plt.show()
sns.distplot(np.log10(target))

plt.title('Distribution of Log10-transformed SalePrice')

plt.xlabel('log10(SalePrice)')

plt.show()
sns.distplot(np.log(target))

plt.title('Distribution of Log-transformed SalePrice')

plt.xlabel('log(SalePrice)')

plt.show()
sns.distplot(np.sqrt(target))

plt.title('Distribution of Square root-transformed SalePrice')

plt.xlabel('sqrt(SalePrice)')

plt.show()
info_df = MissingandUniqueStats(train_df)

info_df.head(20)
num_correlation = train_df.select_dtypes(exclude='object').corr()

plt.figure(figsize=(20,20))

plt.title('High Correlation')

sns.heatmap(num_correlation > 0.8, annot=True, square=True, cmap="YlGnBu")
num_columns = train_df.select_dtypes(exclude='object').columns

corr_to_price = num_correlation['SalePrice']

n_cols = 5

n_rows = 8

fig, ax_arr = plt.subplots(n_rows, n_cols, figsize=(20,20), sharey=True)

plt.subplots_adjust(bottom=-0.9)

for j in range(n_rows):

    for i in range(n_cols):

        plt.sca(ax_arr[j, i])

        index = i + j*n_cols

        if index < len(num_columns):

            plt.scatter(train_df[num_columns[index]], train_df.SalePrice)

            plt.xlabel(num_columns[index])

            plt.title('Corr to SalePrice = '+ str(np.around(corr_to_price[index], decimals=3)))

plt.show()

numerical_features = get_numerical_features(train_df).drop(['SalePrice'], axis=1).copy()

info_numeric = MissingandUniqueStats(numerical_features)

info_numeric
fig = plt.figure(figsize=(12,18))

for i in range(len(numerical_features.columns)):

    fig.add_subplot(9,4,i+1)

    sns.boxplot(y=numerical_features.iloc[:,i])



plt.tight_layout()

plt.show()
f = plt.figure(figsize=(18,30))



for i in range(len(numerical_features.columns)):

    f.add_subplot(9, 4, i+1)

    sns.scatterplot(numerical_features.iloc[:,i], target)

    

plt.tight_layout()

plt.show()
categorical_features = get_categorical_features(train_df).copy()

info_categorical = MissingandUniqueStats(categorical_features)

info_categorical
col = train_df['Fence']

f, ax = plt.subplots(figsize=(10,6))

sns.boxplot(y=target, x=col)

plt.show()
col = train_df['FireplaceQu']

f, ax = plt.subplots(figsize=(10,6))

sns.boxplot(y=target, x=col)

plt.show()
from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
train_df_copy = train_df.copy()

# MasVnrArea column(8 missing)

train_df_copy.MasVnrArea = train_df_copy.MasVnrArea.fillna(0)



# Categorical columns:

cat_cols_fill_none = ['Fence', 'FireplaceQu', 'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType',

                     'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond', 'MasVnrType']

for col in cat_cols_fill_none:

    train_df_copy[col] = train_df_copy[col].fillna("None")
print(train_df_copy.shape)

train_df_copy = train_df_copy.drop(train_df_copy[train_df_copy['LotFrontage'] > 200].index)

train_df_copy = train_df_copy.drop(train_df_copy[train_df_copy['LotArea'] > 100000].index)

train_df_copy = train_df_copy.drop(train_df_copy[train_df_copy['BsmtFinSF1'] > 4000].index)

train_df_copy = train_df_copy.drop(train_df_copy[train_df_copy['TotalBsmtSF'] > 6000].index)

train_df_copy = train_df_copy.drop(train_df_copy[train_df_copy['LowQualFinSF'] > 550].index)

train_df_copy = train_df_copy.drop(train_df_copy[(train_df_copy['GrLivArea'] > 4000) & (target < 300000)].index)

print(train_df_copy.shape)
train_df_copy['SalePrice'] = np.log(train_df_copy['SalePrice'])

train_df_copy = train_df_copy.rename(columns={'SalePrice': 'SalePrice_log'})



# train_df_copy['SalePrice'] = np.log10(train_df_copy['SalePrice'])

# train_df_copy = train_df_copy.rename(columns={'SalePrice': 'SalePrice_log10'})
train_df_copy.drop(columns=['GarageArea','TotRmsAbvGrd','GarageYrBlt', 'MoSold', 'YrSold', '1stFlrSF'],axis=1,inplace=True) 

test_df.drop(columns=['GarageArea','TotRmsAbvGrd','GarageYrBlt', 'MoSold', 'YrSold','1stFlrSF'],axis=1,inplace=True)
train_df_copy.drop(columns=['PoolQC', 'MiscFeature', 'Alley'], axis=1, inplace=True)

test_df.drop(columns=['PoolQC', 'MiscFeature', 'Alley'], axis=1, inplace=True)
y = train_df_copy.SalePrice_log

train_df_copy.drop('SalePrice_log', axis=1, inplace=True)
X = train_df_copy

# One-hot encoding

X = pd.get_dummies(X)

# Train-Validation split

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size= 0.25 ,random_state=1)

print("Train_X shape:",train_X.shape,"val_X shape:",val_X.shape)

my_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

train_X = my_imputer.fit_transform(train_X)

val_X = my_imputer.transform(val_X)
from sklearn.metrics import mean_absolute_error



from sklearn.linear_model import Lasso

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
# Series to collate mean absolute errors for each algorithm

mae_compare = pd.Series()

mae_compare.index.name = 'Algorithm'



# Random Forest. Define the model. =============================

rf_model = RandomForestRegressor(random_state=5)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(inv_y(rf_val_predictions), inv_y(val_y))



mae_compare['RandomForest'] = rf_val_mae



# XGBoost. Define the model. ======================================

xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

xgb_model.fit(train_X, train_y, early_stopping_rounds=5, 

              eval_set=[(val_X,val_y)], verbose=False)

xgb_val_predictions = xgb_model.predict(val_X)

xgb_val_mae = mean_absolute_error(inv_y(xgb_val_predictions), inv_y(val_y))



mae_compare['XGBoost'] = xgb_val_mae



# Linear Regression =================================================

linear_model = LinearRegression()

linear_model.fit(train_X, train_y)

linear_val_predictions = linear_model.predict(val_X)

linear_val_mae = mean_absolute_error(inv_y(linear_val_predictions), inv_y(val_y))



mae_compare['LinearRegression'] = linear_val_mae



# Lasso ==============================================================

lasso_model = Lasso(alpha=0.0005, random_state=5)

lasso_model.fit(train_X, train_y)

lasso_val_predictions = lasso_model.predict(val_X)

lasso_val_mae = mean_absolute_error(inv_y(lasso_val_predictions), inv_y(val_y))



mae_compare['Lasso'] = lasso_val_mae



# Ridge ===============================================================

ridge_model = Ridge(alpha=0.002, random_state=5)

ridge_model.fit(train_X, train_y)

ridge_val_predictions = ridge_model.predict(val_X)

ridge_val_mae = mean_absolute_error(inv_y(ridge_val_predictions), inv_y(val_y))



mae_compare['Ridge'] = ridge_val_mae



# ElasticNet ===========================================================

elastic_net_model = ElasticNet(alpha=0.02, random_state=5, l1_ratio=0.7)

elastic_net_model.fit(train_X, train_y)

elastic_net_val_predictions = elastic_net_model.predict(val_X)

elastic_net_val_mae = mean_absolute_error(inv_y(elastic_net_val_predictions), inv_y(val_y))



mae_compare['ElasticNet'] = elastic_net_val_mae



# Gradient Boosting Regression ==========================================

gbr_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, 

                                      max_depth=4, random_state=5)

gbr_model.fit(train_X, train_y)

gbr_val_predictions = gbr_model.predict(val_X)

gbr_val_mae = mean_absolute_error(inv_y(gbr_val_predictions), inv_y(val_y))



mae_compare['GradientBoosting'] = gbr_val_mae





print('MAE values for different algorithms:')

mae_compare.sort_values(ascending=True).round()
from sklearn.model_selection import cross_val_score



imputer = SimpleImputer()

imputed_X = imputer.fit_transform(X)

n_folds = 10
scores = cross_val_score(lasso_model, imputed_X, y, scoring='neg_mean_squared_error', 

                         cv=n_folds)

lasso_mae_scores = np.sqrt(-scores)



print('For LASSO model:')

print(lasso_mae_scores.round(decimals=2))



print('Mean RMSE = ' + str(lasso_mae_scores.mean().round(decimals=3)))
scores = cross_val_score(gbr_model, imputed_X, y, scoring='neg_mean_squared_error', 

                         cv=n_folds)

gbr_mae_scores = np.sqrt(-scores)



print('For Gradient Boosting model:')

# print(lasso_mae_scores.round(decimals=2))

print('Mean RMSE = ' + str(gbr_mae_scores.mean().round(decimals=3)))
scores = cross_val_score(xgb_model, imputed_X, y, scoring='neg_mean_squared_error', 

                         cv=n_folds)

mae_scores = np.sqrt(-scores)



print('For XGBoost model:')

# print(mae_scores.round(decimals=2))

print('Mean RMSE = ' + str(mae_scores.mean().round(decimals=3)))
scores = cross_val_score(rf_model, imputed_X, y, scoring='neg_mean_squared_error', 

                         cv=n_folds)

mae_scores = np.sqrt(-scores)



print('For Random Forest model:')

# print(mae_scores.round(decimals=2))

print('Mean RMSE = ' + str(mae_scores.mean().round(decimals=3)))
from sklearn.model_selection import GridSearchCV



param_grid = [{'alpha': [0.0007, 0.0005, 0.005]}]

top_reg = Lasso()



grid_search = GridSearchCV(top_reg, param_grid, cv=5, 

                           scoring='neg_mean_squared_error')



grid_search.fit(imputed_X, y)



grid_search.best_params_
_ , test_df = read_train_test_data()
# create test_X which to perform all previous pre-processing on

test_X = test_df.copy()



# Repeat treatments for missing/null values =====================================

# Numerical columns:

test_X.MasVnrArea = test_X.MasVnrArea.fillna(0)



# Categorical columns:

for cat in cat_cols_fill_none:

    test_X[cat] = test_X[cat].fillna("None")



test_X = test_df.drop(['GarageArea','TotRmsAbvGrd','GarageYrBlt', 'MoSold', 'YrSold', '1stFlrSF'], axis=1)



# One-hot encoding for categorical data =========================================

test_X = pd.get_dummies(test_X)





# ===============================================================================

# Ensure test data is encoded in the same manner as training data with align command

final_train, final_test = X.align(test_X, join='left', axis=1)



# Imputer for all other missing values in test data. Note: Do not 'fit_transform'

final_test_imputed = my_imputer.transform(final_test)
# Create model - on full set of data (training & validation)

# Best model = Lasso

final_model = Lasso(alpha=0.0005, random_state=5)

# final_model = XGBRegressor(n_estimators=1500, learning_rate=0.03)

final_train_imputed = my_imputer.fit_transform(final_train)



# Fit the model using all the data - train it on all of X and y

final_model.fit(final_train_imputed, y)
# make predictions which we will submit. 

test_preds = final_model.predict(final_test_imputed)



# The lines below shows you how to save your data in the format needed to score it in the competition

# Reminder: predictions are in log(SalePrice). Need to inverse-transform.

output = pd.DataFrame({'Id': test_df.index,

                       'SalePrice': inv_y(test_preds)})



output.to_csv('submission.csv', index=False)