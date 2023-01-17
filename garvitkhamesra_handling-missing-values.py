# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import Imputer

# dataset_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'
# houses_data = pd.read_csv(dataset_file_path)
# df = pd.DataFrame(houses_data)

# # target
# y = df.SalePrice
# # because SalePrice is target
# CanBeX = df.drop(['SalePrice'], axis = 1)
# X = CanBeX.select_dtypes(exclude=['object'])
# # print(X.columns)

# # test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# # test_X = test.select_dtypes(exclude=['object'])
# # print(test_X.columns)

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)
# def score_dataset(X_train, X_test, y_train, y_test):
#     model = RandomForestRegressor()
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     return mean_absolute_error(y_test, preds)

# # # without nan
# # col_missing_values = [col for col in X_train.columns 
# #                                  if X_train[col].isnull().any()]
# # reduced_X_train = X_train.drop(col_missing_values, axis=1)
# # reduced_X_test  = X_test.drop(col_missing_values, axis=1)
# # print("Mean Absolute Error from dropping columns with Missing Values:")
# # print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))
# # result :
# # Mean Absolute Error from dropping columns with Missing Values:
# # 19572.388127853883


# # with imputation
# # my_imputer = Imputer()
# # imputed_X_train = my_imputer.fit_transform(X_train) #fit_transform on training data
# # imputed_X_test = my_imputer.transform(X_test) #transform on test data
# # print("Mean Absolute Error from Imputation:")
# # print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))
# # result :
# # Mean Absolute Error from Imputation:
# # 19291.13675799087

# # Get Score from Imputation with Extra Columns Showing What Was Imputed
# imputed_X_train_plus = X_train.copy()
# imputed_X_test_plus = X_test.copy()

# cols_with_missing = (col for col in X_train.columns 
#                                  if X_train[col].isnull().any())
# for col in cols_with_missing:
#     imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
#     imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# # Imputation
# my_imputer = Imputer()
# imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
# imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

# print("Mean Absolute Error from Imputation while Track What Was Imputed:")
# print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))
# # result :
# # Mean Absolute Error from Imputation while Track What Was Imputed:
# # 19174.150913242014
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score

dataset_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'
houses_data = pd.read_csv(dataset_file_path)
df = pd.DataFrame(houses_data)

# target
y = df.SalePrice
pridictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = df[pridictors]

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_X = test[pridictors]

#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)

def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50), 
                                X, y, 
                                scoring = 'neg_mean_absolute_error').mean()

predictors_without_categoricals = X.select_dtypes(exclude=['object'])
one_hot_encoded_training_predictors = pd.get_dummies(X)

mae_without_categoricals = get_mae(predictors_without_categoricals, y)

mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, y)

print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))
# result :
#     Mean Absolute Error when Dropping Categoricals: 23505
#     Mean Abslute Error with One-Hot Encoding: 23499
model_rf = RandomForestRegressor()
m_rf_fit = model_rf.fit(X, y)
ml_rf_pred = model_rf.predict(test_X)
print(ml_rf_pred)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': ml_rf_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)