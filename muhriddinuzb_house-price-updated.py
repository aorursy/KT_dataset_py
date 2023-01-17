import pandas as pd
import numpy as np

data = pd.read_csv('../input/train.csv')
target = data.SalePrice
predictors = data.drop(['SalePrice'],axis=1)

numeric_predictors = predictors.select_dtypes(exclude=['object'])


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(numeric_predictors,
                                                    target,
                                                    train_size=0.7,
                                                    test_size=0.3,
                                                    random_state=0)
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test,preds)
col_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(col_with_missing,axis=1)
reduced_X_test = X_test.drop(col_with_missing,axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))
from sklearn.preprocessing import Imputer

my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

col_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

for col in col_with_missing:
    imputed_X_train_plus[col+'_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col+'_was_missing'] = imputed_X_test_plus[col].isnull()
    
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))


# Read the data
import pandas as pd
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# Drop houses where the target is missing
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

target = train_data.SalePrice

# Since missing values isn't the focus of this tutorial, we use the simplest
# possible approach, which drops these columns. 
# For more detail (and a better approach) to missing values, see
# https://www.kaggle.com/dansbecker/handling-missing-values
   
missing_train = [col for col in train_data.columns 
                                 if train_data[col].isnull().any()]   
missing_test = [col for col in test_data.columns 
                                 if test_data[col].isnull().any()]   
cols_with_missing = list(set(missing_train+missing_test))
candidate_train_predictors = train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)
candidate_test_predictors = test_data.drop(['Id'] + cols_with_missing, axis=1)
# "cardinality" means the number of unique values in a column.
# We use it as our only way to select categorical columns here. This is convenient, though
# a little arbitrary.
low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].nunique() < 10 and
                                candidate_train_predictors[cname].dtype == "object"]
numeric_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols
train_predictors = candidate_train_predictors[my_cols]
test_predictors = candidate_test_predictors[my_cols]

one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def get_mae(X,y):
    return -1 * cross_val_score(RandomForestRegressor(50),
                               X,y,
                               scoring='neg_mean_absolute_error').mean()
predictors_without_categoricals = train_predictors.select_dtypes(exclude=['object'])

mae_without_categoricals = get_mae(predictors_without_categoricals,target)
mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors,target)
print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))
one_hot_encoded_train_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                   join='inner',
                                                                   axis=1)
print(train_predictors.shape,test_predictors.shape)
missing_train = [col for col in final_train.columns 
                                 if final_train[col].isnull().any()]   
missing_test = [col for col in final_test.columns 
                                 if final_test[col].isnull().any()]   

a= set(missing_train+missing_test)
print(len(missing_train))
print(missing_test)

print(len(list(a)))
model = RandomForestRegressor()
model.fit(final_train,target)
predicted_prices = model.predict(final_test)
print(predicted_prices)
my_submission = pd.DataFrame({'Id':test_data.Id,'SalePrice':predicted_prices})
my_submission.to_csv('submission.csv',index=False)
