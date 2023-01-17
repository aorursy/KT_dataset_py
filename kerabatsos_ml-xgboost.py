import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OrdinalEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

sample_submission = pd.read_csv("../input/home-data-for-ml-course/sample_submission.csv")

X_full = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')
# The entries with NaN values on SalePrice column (which we are going to use to predict) have been dropped.

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)



# y created using the data with SalePrice column.

y = X_full.SalePrice



# SalePrice columns has been dropped from X_full data.

X_full.drop(['SalePrice'], axis=1, inplace=True)



# Training and Validation data has been splited both for X and y.

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)



# The columns with possibly ordering values have been choosen for Ordinal Encoding.

cat_to_ord = ['ExterQual', 'ExterCond',  'BsmtQual', 'BsmtCond', 'HeatingQC', 

               'FireplaceQu', 'GarageQual','GarageCond', 'PoolQC']



# Other categorical data choosen for One-Hot Encoding.

cat_to_onehot = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10 and 

                    X_train_full[cname].dtype == "object" and

                   cname not in cat_to_ord]



# Numerical columns have been choosen to process differently.

numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]



# X_train, X_valid and X_test have been created with just selected columns.

my_cols = cat_to_ord + cat_to_onehot + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
y
numerical_transformer = SimpleImputer(strategy='mean')



cat_to_onehot_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant')),

                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])



cat_to_ord_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),

                                          ('label', OrdinalEncoder())])



preprocessor = ColumnTransformer(transformers=[

    ('num', numerical_transformer, numerical_cols),

    ('cat_ordinal', cat_to_ord_transformer, cat_to_ord),

    ('cat_onehot', cat_to_onehot_transformer, cat_to_onehot)])



model = XGBRegressor(n_estimators=692,learning_rate=0.05111, n_jobs=8, min_child_weight=5, 

                     subsample=0.76, colsample_bytree=0.9, base_score=3.819)
my_pipeline2 = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])



my_pipeline2.fit(X_train, y_train)



preds = my_pipeline2.predict(X_valid)



score = mean_absolute_error(y_valid, preds)

print('MAE:', score)
preds_test = my_pipeline2.predict(X_test)
output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)