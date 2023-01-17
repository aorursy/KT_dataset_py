import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
file = "/kaggle/input/house-prices-advanced-regression-techniques/train.csv"
pd.set_option('display.max_columns', None)
data = pd.read_csv(file, index_col='Id')
X = data.drop(['SalePrice'], axis = 1).copy()
y = data['SalePrice'].copy()
#/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv
#/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt
#/kaggle/input/house-prices-advanced-regression-techniques/test.csv
#/kaggle/input/house-prices-advanced-regression-techniques/train.csv

all_cols = X.columns
missing_over_10_cols = [col for col in all_cols if data[col].isnull().sum()>10]
missing_all_cols = [col for col in all_cols if data[col].isnull().any()]
nunique_over_7 = [col for col in X.select_dtypes(['object']).columns
                  if X[col].nunique()>7]
obj_cols = list(set(X.select_dtypes(['object']).columns)-set(nunique_over_7))
num_cols = list(X.select_dtypes(exclude=['object']).columns)

print(type(missing_all_cols))
print(type(num_cols))




from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
def get_score(X,y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, 
                                                          random_state = 0)
    model = RandomForestRegressor(n_estimators = 10, random_state = 0)
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    return mae
test1_num_cols = list(set(num_cols)-set(missing_all_cols))
test1_obj_cols = list(set(obj_cols)-set(missing_all_cols))

from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(handle_unknown = 'ignore', sparse=False)

encoded_test1_obj = pd.DataFrame(one_hot_encoder.fit_transform(X[test1_obj_cols]))
encoded_test1_obj.index = X.index
test_X = pd.concat([X[test1_num_cols],encoded_test1_obj], axis = 1)
print(get_score(test_X,y))
test2_num_cols = set(num_cols)-set(missing_over_10_cols)
test2_obj_cols = set(obj_cols)-set(missing_over_10_cols)

from sklearn.impute import SimpleImputer
simple_imputer_num = SimpleImputer()
simple_imputer_obj = SimpleImputer(strategy = 'most_frequent')

imputed_test2_num = pd.DataFrame(simple_imputer_num.fit_transform(X[test2_num_cols]))
imputed_test2_obj = pd.DataFrame(simple_imputer_obj.fit_transform(X[test2_obj_cols]))

imputed_test2_num.columns = X[test2_num_cols].columns
imputed_test2_obj.columns = X[test2_obj_cols].columns

encoded_imputed_test2_obj = pd.DataFrame(one_hot_encoder.fit_transform(imputed_test2_obj))
encoded_imputed_test2_obj.index = imputed_test2_obj.index

test2_X = pd.concat([imputed_test2_num, encoded_imputed_test2_obj] , axis = 1)
print(get_score(test2_X,y))

def get_score(pipeline, X, y ):
    pred = pipeline.predict(test_X)
    mae = mean_absolute_error(test_y , pred)
    return mae

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

categorical_transformer = Pipeline(steps = [('impute' , SimpleImputer(strategy = 'most_frequent')), 
                                            ('encode' , OneHotEncoder(handle_unknown='ignore'))])

numerical_transformer= SimpleImputer()

preprocessing = ColumnTransformer(transformers = [
    ('num' , numerical_transformer , num_cols),
    ('cat' , categorical_transformer , obj_cols)
])
                                           
model = RandomForestRegressor(n_estimators = 100 , random_state = 0)

pipeline_rf = Pipeline(steps = [('preprocess' , preprocessing),
                              ('model' , model)])
all_col = num_cols + obj_cols
                                       
from xgboost import XGBRegressor

preprossing_categorical = Pipeline(steps = [('impute' , SimpleImputer(strategy = 'most_frequent')),
                                           ('encode' , OneHotEncoder(handle_unknown= 'ignore'))])
preprocessing = ColumnTransformer(transformers = [('num', SimpleImputer(),num_cols),
                                             ('cat', preprossing_categorical, obj_cols)])
model = XGBRegressor(n_estimators=100, learn_rate = 0.02)
pipeline_XGB = Pipeline(steps = [('preprocess' , preprocessing),
                                ('model',model)])
 

train_X, test_X , train_y , test_y = train_test_split(X,y,train_size=0.8 , test_size = 0.2 ,random_state = 2)

pipeline_rf.fit(train_X, train_y)
print("RandomForestRegressor : ",get_score(pipeline_rf , X, y))

pipeline_XGB.fit(train_X, train_y)
print("XGBRegressor : ", get_score(pipeline_XGB , X, y))


