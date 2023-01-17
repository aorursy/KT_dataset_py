import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
X = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
X.head(1)
X_test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
X_test.head(1)
X = X.drop(columns=['MiscFeature','Fence','PoolQC','FireplaceQu','Alley'])
X.head(1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)
num_cols = X_train.select_dtypes(include='number').columns.to_list()
cat_cols = X_train.select_dtypes(exclude='number').columns.to_list()
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
ct = ColumnTransformer(remainder='drop',
                       transformers=[
                           ('numerical', num_pipe, num_cols),
                           ('categorical', cat_pipe, cat_cols)
                       ])
model=Pipeline([
    ('transformer', ct),   
    ('predictor', RandomForestRegressor())
])
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
mean_squared_error(y_pred, y_valid, squared=False)
model.fit(X, y)
y_res = model.predict(X_test)
res = pd.DataFrame({'Id': X_test.Id, 
                    'SalePrice': y_res})
res.to_csv('submission.csv',index=False)
