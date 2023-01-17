import numpy as np

import pandas as pd

from sklearn.feature_selection import SelectKBest,chi2

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

print(train.shape)

print(test.shape)
train.describe()
train.dropna(axis=0,subset=['SalePrice'],inplace=True)

y=train.SalePrice

train.drop(['SalePrice'],axis=1,inplace=True)



categorical_col=[cname for cname in train.columns if train[cname].nunique()<10

                 and train[cname].dtype=='object']



numeric_col=[cname for cname in train.columns if train[cname].dtype in ['int64','float64']]



my_col=categorical_col+numeric_col

X_train=train[my_col].copy()

X_test=test[my_col].copy()
numerical_transformer=SimpleImputer(strategy='mean')



categorical_transformer=Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore'))])

preprocessor=ColumnTransformer(transformers=[('num',numerical_transformer,numeric_col),('cat',categorical_transformer,categorical_col)])

model=RandomForestRegressor(n_estimators=200,random_state=0)

my_pipeline=Pipeline(steps=[('preprocessor',preprocessor),('model',model)])  

my_pipeline.fit(X_train,y)

preds_test=my_pipeline.predict(X_test)

samplesubmission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")



output = pd.DataFrame({'Id': samplesubmission.Id, 'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)







# output = pd.DataFrame({'Id': X_test.index,

#                        'SalePrice': preds_test})

# output.to_csv('submission.csv', index=False)