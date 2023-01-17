

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)f

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')



low_cardinality_cols = [cname for cname in train.columns if train[cname].nunique() < 10 and 

                        train[cname].dtype == "object"]



numerical_cols = [cname for cname in train.columns if train[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numerical_cols

train = train[my_cols].copy()

numerical_col1s = [cname for cname in test.columns if test[cname].dtype in ['int64', 'float64']]

my_col1s = low_cardinality_cols + numerical_col1s

test = test[my_col1s].copy()

test


for c in low_cardinality_cols :

    train[c].fillna((train[c].value_counts().idxmax()),inplace=True)

    test[c].fillna((train[c].value_counts().idxmax()),inplace=True)

    
test.isnull().sum()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for col in low_cardinality_cols:

    train[col] = label_encoder.fit_transform(train[col])

    test[col] = label_encoder.transform(test[col])
test
cols=[col for col in train.columns if train[col].isnull().any()]



col1s=[col for col in test.columns if test[col].isnull().any()]
for c in numerical_cols :

    train[c].fillna((train[c].mean()),inplace=True)
for c in test.columns :

    test[c].fillna((test[c].mean()),inplace=True)
y=train['SalePrice']
train=train.drop('SalePrice',axis=1)
X_train,X_test,y_train,y_test=train_test_split(train,y)
model=XGBRegressor(n_iterations=1000,learning_rate=0.05)

model.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_test, y_test)], 

             verbose=False)
pred=model.predict(X_test)
print(mean_absolute_error(pred,y_test))
pred=model.predict(test)

pred.shape




preds_test = model.predict(test)

output = pd.DataFrame({'Id': test.Id,

                       'SalePrice': preds_test})

output.to_csv('output.csv', index=False)

print(output)