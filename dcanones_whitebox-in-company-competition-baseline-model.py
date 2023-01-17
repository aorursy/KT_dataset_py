# imports

import numpy as np

import pandas as pd
# dataviz

import matplotlib as mpl

import matplotlib.pyplot as plt



%matplotlib inline

%config InlineBackend.figure_format = 'svg'
# mpl.rcParams['figure.figsize'] = (12.0, 8.0)
train = pd.read_csv('../input/whitebox-in-company-training/diamonds_train.csv')

test = pd.read_csv('../input/whitebox-in-company-training/diamonds_test.csv')

sample_sub = pd.read_csv('../input/whitebox-in-company-training/sample_submission.csv')
# ejemplo

train['carat'].plot(kind='hist', bins=20, title='histogram', figsize=(10, 7));
target = 'price'

cat_features = ['cut', 'color', 'clarity']

num_features = ['carat', 'depth', 'table', 'x', 'y', 'z']



for cat_feat in cat_features:

    train[cat_feat] = train[cat_feat].astype('category')

    test[cat_feat] = test[cat_feat].astype('category')

    

cat_df = pd.get_dummies(train[cat_features])

num_df = train.loc[:,num_features]

train_df = pd.concat([cat_df, num_df], axis=1)



cat_df = pd.get_dummies(test[cat_features])

num_df = test.loc[:,num_features]

test_df = pd.concat([cat_df, num_df], axis=1)





features = list(cat_df.columns) + list(num_df.columns)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(train_df.loc[:,features].values)

y = train[target]
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X=X, y=y)
X_test = scaler.transform(test_df.loc[:,features].values)

y_hat = model.predict(X_test).clip(0, 30000)

submission = pd.DataFrame({'id': test['id'], 'price': y_hat})

submission.to_csv('submission.csv', index=False)