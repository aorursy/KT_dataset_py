import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

# Have a quck look at the data

train.head()
train.shape
test.shape
y = train.revenue # This is our target feature

test_id = test.Id # 


train = train.drop(['Id', 'revenue'], axis=1)

test = test.drop(['Id'], axis=1 )
# Look at the data type of each attribute

train.dtypes
cont_features = train.columns[train.dtypes != 'object']

cat_features = train.columns[train.dtypes == 'object']

cat_features
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

X = train[cont_features].values

model = LinearRegression()

result = np.sqrt(-cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error'))

print("%f (%f)" % (result.mean(), result.std()) )
test_X = test[cont_features].values

model = LinearRegression()

model.fit(X, y)

preds = model.predict(test_X)

submission = pd.DataFrame({'Id':test_id, 'Prediction': preds})

submission.to_csv('sub1.csv', index=False) # Score 2416508, 2125 on LB (Post competition deadline)
pd.set_option('precision', 1)

train.describe()
from sklearn.preprocessing import Normalizer

X = train[cont_features].values

scaler = Normalizer().fit(X)

rescaledX = scaler.transform(X)

model = LinearRegression()

result = np.sqrt(-cross_val_score(model, rescaledX, y, cv=10, scoring='neg_mean_squared_error'))

print("%f (%f)" % (result.mean(), result.std()) )
# Second Submission

test_X = test[cont_features].values

rescaled_test_X = scaler.transform(test_X)

model = LinearRegression()

model.fit(rescaledX, y)

preds = model.predict(rescaled_test_X)

submission = pd.DataFrame({'Id':test_id, 'Prediction': preds})

submission.to_csv('sub2.csv', index=False) # Score 2314493, 2095 on LB (Post competition deadline)