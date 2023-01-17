"""Importing libraries and stuff"""
# Author: Fernando-Lopez-Velasco

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
import category_encoders as ce
from sklearn import preprocessing
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
"""Loading files as a pandas dataframe"""

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
"""Splitting data"""

Y = train['Survived'].copy() # We extract the target vector
Xtrain = train.drop(['Survived','PassengerId', 'Name'], axis=1) # Drop some columns which are not useful
Xtest = test.drop(['PassengerId','Name'], axis=1)
Xtest.head()
"""First we split data in categorical and no categorical values"""

train_category = Xtrain.select_dtypes(include=['object']).copy()
test_category = Xtest.select_dtypes(include=['object']).copy()
train_float = Xtrain.select_dtypes(exclude=['object']).copy()
test_float = Xtest.select_dtypes(exclude=['object']).copy()
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(train_float)
Xtrain_float= imp.transform(train_float)
Xtest_float = imp.transform(test_float)
"""Declaring the object of BackwardDifferenceEncoder and fitting"""

encoder = ce.BackwardDifferenceEncoder(cols=['Sex', 'Ticket','Cabin','Embarked'])
encoder.fit(train_category)
"""Transforming data"""

Xtrain_category = encoder.transform(train_category)
Xtest_category = encoder.transform(test_category)
"""We need to drop some columns, this is because the transformation have generated extra columns"""

train_cols = Xtrain_category.columns
test_cols = Xtest_category.columns
flag = 0
cols_to_drop = []
for i in train_cols:
    for j in test_cols:
        if i == j:
            flag = 1
    if flag == 0:
        cols_to_drop.append(i)
    else:
        flag = 0
"""Dropping columns"""

Xtrain_category = Xtrain_category.drop(cols_to_drop, axis=1)
print(Xtrain_category.shape)
print(Xtest_category.shape)
"""Intialize the object imputer"""

imp.fit(Xtrain_category)
"""Transforming data"""

Xtrain_category = pd.DataFrame(imp.transform(Xtrain_category), columns = Xtrain_category.columns)
Xtest_category = pd.DataFrame(imp.transform(Xtest_category), columns = Xtest_category.columns)
"""Initializing and fiting"""

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(Xtrain_float)
"""Scaling"""

Xtrain_float = pd.DataFrame(min_max_scaler.transform(Xtrain_float), columns = train_float.columns)
Xtest_float = pd.DataFrame(min_max_scaler.transform(Xtest_float), columns = test_float.columns)
Xtest_float.head()
Xtest_category.head()
"""As we have two kinds of datasets which are categorical and not categorical data, we need to concatenate both"""

Xtrain = pd.concat([Xtrain_float,Xtrain_category], axis=1)
Xtest = pd.concat([Xtest_float,Xtest_category], axis=1)
"""Initializing the XBoost classifier"""

model = xgb.XGBClassifier(n_estimators=2000, max_depth=5, learning_rate=0.1)
"""Fitting"""

model.fit(Xtrain, Y)
"""Making a prediction"""

Ypred = model.predict(Xtest)
"""Saving data"""
Ypred = pd.DataFrame({'Survived':Ypred})
prediction = pd.concat([test['PassengerId'], Ypred], axis=1)
prediction.to_csv('predictions_xboost.csv', sep=',', index=False)
prediction.head()
