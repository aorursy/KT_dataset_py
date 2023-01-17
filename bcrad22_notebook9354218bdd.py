import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn import linear_model
# Import the data



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_features = train_df[['MSSubClass',

                           'LotFrontage',

                           'LotArea',

                           'OverallQual',

                           'OverallCond',

                           'YearBuilt',

                           'YearRemodAdd',

                           'MasVnrArea',

                           'BsmtFinSF1',

                           'BsmtFinSF2',

                           'TotalBsmtSF',

                           '1stFlrSF',

                           '2ndFlrSF',

                           'LowQualFinSF',

                           'GrLivArea',

                           'BsmtFullBath',

                           'BsmtHalfBath',

                           'FullBath',

                           'HalfBath',

                           'BedroomAbvGr',

                           'KitchenAbvGr',

                           'TotRmsAbvGrd',

                           'Fireplaces',

                           'GarageYrBlt',

                           'GarageCars',

                           'GarageArea',

                           'WoodDeckSF',

                           'OpenPorchSF',

                           'EnclosedPorch',

                           '3SsnPorch',

                           'ScreenPorch',

                           'PoolArea',

                           'MiscVal',

                           'MoSold',

                           'YrSold',

                           'SalePrice']]

train_features = train_features.astype('float')

train_features = train_features.fillna(-9999)

train_labels = train_features['SalePrice']

train_features = train_features.drop(train_features.columns[[19]],1)

#print(train_features.dtypes)

#print(train_df.dtypes[60:])
test_features = test_df[['MSSubClass',

                           'LotFrontage',

                           'LotArea',

                           'OverallQual',

                           'OverallCond',

                           'YearBuilt',

                           'YearRemodAdd',

                           'MasVnrArea',

                           'BsmtFinSF1',

                           'BsmtFinSF2',

                           'TotalBsmtSF',

                           '1stFlrSF',

                           '2ndFlrSF',

                           'LowQualFinSF',

                           'GrLivArea',

                           'BsmtFullBath',

                           'BsmtHalfBath',

                           'FullBath',

                           'HalfBath',

                           'BedroomAbvGr',

                           'KitchenAbvGr',

                           'TotRmsAbvGrd',

                           'Fireplaces',

                           'GarageYrBlt',

                           'GarageCars',

                           'GarageArea',

                           'WoodDeckSF',

                           'OpenPorchSF',

                           'EnclosedPorch',

                           '3SsnPorch',

                           'ScreenPorch',

                           'PoolArea',

                           'MiscVal',

                           'MoSold',

                           'YrSold']]

test_features = test_features.astype('float')

test_features = test_features.fillna(-99999)
# Scale data

train_features = preprocessing.scale(train_features)

test_features = preprocessing.scale(test_features)

#print(test_features)
X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=20)

#print(len(X_train))

#print(len(X_test))

#print(len(y_train))

#print(len(y_test))

#print(y_test.head())
clf = MLPClassifier()

clf.fit(train_features, train_labels)
# Split test score



#score = clf.score(X_test, y_test)

#print(score)

#print(y_test.head())
prediction = clf.predict(test_features)

print(prediction)
Id = []

for i in range(len(test_features)):

    Id.append(i+1461)



submission = pd.DataFrame({

    'Id': Id,

    'SalePrice': prediction

})

print(submission)

print(len(submission))
submission.to_csv('kaggle_house.csv', index=False)