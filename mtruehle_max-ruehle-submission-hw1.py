import numpy as np

import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train.head(5)
train.info()
train = train.dropna(axis = 'columns')
train1 = pd.concat([train, pd.get_dummies(train['LotShape'], prefix = 'LotShape'), pd.get_dummies(train['LandContour'], prefix = 'LandContour')], axis = 1)
train1.head(5)
train1.info()
train_y = train1.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'OverallCond', 'YearRemodAdd', 'TotRmsAbvGrd', 'PoolArea',

                  'LotShape_IR1', 'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg', 

                  'LandContour_Bnk', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl']
train_X = train1[predictor_cols]
svm_clf = Pipeline([

    ("scaler", StandardScaler()),

    ("linear_svc", LinearSVC(C=1, loss="hinge")),

    ])



svm_clf.fit(train_X,train_y)
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
test = test.dropna(axis = 'columns')
test1 = pd.concat([test, pd.get_dummies(test['LotShape'], prefix = 'LotShape'), pd.get_dummies(test['LandContour'], prefix = 'LandContour')], axis = 1)
test_X = test1[predictor_cols]
predicted_prices = svm_clf.predict(test_X)
print(predicted_prices)
my_submission2 = pd.DataFrame({'Id': test1.Id, 'SalePrice': predicted_prices})
my_submission2.to_csv('submission2-SVR_RuehleMax.csv', index=False)