import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex7 import *





import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor





df = pd.read_csv('../input/train.csv')



y = df.SalePrice



features = new_features = ['OpenPorchSF','LotArea','BsmtFinSF1','BsmtUnfSF','BsmtFullBath','HalfBath','BedroomAbvGr','Fireplaces','WoodDeckSF','OpenPorchSF','ScreenPorch',

                     'OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','2ndFlrSF',

               'GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']







X2 = df[new_features]



y= df['SalePrice']



X_train, X_test, y_train, y_test = train_test_split(X2, y, random_state=1)







dtree = DecisionTreeRegressor(random_state=1)

dtree.fit(X_train,y_train)

dtree_pred = dtree.predict(X_test)

dtree_mae = mean_absolute_error(dtree_pred, y_test)

print("Validation MAE using Decision Tress: {:,.0f}".format(dtree_mae))





rfr = RandomForestRegressor(random_state=1)

rfr.fit(X_train,y_train)

rfr_pred= rfr.predict(X_test)

rfr_mae = mean_absolute_error(rfr_pred, y_test)

print("Validation MAE using Random Forests: {:,.0f}".format(rfr_mae))
rfr_full = RandomForestRegressor(random_state=1)

rfr_full.fit(X2,y)
test_data = pd.read_csv("../input/test.csv")





test_X = test_data[new_features]



test_X.info()



test_X['BsmtFinSF1'] = test_X['BsmtFinSF1'].fillna(test_X['BsmtFinSF1'].mean())

test_X['BsmtUnfSF'] = test_X['BsmtUnfSF'].fillna(test_X['BsmtUnfSF'].mean())

test_X['BsmtFullBath'] = test_X['BsmtFullBath'].fillna(test_X['BsmtFullBath'].mean())

test_X['TotalBsmtSF'] = test_X['TotalBsmtSF'].fillna(test_X['TotalBsmtSF'].mean())

test_X['GarageCars'] = test_X['GarageCars'].fillna(test_X['GarageCars'].mean())

test_X['GarageArea'] = test_X['GarageArea'].fillna(test_X['GarageArea'].mean())
test_X.info()
test_preds = rfr_full.predict(test_X)

        

output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)