import pandas as pd



train = pd.read_csv('../input/home-data-for-ml-course/train.csv')

test = pd.read_csv('../input/home-data-for-ml-course/test.csv')
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_log_error



def cal_err(model, X, y):

    train_X, val_X, train_y, val_y = train_test_split(X,y)

    print(train_X.columns)

    model.fit(train_X, train_y)

    preds = model.predict(val_X)

    err = mean_squared_log_error(val_y, preds)

    return err
from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor()
features = ['Neighborhood', 'Condition1', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'CentralAir', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'KitchenAbvGr', 'TotRmsAbvGrd', 'KitchenQual', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'SaleType', 'SaleCondition']



y = train['SalePrice']

X = train[features]



for col in X.columns:

    if(X[col].dtype == 'object'):

        X.drop(col, axis=1, inplace=True)

    else:

        X[col].fillna(X[col].mean(), inplace=True)



print(cal_err(model, X, y))
model.fit(X,y)



X = test[features]



for col in X.columns:

    if(X[col].dtype == 'object'):

        X.drop(col, axis=1, inplace=True)

    else:

        X[col].fillna(X[col].mean(), inplace=True)



preds = model.predict(X)
submission = pd.read_csv('../input/home-data-for-ml-course/sample_submission.csv')

submission.head()



submission['SalePrice'] = preds



submission.to_csv('submission_3.csv', index=False)

print('Completed')