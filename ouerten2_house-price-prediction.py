import pandas as pd





train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Veriyi predict ve validation olarak ayır

predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'YrSold', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd','MSSubClass','OverallQual','OverallCond','YearRemodAdd',]

train_y = train.SalePrice





print(train_y.head())

train_X = train[predictors]

print(train_X.head())
# Model tanımlama

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

iowa_model = RandomForestRegressor()

iowa_model.fit(train_X, train_y)

## ortalama mutlak hata hesabı

tahmin=iowa_model.predict(train_X)

mean_absolute_error(train_y,tahmin)

test_X = test[predictors]

pred_y = iowa_model.predict(test_X)

print(pred_y)



mysub = pd.DataFrame({'Id': test.Id, 'SalePrice': pred_y})

print(mysub.head())

mysub.to_csv('my_submission.csv', index=False)
