import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor



train = pd.read_csv('../input/home-data-for-ml-course/train.csv')

test = pd.read_csv('../input/home-data-for-ml-course/test.csv')



#model = RandomForestRegressor(n_estimators=100, criterion="mse", random_state=50)

model = XGBRegressor(n_estimators=700, learning_rate = 0.05)



features = ['OverallQual','OverallCond', 'Neighborhood', 'Condition1', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'CentralAir', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'KitchenQual', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'GarageArea', 'WoodDeckSF',  'SaleType', 'SaleCondition']

features += ['BsmtFullBath', 'BsmtHalfBath', 'HalfBath']



y = train['SalePrice']

X = train[features]



X = X.drop('Condition1', axis=1)

#X = X.drop('CentralAir', axis=1)

#X = X.drop('KitchenQual', axis=1)

X['12FlrSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']

X['No2ndFlr']=(X['2ndFlrSF']==0)

X['NoBsmt']=(X['TotalBsmtSF']==0)

X = X.drop('TotalBsmtSF', axis=1)

X = X.drop('1stFlrSF', axis=1)

X = X.drop('2ndFlrSF', axis=1)

#X = X.drop('GarageCars', axis=1)

X = X.drop('SaleType', axis=1)#

#X = X.drop('SaleCondition', axis=1)

X['AboutBath'] = X['BsmtFullBath'] + X['FullBath'] + X['BsmtHalfBath'] + X['HalfBath']

X = X.drop('FullBath', axis=1)

X = X.drop('BsmtFullBath', axis=1)

X = X.drop('BsmtHalfBath', axis=1)

X = X.drop('HalfBath', axis=1)

X['YearSum'] = X['YearBuilt']+X['YearRemodAdd']

X['hadRemod'] = (X['YearBuilt']==X['YearRemodAdd']) ^ 1

#X = X.drop('GarageYrBlt', axis=1)

#X = X.drop('YearBuilt', axis=1)

X = X.drop('YearRemodAdd', axis=1)



train_X, val_X, train_y, val_y = train_test_split(X,y)

print('settings completed')
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('ggplot')

print("Setup Complete")
from sklearn.metrics import mean_squared_log_error



def cal_err(gmodel, train_X, train_y, val_X, val_y):

    gmodel.fit(train_X, train_y)

    preds = gmodel.predict(val_X)

    err = mean_squared_log_error(val_y, preds)

    return err
s = (train_X.dtypes == 'object')

obj_cols = list(s[s].index)

num_cols = list(set(train_X.columns) - set(obj_cols))



print("Categorical variables:")

print(obj_cols)

print("Numerical variabels:")

print(num_cols)
# Shape of training data (num_rows, num_columns)

print(train_X.shape)



# Number of missing values in each column of training data

missing_val_count_by_column = (train_X.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
target_rm = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']



# Make copy to avoid changing original data (when imputing)

X_train_plus = train_X.copy()

X_valid_plus = val_X.copy()



# Make new columns indicating whether the house has garage

train_X['hasGarage'] = X_train_plus['GarageType'].isnull() ^ 1

val_X['hasGarage'] = X_valid_plus['GarageType'].isnull() ^ 1



train_X['GarageYrBlt'].fillna(train_X['YearBuilt'], inplace=True)

val_X['GarageYrBlt'].fillna(train_X['YearBuilt'], inplace=True)

train_X.drop('YearBuilt', axis=1)

val_X.drop('YearBuilt',axis=1)



train_X = train_X.drop(target_rm, axis=1)

val_X = val_X.drop(target_rm, axis=1)



obj_cols = list(set(obj_cols) - set(target_rm))
object_nunique = list(map(lambda col: train_X[col].nunique(), obj_cols))

d = dict(zip(obj_cols, object_nunique))



# Print number of unique entries by column, in ascending order

sorted(d.items(), key=lambda x: x[1])
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



target = []

for item in d:

    if d[item] < 10:

        target.append(item)



OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_X[target]))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(val_X[target]))



# One-hot encoding removed index; put it back

OH_cols_train.index = train_X.index

OH_cols_valid.index = val_X.index



# Remove categorical columns (will replace with one-hot encoding)

num_X_train = train_X.drop(target, axis=1)

num_X_valid = val_X.drop(target, axis=1)



# Add one-hot encoded columns to numerical features

train_X = pd.concat([num_X_train, OH_cols_train], axis=1)

val_X = pd.concat([num_X_valid, OH_cols_valid], axis=1)



# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

train_X['Neighborhood'] = label_encoder.fit_transform(train_X['Neighborhood'])

val_X['Neighborhood'] = label_encoder.transform(val_X['Neighborhood'])



con_encoder = LabelEncoder()

train_X['OverallCond'] = con_encoder.fit_transform(train_X['OverallCond'])

val_X['OverallCond'] = con_encoder.transform(val_X['OverallCond'])
# Nothing to do yet
print(cal_err(model, train_X, train_y, val_X, val_y))

#while True:

#    model = RandomForestRegressor(n_estimators=100, criterion="mse", random_state=50)

#    res = cal_err(model, train_X, train_y, val_X, val_y)

#    print(res)

#    if res < 0.0185:

#        break

#print(res)
test[features].isnull().sum()
test['TotalBsmtSF'].fillna(value=test['TotalBsmtSF'].mode()[0], inplace=True)

test['KitchenQual'].fillna(value=test['KitchenQual'].mode()[0], inplace=True)

test['SaleType'].fillna(value=test['SaleType'].mode()[0], inplace=True)

test['GarageCars'].fillna(value=test['GarageCars'].mean(), inplace=True)

test['GarageArea'].fillna(value=test['GarageArea'].mean(), inplace=True)



test['BsmtFullBath'].fillna(value=test['BsmtFullBath'].median(), inplace=True)

test['BsmtHalfBath'].fillna(value=test['BsmtHalfBath'].median(), inplace=True)



test_X = test[features]
test_X = test_X.drop('Condition1', axis=1)

#test_X = test_X.drop('CentralAir', axis=1)

#test_X = test_X.drop('KitchenQual', axis=1)

test_X['12FlrSF'] = test_X['TotalBsmtSF'] + test_X['1stFlrSF'] + test_X['2ndFlrSF']

test_X['No2ndFlr']=(test_X['2ndFlrSF']==0)

test_X['NoBsmt']=(test_X['TotalBsmtSF']==0)

test_X = test_X.drop('TotalBsmtSF', axis=1)

test_X = test_X.drop('1stFlrSF', axis=1)

test_X = test_X.drop('2ndFlrSF', axis=1)

#test_X = test_X.drop('GarageCars', axis=1)

test_X = test_X.drop('SaleType', axis=1)

#test_X = test_X.drop('SaleCondition', axis=1

test_X['AboutBath'] = test_X['BsmtFullBath'] + test_X['FullBath'] + test_X['BsmtHalfBath'] + test_X['HalfBath']

test_X = test_X.drop('FullBath', axis=1)

test_X = test_X.drop('BsmtFullBath', axis=1)

test_X = test_X.drop('BsmtHalfBath', axis=1)

test_X = test_X.drop('HalfBath', axis=1)

test_X['YearSum'] = test_X['YearBuilt']+test_X['YearRemodAdd']

test_X['hadRemod'] = (test_X['YearBuilt']==test_X['YearRemodAdd']) ^ 1

#X = X.drop('GarageYrBlt', axis=1)



test_X = test_X.drop('YearRemodAdd', axis=1)
target_rm = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']



# Make copy to avoid changing original data (when imputing)

X_plus = X.copy()

test_X_plus = test_X.copy()



# Make new columns indicating whether the house has garage

X['hasGarage'] = X['GarageType'].isnull() ^ 1

test_X['hasGarage'] = test_X['GarageType'].isnull() ^ 1



X['GarageYrBlt'].fillna(X['YearBuilt'], inplace=True)

test_X['GarageYrBlt'].fillna(test_X['YearBuilt'], inplace=True)

X = X.drop('YearBuilt', axis=1)

test_X = test_X.drop('YearBuilt', axis=1)



X = X.drop(target_rm, axis=1)

test_X = test_X.drop(target_rm, axis=1)



obj_cols = list(set(obj_cols) - set(target_rm))



OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_X = pd.DataFrame(OH_encoder.fit_transform(X[target]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(test_X[target]))



# One-hot encoding removed index; put it back

OH_cols_X.index = X.index

OH_cols_test.index = test_X.index



# Remove categorical columns (will replace with one-hot encoding)

num_X = X.drop(target, axis=1)

num_X_test = test_X.drop(target, axis=1)



# Add one-hot encoded columns to numerical features

X = pd.concat([num_X, OH_cols_X], axis=1)

test_X = pd.concat([num_X_test, OH_cols_test], axis=1)



# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

X['Neighborhood'] = label_encoder.fit_transform(X['Neighborhood'])

test_X['Neighborhood'] = label_encoder.transform(test_X['Neighborhood'])



con_encoder = LabelEncoder()

X['OverallCond'] = con_encoder.fit_transform(X['OverallCond'])

test_X['OverallCond'] = con_encoder.transform(test_X['OverallCond'])
#test[['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']].isnull().sum()
model.fit(X,y)



preds = model.predict(test_X)
submission = pd.read_csv('../input/home-data-for-ml-course/sample_submission.csv')

submission.head()



submission['SalePrice'] = preds



submission.to_csv('submission_5.csv', index=False)

print('Completed')