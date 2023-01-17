from sklearn.impute import SimpleImputer
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import os
print(os.listdir("../input"))
df = pd.read_csv("../input/train.csv")
test_final = pd.read_csv("../input/test.csv")
test_final_id = test_final
def Filterdataset (dataset):    
    dataset = dataset.copy()
    
    dataset['has_alley'] = df['Alley'].fillna(0).apply(lambda _: 0 if _ == 0 else 1)
    dataset = dataset.fillna(value= {'Alley':'No alley access'})
    dataset['has_BsmtQual'] = df['BsmtQual'].fillna(0).apply(lambda _: 0 if _ == 0 else 1)
    dataset = dataset.fillna(value= {'BsmtQual':'No Basement'})
    dataset['has_BsmtCond'] = df['BsmtCond'].fillna(0).apply(lambda _: 0 if _ == 0 else 1)
    dataset = dataset.fillna(value= {'BsmtCond':'No Basement'})
    dataset['Age'] = dataset['YrSold'] - dataset['YearBuilt']
    dataset['AgeSinceRemode'] = dataset['YrSold'] - dataset['YearRemodAdd']
    dataset['WholeArea'] = (dataset['GrLivArea'] + dataset['GarageArea'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']).astype('float32')
    #dataset = dataset.select_dtypes(include=['float64','int'])
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    dataset = pd.get_dummies(dataset, drop_first=True)
    datasetc = dataset
    dataset = imp_mean.fit_transform(dataset)
    dataset = pd.DataFrame(data = dataset, index = datasetc.index, columns = datasetc.columns)
    if 'Id' in dataset.columns:
        dataset = dataset.drop(['Id'], axis=1)
    if 'SalePrice' in dataset.columns:
        dataset = dataset.drop(['SalePrice'], axis=1)
    return dataset
y = df['SalePrice']
X = df.drop('SalePrice',axis=1)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.10, random_state=0)

X_train = Filterdataset(X_train)
X_test = Filterdataset(X_test)
test_final = Filterdataset(test_final)
columns = []
for c in X_train.columns:
    if c in X_test.columns:
        if c in test_final.columns:
            columns.append(c)
X_train = X_train[columns]
test_final = test_final[columns]
X_test = X_test[columns]
regr = RandomForestRegressor(max_depth=50, random_state=0, n_estimators=100)
regr.fit(X_train, np.log(y_train))
print(len(X_train.columns), len(X_test.columns), len(test_final.columns))
y_pred_train = regr.predict(X_train)
y_pred_test = regr.predict(X_test)
y_pred_final = np.exp(regr.predict(test_final))
print(y_pred_final)
def Mrmse(y_true,y_pred):
    y_true = np.log(y_true)
    #y_pred = np.log(y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return rmse
print(Mrmse(y_train,y_pred_train))
print(Mrmse(y_test,y_pred_test))
my_submission = pd.DataFrame({'Id': test_final_id['Id'], 'SalePrice': y_pred_final})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
#my_submission