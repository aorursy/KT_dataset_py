# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

#Импортируем необходимые пакеты

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.impute import SimpleImputer



#Открываем файлы с данными

train_data_file_path = '/kaggle/input/train.csv'

train_data = pd.read_csv(train_data_file_path) 

test_data_file_path = '/kaggle/input/test.csv'

test_data = pd.read_csv(test_data_file_path) 



# y - это цена продажи, которую будем прогнозировать 

y = train_data.SalePrice



# X - это признаки, на основании которых будет обучаться модель

features = ['MSSubClass','MSZoning','LotFrontage','LotArea','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2',

'BldgType','HouseStyle','OverallQual','OverallCond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual',

'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC',

'CentralAir','Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual',

'TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive','WoodDeckSF',

'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','PoolQC','Fence','MiscFeature','MiscVal','MoSold','YrSold','SaleType','SaleCondition']



#Создаем обучающий и тестовый наборы данных

X_train = train_data[features]

X_test  = test_data[features]



#Выполняем бинарное кодирование категориальных данных в дополнительных столбцах

one_hot_encoded_training_predictors = pd.get_dummies(X_train)

one_hot_encoded_test_predictors = pd.get_dummies(X_test)



#Выравниваем количество столбцов в обучающем и тестовом наборах данных

final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,join='left', axis=1)



#Подготоавливаем обучающий и тестовый наборы данных для заполнения неопределенных значений

imputed_X_train_plus = final_train.copy()

imputed_X_test_plus = final_test.copy()

cols_with_missing = (col for col in final_train.columns 

                                 if final_train[col].isnull().any())

for col in cols_with_missing:

    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()

    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()



#Заполняем неопределенные значения статистическими данными в обучающем и тестовом наборах данных

my_imputer = SimpleImputer()

imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)

imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)



#Разбиваем обучающий набор на подмножества дял обучения (train_X,train_y) и оценки (val_X,val_y)

train_X, val_X, train_y, val_y = train_test_split(imputed_X_train_plus, y, random_state=1)



#Создаем модель 

rf_model = RandomForestRegressor(random_state=1)



#Подаем на вход модели данные для обучения

rf_model.fit(train_X, train_y)



#Рассчитываем среднюю абсолютную ошибку модели, используя данные для оценки модели (val_X,val_y) 

rf_val_mae = mean_absolute_error(rf_model.predict(val_X), val_y)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))



#Запускаем модель на тестовом наборе данных

y = rf_model.predict(imputed_X_test_plus)



#Получаем рассчитанные прогнозные значения

print(y)

my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': y})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)