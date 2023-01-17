# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler

# выбранная модель
from sklearn.ensemble import GradientBoostingRegressor

# Дата для работоспосбности кода в 2020+
from datetime import datetime

# Метрика
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import KFold

# Magic commands
%matplotlib inline 
# вывод графики в ноутбук
%config InlineBackend.figure_format = 'svg' 
# более четкое отображение, формат фала фигуры svg
TRAIN_DATASET_PATH = '/kaggle/input/realestatepriceprediction/train.csv'
TEST_DATASET_PATH = '/kaggle/input/realestatepriceprediction/test.csv'
df_train = pd.read_csv(TRAIN_DATASET_PATH) # загружаем тренировочный датасет в датафрейм df_train
df_test = pd.read_csv(TEST_DATASET_PATH) # загружаем тестовый датасет в датафрейм df_test
df_train.dtypes # Типы признаков
# Превратим ID и DistrictId в строку, так как по сути это название района.
df_train['Id'] = df_train['Id'].astype(str) 
df_train['DistrictId'] = df_train['DistrictId'].astype(str)
df_test['Id'] = df_test['Id'].astype(str) 
df_test['DistrictId'] = df_test['DistrictId'].astype(str)
# Обзор целевой переменной

plt.figure(figsize = (10, 3))

df_train['Price'].hist(bins=30)
plt.ylabel('Count')
plt.xlabel('Price')

plt.title('Target distribution')
plt.show()
df_train.describe() # Обзор всех числовых признаков
df_train.select_dtypes(include='object').columns.tolist() # Категориальные признаки
class Cleaner():
    district_reach = []
    
    def __init__(self):
        #self.med_price_by_district = None
        print(type(self))

    
    def transform(self, X):
        
        # Добавляю признак неадекватности комнат
        X['Rooms_outlier'] = 0
        X.loc[(X['Rooms'] == 0) | (X['Rooms'] >= 6), 'Rooms_outlier'] = 1
        
        # Площадь более 280 делю на 10, площадь менее 16 умножаю на 10
        X.loc[(X['Square'] > 280), 'Square'] = ((X.loc[(X['Square'] > 280), 'Square']) / 10)
        X.loc[(X['Square'] < 16), 'Square'] = ((X.loc[(X['Square'] < 16), 'Square']) * 10)
        
        
        # заменяю неадекватное кол-во комнат на 1,2,3,4 в зависимости от площади
        missing_rooms_data = X.loc[ (X['Rooms'] ==0) | (X['Rooms'] > 6), ['Square', 'Rooms']]
        if missing_rooms_data['Square'].count() !=0:
            for i in missing_rooms_data.index:
                if (missing_rooms_data.loc[(i),'Square']) < 43:
                    (missing_rooms_data.loc[(i),'Rooms']) = 1
                elif (missing_rooms_data.loc[(i),'Square']) < 60:
                    (missing_rooms_data.loc[(i),'Rooms']) = 2
                elif (missing_rooms_data.loc[(i),'Square']) < 100:
                    (missing_rooms_data.loc[(i),'Rooms']) = 3
                elif (missing_rooms_data.loc[(i),'Square']) > 100:
                    (missing_rooms_data.loc[(i),'Rooms']) = 4
            X.loc[ (X['Rooms'] ==0) | (X['Rooms'] > 6), 'Rooms'] = missing_rooms_data['Rooms']
        
        # Делаем выборку наблюдений, где площадь кухни больше площади всей квартиры, либо <5м, либо >50 
        var_kitchen = X.loc[(X['Square'] < X['KitchenSquare']) | 
                    (X['KitchenSquare'] < 5) | 
                    (X['KitchenSquare'] > 50), 
                    ['KitchenSquare','Square']]
        
        # если в выборку попали наблюдения, то умножаем площадь квартиры на 0.14 и записываем в KitchenSquare
        if var_kitchen['Square'].count() !=0:
            #print('ok')
            for i in var_kitchen.index:
                var_kitchen.loc[(i), 'KitchenSquare'] = round((var_kitchen.loc[(i), 'Square']) * 0.14)
        X.loc[(X['Square'] < X['KitchenSquare']) | 
                    (X['KitchenSquare'] < 5) | 
                    (X['KitchenSquare'] > 50), 
                    'KitchenSquare'] = var_kitchen['KitchenSquare']
        
        # Healthcare_1 удаляем так как много пропусков
        if 'Healthcare_1' in X.columns:
            X.drop('Healthcare_1', axis=1, inplace=True)
        
        
        #  Преобразовываем экстремальные данные площади жилой
        X["LifeSquare"].fillna(0, inplace=True)
        missing_lifesquare = X.loc[(X['LifeSquare'] == 0) | 
                                   (X['LifeSquare'] > X['Square']), 
                                   ['Square', 'LifeSquare']]
        if missing_lifesquare['LifeSquare'].count() !=0:
            for i in missing_lifesquare.index:
                missing_lifesquare.loc[(i), 'LifeSquare'] = round((missing_lifesquare.loc[(i), 'Square']) * 0.64)
        X.loc[(X['LifeSquare'] == 0) | 
              (X['LifeSquare'] > X['Square']), 
              'LifeSquare'] = missing_lifesquare['LifeSquare']
        
        
        #if 'HouseYear' in X.keys():
        #    X['age_house'] = (current_year - X['HouseYear'])
        #var_house_age = X.loc[(X['age_house'] < 0) |(X['age_house'] > 110) ,'age_house']
        
        #HouseYear
        current_year = datetime.now().year
        X.loc[X['HouseYear'] > current_year, 'HouseYear'] = current_year
        
        
        # Заменяем в наблюдениях по признаку Shops_2  буквы A и B на цифры 0 и 1
        binary_to_numbers = {"B": 1, "A": 0,}
        X["Shops_2"] = X["Shops_2"].replace(binary_to_numbers) 

        # Заменяем в наблюдениях по признаку Ecology_2  буквы A и B на цифры 0 и 1
        X["Ecology_2"] = X["Ecology_2"].replace(binary_to_numbers)

        # Заменяем в наблюдениях по признаку Ecology_3  буквы A и B на цифры 0 и 1
        X["Ecology_3"] = X["Ecology_3"].replace(binary_to_numbers)
        
        return X
    
    # DistrictReach
    def districtReach(self, X):   
        if 'Price' in X.keys():
            self.district_reach = X.groupby('DistrictId').median()[['Price']]\
                                .rename(columns={'Price':'district_median_price'})

            X = X.merge(self.district_reach, on=['DistrictId'], how='left')
        
        return X
    
    
    def addReach(self, X):
        X['district_median_price'] = X.merge(self.district_reach, on=['DistrictId'], how='left')
        
        return X
X_clean = Cleaner()
X = X_clean.transform(df_train)
X = X_clean.districtReach(df_train)
var_district_reach = X_clean.district_reach
var_district_reach
y = X[['Price']]
X = X.drop(['Price','Id'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.38, shuffle=True, random_state=66)
model_gbt = GradientBoostingRegressor(random_state=66, n_estimators=334)
model_gbt.fit(X_train, y_train)
y_train_preds = model_gbt.predict(X_train)
y_valid_preds = model_gbt.predict(X_valid)
cv_score = cross_val_score(model_gbt, X, y, scoring='r2', cv=KFold(n_splits=3, shuffle=True, random_state=66))
cv_score
feature_importances = pd.DataFrame(zip(X_train.columns, model_gbt.feature_importances_), 
                                   columns=['feature_name', 'importance'])

feature_importances.sort_values(by='importance', ascending=False)
df_test.head()
X_test = X_clean.transform(df_test)
X_test = X_test.merge(var_district_reach, on=['DistrictId'], how='left')
predictions = pd.DataFrame(index = df_test['Id'])
X_test.drop('Id', axis=1, inplace=True)
X_test.isnull().sum()
X_test['district_median_price'].fillna(X_test['district_median_price'].median(), inplace=True)
y_pred = model_gbt.predict(X_test)
predictions['Price'] = y_pred
predictions
predictions.to_csv('try_test_14.csv', sep=',')
#!kaggle competitions submit -c realestatepriceprediction -m "ILYA Shubenko" -f try_test_14.csv