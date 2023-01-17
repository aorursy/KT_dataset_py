import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import train_test_split



import os

print(os.listdir("../input"))

training = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

test2 = pd.read_csv('../input/test.csv')



df_train = training.copy()

df_test = test.copy()
target = df_train['SalePrice']  #Целевая Переменная

df_train = df_train.drop('SalePrice', axis=1) #удаляю целевую переменную из набора обучающих данных

df_train['training_set'] = True #добавляю дополнительную переменную к обучающему и тестирующему набору

df_test['training_set'] = False

df_full = pd.concat([df_train, df_test]) #обьединяю оба датафрейма для дальнейшего анализа 


df_full.drop('Id', axis=1, inplace=True) #удаляю столбец столбец ID, для анализа не нужен. 

df_full.shape
df_full.head()
df_full.columns.to_series().groupby(df_full.dtypes).groups #разберу столбцы по их типам (int64, float64, object)
len(df_full.columns)
# Изучу SalePrice целевая переменную

min_price = np.min(target)

max_price = np.max(target)

mean_price = np.mean(target)

median_price = np.median(target)

std_price = np.std(target)



print("Минимальная цена: ${:,.2f}".format(minimum_price))

print("Максимальная цена: ${:,.2f}".format(maximum_price))

print("Средняя цена: ${:,.2f}".format(mean_price))

print("Медиана ${:,.2f}".format(median_price))

print("Стандартное отклонение цен: ${:,.2f}".format(std_price))
corrmat = training.corr()

plt.subplots(figsize=(10,10))

sns.heatmap(corrmat,square=True, cmap="YlGnBu");
#Какие переменные имеют пропущенные значения:

perc_na = (df_full.isnull().sum()/len(df_full))*100

ratio_na = perc_na.sort_values(ascending=False) #сортирую в порядке убывания

missing_data = pd.DataFrame({'Missing Ratio' :ratio_na}) #Отсутствующее соотношение

missing_data.head(20) # Переменные с наибольшим процентом пропущенных значений:
#Переменные (>15% пропущенных значений) вряд ли окажут большое влияние на цены. Удалю эти переменные из учебного набора данных

df_full.drop('PoolQC', axis=1, inplace=True)

df_full.drop('MiscFeature', axis=1, inplace=True)

df_full.drop('Alley', axis=1, inplace=True)

df_full.drop('Fence', axis=1, inplace=True)

df_full.drop('FireplaceQu', axis=1, inplace=True)

df_full.drop('LotFrontage', axis=1, inplace=True)
df_full.shape
df_full.columns
# Перечислим все числовые столбцы

numeric_variables = list(df_full.select_dtypes(include=['int64', 'float64']).columns.values)

# Медиана для этих столбцов и заполните этим значением

df_full[numeric_variables] = df_full[numeric_variables].apply(lambda x: x.fillna(x.median()),axis=0)
# Перечислим все нечисловые столбцы

categorial_variables = list(df_full.select_dtypes(exclude=['int64', 'float64', 'bool']).columns.values)

# Медиана для этих столбцов и заполните этим значением

df_full[categorial_variables] = df_full[categorial_variables].apply(lambda x: x.fillna("None"),axis=0)
#Преобразую категориальные переменные в числовые переменные, используя one-hot encoding.

df_full = pd.get_dummies(df_full)
df_full.shape
df_full.columns
#Прогноз по цене квартиры.

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, make_scorer, mean_squared_error #sklearn

from sklearn.model_selection import train_test_split
#Разделю на обучающий и тренировочный набор данных.

df_train = df_full[df_full['training_set']==True]

df_train = df_train.drop('training_set', axis=1)

df_test = df_full[df_full['training_set']==False]

df_test = df_test.drop('training_set', axis=1)



(df_train.shape, df_test.shape)
#делю на обучающую и тестовую выборку

X_train, X_test, y_train, y_test = train_test_split(df_train, target, random_state=42) 
rf_model = RandomForestRegressor(n_estimators=100 , n_jobs=-1) #n_estimators – число деревьев в ансамбле
#Random Forest Regressor

rf_model.fit(X_train, y_train)

y_predict = rf_model.predict(X_test)
from sklearn.model_selection import GridSearchCV # Поиск по указанным значениям параметров для оценки. Кросс-валидация и подбор гиперпараметров

from sklearn.model_selection import RandomizedSearchCV # Search over specified parameter values for an estimator.

from sklearn.model_selection import ShuffleSplit # Кросс-валидатор случайной перестановки
from time import time

start = time() 

rf_regressor = RandomForestRegressor(random_state=42)

cv_sets = ShuffleSplit(random_state = 4) # Кросс-валидация

parameters = {'n_estimators':[100, 120, 140], 

              'min_samples_leaf':[1, 2, 3], #минимальное число объектов в листе

              'max_depth':[10,15,20]}

scorer = make_scorer(r2_score)

n_iter_search = 10

grid_obj = RandomizedSearchCV(rf_regressor, 

                              parameters, 

                              n_iter = n_iter_search, 

                              scoring = scorer, 

                              cv = cv_sets,

                              random_state= 99)

grid_fit = grid_obj.fit(X_train, y_train)

rf_opt = grid_fit.best_estimator_

end = time()

rf_time = (end-start)/60

print('{0:.2f} минуты'.format(rf_time))
grid_fit.best_params_
rf_opt_preds = rf_opt.predict(X_test)

from sklearn.metrics import mean_absolute_error

print("mean_squared_log_error: " + str(np.sqrt(mean_squared_log_error(rf_opt_preds, y_test))))
#Самые важные параметры по RF 

importances = rf_opt.feature_importances_ 

X_train.columns.values[(np.argsort(importances)[::-1])[:5]] # топ 5
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

my_model.fit(X_train, y_train, early_stopping_rounds=5, 

             eval_set=[(X_test, y_test)], verbose=False)



xgboost_pred = my_model.predict(X_test)

print("mean_squared_error: " + str(np.sqrt(mean_squared_log_error(y_test, xgboost_pred))))
from sklearn.linear_model import Lasso

b_alpha = 0.00099



regr = Lasso(alpha = b_alpha, max_iter=50000)

regr.fit(X_train, y_train)

lasso_pred = regr.predict(X_test)

print("mean_squared_error: " + str(np.sqrt(mean_squared_log_error(y_test, lasso_pred))))
submit = (regr.predict(df_test) + my_model.predict(df_test) + rf_opt.predict(df_test)) / 3

print (submit)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': submit})

my_submission.to_csv('submission-shatov.csv', index=False)