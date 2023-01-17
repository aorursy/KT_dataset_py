import numpy as np
import pandas as pd

data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

target = data.SalePrice
predictors = data.drop(['SalePrice'], axis=1)

# разделим датафреймы на категориальные и численные признаки
numeric_predictors = predictors.select_dtypes(exclude=['object'])
categorial_predictors = predictors.select_dtypes(include=['object'])
numeric_test = test.select_dtypes(exclude=['object'])
categorial_test = test.select_dtypes(include=['object'])

print(numeric_predictors.shape, categorial_predictors.shape, numeric_test.shape, categorial_test.shape)
# Удалим из категориальных признаков те столбцы, которые содержат пустоты
cols_with_missing = [col for col in categorial_test.columns if categorial_test[col].isnull().any() or categorial_predictors[col].isnull().any()]
reduced_categorial_predictors = categorial_predictors.drop(cols_with_missing, axis = 1)
reduced_categorial_test = categorial_test.drop(cols_with_missing, axis = 1)
print(reduced_categorial_predictors.shape, reduced_categorial_test.shape)
# В датафреймы, содержащие численные признаки сделаем импьютацию.

# копируем датафреймы, видимо, чтобы иметь возможность вернуться к исходным тренировочным и валидационным вариантам.
imputed_predictors_plus = numeric_predictors.copy()
imputed_test_plus = numeric_test.copy()

# добавляем столбцы с булевыми переменными, показывающими на каких строках существуют пустоты в соответствующих столбцах для последующего использования Imputer
cols_with_missing = [col for col in numeric_predictors.columns if numeric_predictors[col].isnull().any()]
for col in cols_with_missing:
    imputed_predictors_plus[col + ' was missing'] = imputed_predictors_plus[col].isnull()
    imputed_test_plus[col + ' was missing'] = imputed_test_plus[col].isnull()

print(imputed_predictors_plus.shape, imputed_test_plus.shape)

# вставляем новые значения в пустые ячейки, делая их, внезапно, непустыми
my_imputer = Imputer()
imputed_predictors_plus = my_imputer.fit_transform(imputed_predictors_plus)
imputed_test_plus = my_imputer.fit_transform(imputed_test_plus)

print(imputed_predictors_plus.shape, imputed_test_plus.shape)
#new_predictors = pd.concat([imputed_predictors_plus, categorial_predictors], axis=1)
print(type(categorial_predictors))
# разделим исходный датасет на тренировочную и валидационную части
train_X, val_X, train_y, val_y = train_test_split(predictors,target,random_state=0)
one_hot_predictors = pd.get_dummies(imputed_train_X_plus)
one_hot_test_X = pd.get_dummies(imputed_test_X_plus)
final_train, final_test = one_hot_predictors.align(one_hot_test_X, join='left', axis=1)
print(final_train.info(), final_test.info())
my_model = RandomForestRegressor()
my_model.fit(imputed_train_X_plus,train_y)
val_predictions = my_model.predict(imputed_val_X_plus)

# средняя абсолютная ошибка снизилась по сравнению с базовым предсказанием с 24 до 19 тысяч
print(mean_absolute_error(val_y,val_predictions))
predicted_prices = my_model.predict(imputed_test_X_plus)
my_submission = pd.DataFrame({'Id':test.Id, 'SalePrice':predicted_prices})
my_submission.to_csv('submission.csv',index=False)
