import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Прочитать набор данных из файла
melbourne_file_path = '../input/melb_data.csv'
data = pd.read_csv(melbourne_file_path)
melbourne_data = data.dropna(0, subset=['Price'])

type_data = pd.get_dummies(melbourne_data.Type)
melbourne_data = melbourne_data.join(type_data)
melbourne_data = melbourne_data.select_dtypes(exclude=['object'])
melbourne_data['Type_Cat'] = 4 * melbourne_data['h'] + 2 * melbourne_data['t'] + melbourne_data['u']
melbourne_data = melbourne_data.drop(columns=['h', 't', 'u'])
melbourne_data = melbourne_data.select_dtypes(exclude=['object'])

fields = ['Price', 'Rooms', 'Bathroom', 'Distance', 'Landsize', 'Lattitude', 'Longtitude', 'Type_Cat', 'YearBuilt', 'Car']
melbourne_data = melbourne_data[fields]
melbourne_data.dropna(axis = 1)
data.describe()
# Определить модель поиска цены (Price) по набору столбцов 

y = melbourne_data.Price
fields.remove('Price')
melbourne_features = fields
X = melbourne_data[melbourne_features]

# Создать набор данных для обучения и валидации
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

print('Mean price is', val_y.mean())
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_train_X = pd.DataFrame(columns = melbourne_features, data = my_imputer.fit_transform(train_X))
imputed_val_X = pd.DataFrame(columns = melbourne_features, data = my_imputer.transform(val_X))
# Построение модели по алгоритму Decision Tree и расчет среднеквадратичного отклонения
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Создание и обучение модели на train наборе данных
dtr_melbourne_model = DecisionTreeRegressor(max_leaf_nodes=500)
%time dtr_melbourne_model.fit(imputed_train_X, train_y)
# Тестирование модели на validation наборе данных
%time dtr_preds = dtr_melbourne_model.predict(imputed_val_X)

print('Desision tree prediction mean error:', mean_absolute_error( val_y, dtr_preds))
# Построение модели по алгоритму Random Forest и расчет среднеквадратичного отклонения
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

rfg_forest_model = RandomForestRegressor(random_state = 3)
%time rfg_forest_model.fit(imputed_train_X, train_y)
%time rfg_preds = rfg_forest_model.predict(imputed_val_X)

print('Random Forest prediction mean error:', mean_absolute_error(val_y, rfg_preds))

# Алгоритм Random Forest дает ошибку 218482 и это меньше, чем наилучший вариант Decision Tree с результатом 261718
# Однако модель обсчитывается дольше
from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=2000, learning_rate=0.15)
xgb_model.fit(train_X, train_y,early_stopping_rounds=15, eval_set=[(val_X, val_y)], verbose=False)
xgb_preds = xgb_model.predict(val_X)

print('Gradient Boost prediction mean error:', mean_absolute_error(val_y, xgb_preds))
#83.83 XGBoost
print('Точность прогноза Decision Tree', 100 - (100 * mean_absolute_error(val_y, dtr_preds) / val_y.mean()), '%')
print('Точность прогноза Random Forest', 100 - (100 * mean_absolute_error(val_y, rfg_preds) / val_y.mean()), '%')
print('Точность прогноза Gradient boost', 100 - (100 * mean_absolute_error(val_y, xgb_preds) / val_y.mean()), '%')