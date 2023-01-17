import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from scipy.stats import normaltest, shapiro, skewtest

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold

from sklearn.metrics import r2_score



%matplotlib inline
## Просмотр данных

def prosmotr(data):

  pd.set_option('display.max_columns', 100) #Размеры таблицы

  pd.set_option('display.max_rows', 100)

  pd.set_option('precision', 2) #Регулируем количество знаков после запятой:

  print('~~~~Содержание данных~~~~\n', data.head())

  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

  print('~~~Размеры данных~~~\n', data.shape)

  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

  print('~~~Названия колонок~~~\n', data.columns)

  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

  print('~~~Информация о данных~~~\n')

  print(data.info())

  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

  print('~~~Наличие пропусков в данных~~~\n', data.isna().sum())

  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

  print('~~~Количество типов в данных~~~')

  print(data.dtypes.value_counts())

  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

  kateg = list(data.select_dtypes(include=['object']).columns) # Делаем список категориальных данных

  print('~~~Категориальные данные~~~~')

  print(kateg)

  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

  chislov_float = list(data.select_dtypes(include=['float64'])) #Делаем список числовых данных float

  print('~~~Числове данные float~~~~')

  print(chislov_float)

  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

  chislov_int = list(data.select_dtypes(include=['int64'])) #Делаем список числовых данных int

  print('~~~Числове данные int~~~~')

  print(chislov_int)

  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

  print('~~~Основные статистические характеристики данных по каждому числовому признаку (типы int64)~~~\n', data.describe(include=['int64']))

  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

  print('~~~Основные статистические характеристики данных по каждому числовому признаку (типы float64)~~~\n', data.describe(include=['float64']))

  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

  print('~~~Cтатистика по нечисловым признакам object ~~~\n', data.describe(include=['object']))

  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#построение графика после обучения

def model_test(model, name, test, valid):

    model_pred = model.predict(test)

    r2 = r2_score(valid, model_pred)

    mse = mean_squared_error(valid, model_pred)

    plt.scatter(valid, (model_pred - valid))

    plt.xlabel("Predicted values")

    plt.ylabel("Real values")

    plt.title(name)

    plt.legend([f'R2= {r2:.4f} and mse= {mse:.0e}'])

    plt.axhline(0, color='red')

    plt.show()
#сравнение лучших моделей

def models_r2(models, test, valid):

    scores = pd.DataFrame(columns=['name', 'r2', 'mse'])

    for name, model in models.items():

        test_pred = model.predict(test)

        r2 = r2_score(valid, test_pred)

        mse = mean_squared_error(valid, test_pred)

        scores = scores.append(

            {'name': name, 'r2': r2, 'mse': mse}, ignore_index=True)

    scores.sort_values('r2', ascending=False, inplace=True)

    return scores
#для преобразования категориальных признаков

def SeriesFactorizer(series):

    series, unique = pd.factorize(series)

    reference = {x: i for x, i in enumerate(unique)}

    print(reference)

    return series, reference
## неправильные года

def df_fix_house_year_manual(df):

    df.loc[df['HouseYear'] == 20052011, 'HouseYear'] = int((2005 + 2011) / 2)

    df.loc[df['HouseYear'] == 4968, 'HouseYear'] = 1968

    return df
df_fix_house_year_manual(df)

df_fix_house_year_manual(test)
df_fix_room(df)

df_fix_room(test)
df_fix_square_manual(df)

df_fix_square_manual(test)
prepare_lifesquare(df)

prepare_lifesquare(test)
prosmotr(df)
df_p = df.drop('Price', axis=1)

y = df.Price.values
numerical = df_p.select_dtypes(exclude = ["object"]).columns

numerical
categorical = df_p.select_dtypes(include = ["object"]).columns

categorical
numerical = numerical.drop("Id")

numerical = numerical.drop("LifeSquare")

numerical = numerical.drop("Healthcare_1")

numerical
train_numerical = df_p[numerical]

train_numerical.head(3)
train_numerical.isnull().values.sum()
#если есть пропуски

train_numerical = train_numerical.fillna(train_numerical.median())
# Среднее значение

mean = train_numerical.mean(axis=0)

# Стандартное отклонение

std = train_numerical.std(axis=0)

train_numerical -= mean

train_numerical /= std



train_numerical.head()
train_categorical = df_p[categorical]

train_categorical.describe()

train_categorical.Shops_2, Shop_2Ref = SeriesFactorizer(df.Shops_2)

train_categorical.Ecology_2, Ecology_2Ref = SeriesFactorizer(df.Ecology_2)

train_categorical.Ecology_3, Ecology_3Ref = SeriesFactorizer(df.Ecology_3)
x_train = pd.concat([train_numerical, train_categorical], axis = 1)

x_train.info()
y_train = df['Price']

y_train[:5]
X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, shuffle=True, random_state=42)
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error
#создаем список для складывания результатов моделей

models_dict = {}
random_forest_regressor_model = RandomForestRegressor(n_estimators=2000, max_depth=17, random_state=42)

random_forest_regressor_model.fit(X_train, y_train)
models_dict['Random Forest Regressor'] = random_forest_regressor_model
model_test(random_forest_regressor_model,

           'Random Forest Regressor', X_valid, y_valid)
y_train_preds = random_forest_regressor_model.predict(X_train)

y_test_preds = random_forest_regressor_model.predict(X_valid)
plt.figure(figsize=(20,10))



plt.subplot(121)

sns.scatterplot(x = y_train_preds, y = y_train)

plt.title('Train')



plt.subplot(122)

sns.scatterplot(x = y_test_preds, y = y_valid)

plt.title('Test')



plt.show()
print(f'R2 train : {r2_score(y_train, y_train_preds)}')

print(f'R2 test : {r2_score(y_valid, y_test_preds)}')
gradient_boosting_regressor_model = GradientBoostingRegressor(random_state=42)

gradient_boosting_regressor_model.fit(X_train, y_train)
models_dict['Gradient Boosting Regressor'] = gradient_boosting_regressor_model
model_test(gradient_boosting_regressor_model,

           'Gradient Boosting Regressor', X_valid, y_valid)
y_train_preds_grad = gradient_boosting_regressor_model.predict(X_train)

y_test_preds_grad = gradient_boosting_regressor_model.predict(X_valid)
plt.figure(figsize=(20,10))



plt.subplot(121)

sns.scatterplot(x = y_train_preds_grad, y = y_train)

plt.title('Train')



plt.subplot(122)

sns.scatterplot(x = y_test_preds_grad, y = y_valid)

plt.title('Test')



plt.show()
print(f'R2 train : {r2_score(y_train, y_train_preds_grad)}')

print(f'R2 test : {r2_score(y_valid, y_test_preds_grad)}')
lgbm_regressor_model_2 = LGBMRegressor(

                                        max_bin=800,

                                        n_estimators=220,

                                        max_depth=17,

                                        random_state=42

                                        )



lgbm_regressor_model_2.fit(X_train, y_train)
models_dict['LGBMRegressore2'] = lgbm_regressor_model_2
model_test(lgbm_regressor_model_2,

           'LGBMRegressore2', X_valid, y_valid)
y_train_preds_lgm = lgbm_regressor_model_2.predict(X_train)

y_test_preds_lgm = lgbm_regressor_model_2.predict(X_valid)
plt.figure(figsize=(20,10))



plt.subplot(121)

sns.scatterplot(x = y_train_preds_lgm, y = y_train)

plt.title('Train')



plt.subplot(122)

sns.scatterplot(x = y_test_preds_lgm, y = y_valid)

plt.title('Test')



plt.show()
print(f'R2 train : {r2_score(y_train, y_train_preds_lgm)}')

print(f'R2 test : {r2_score(y_valid, y_test_preds_lgm)}')
xgb_model = XGBRegressor(max_depth=8,

                         n_estimators=160,

                         colsample_bytree=0.2,

                         n_jobs=-1,

                         random_state=42

                         )

                     

xgb_model.fit(X_train, y_train)
models_dict['XGBMRegressor'] = xgb_model
model_test(xgb_model, 'XGBMRegressor', X_valid, y_valid)
y_train_preds_xgm = xgb_model.predict(X_train)

y_test_preds_xgm = xgb_model.predict(X_valid)
plt.figure(figsize=(20,10))



plt.subplot(121)

sns.scatterplot(x = y_train_preds_xgm, y = y_train)

plt.title('Train')



plt.subplot(122)

sns.scatterplot(x = y_test_preds_xgm, y = y_valid)

plt.title('Test')



plt.show()
print(f'R2 train : {r2_score(y_train, y_train_preds_xgm)}')

print(f'R2 test : {r2_score(y_valid, y_test_preds_xgm)}')
models_score_test = models_r2(models_dict, X_valid, y_valid)

models_score_train = models_r2(models_dict, X_train, y_train)
models_score_test[['name', 'r2']]
r2_max_test = models_score_test['r2'].max()

r2_max_train = models_score_train['r2'].max()

plt.barh(models_score_test['name'], models_score_test['r2'],

         alpha=0.5, color='red', label=f'Test  Data: R2 max: {r2_max_test:.4f}')

plt.barh(models_score_train['name'], models_score_train['r2'],

         alpha=0.5, color='grey', label=f'Train Data: R2 max: {r2_max_train:.4f}')

plt.title('R2')

plt.legend()

plt.axvline(0.65, color='red')

plt.axvline(r2_max_test, color='yellow')

plt.show()
mse_min_test = models_score_test['mse'].min()

mse_min_train = models_score_train['mse'].min()

plt.barh(models_score_test['name'], models_score_test['mse'],

         alpha=0.5, color='red', label=f'Test  Data MSE min: {mse_min_test:.0e}')

plt.barh(models_score_train['name'], models_score_train['mse'],

         alpha=0.5, color='grey', label=f'Train Data MSE min: {mse_min_train:.0e}')

plt.title('Mean squared error')

plt.legend(loc=2)

plt.axvline(mse_min_test, color='yellow')

plt.show()