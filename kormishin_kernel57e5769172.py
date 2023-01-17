import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import os

import shutil



sns.set()



%matplotlib inline



warnings.filterwarnings("ignore")



# не обрезать колонки (видимость до 100 столбцов)

pd.set_option('max_columns',100)



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Удаляем папку logs (используется tensorboard и catboost) и её содержимое после последних запусков

# if os.path.isdir('/kaggle/working/logs'):

#    shutil.rmtree('/kaggle/working/logs')
df = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/vehicles.csv', 

                 usecols=[4,5,6,7,8,9,10,11,12,13,15,16,17,18,22,23,24],

                 encoding='latin',

                 dtype={4:int, 

                        5:float, 

                        6:object, 

                        7:object, 

                        8:object, 

                        9:object, 

                        10:object, 

                        11:float, 

                        12:object, 

                        13:object,

                        15:object, 

                        16:object, 

                        17:object,

                        18:object, 

                        22:object, 

                        23:float, 

                        24:float})
df.head()
df.shape
df.info()
# проверяем пропуски в cтолбцах

df.isna().sum()
# частотность и тип данных в столбце Year

df.iloc[:,2].value_counts()
# частотность и тип данных в столбце Odometr

df.iloc[:,8].value_counts()
# замена пропущенных данных в year и odometer на округлённый Mean

df['year'].fillna(round(df['year'].mean(), 0), inplace=True)

df['year'] = df['year'].astype(int)



#df['odometer'].fillna(round(df['odometer'].mean(), 0), inplace=True)

#df['odometer'] = df['odometer'].astype(int)



df.info()
# проверяем пропуски в cтолбцах

df.isna().sum()
plt.figure(figsize=(8, 6))

spearman = df.corr(method = 'spearman')

sns.heatmap(spearman, annot = True)
# folium heatmap instructions: https://www.kaggle.com/daveianhickey/how-to-folium-for-maps-heatmaps-time-data

import folium

from folium.plugins import HeatMap



cars=df[df["type"]=="bus"]

cars.lat.fillna(0, inplace = True)

cars.long.fillna(0, inplace = True) 

cars = cars[['lat', 'long']]



CarMap=folium.Map(location=[35,-91],zoom_start=4)

HeatMap(data=cars, radius=16).add_to(CarMap)

CarMap
# статистика по количеству типов автомобилей (что бы не перегружать карту)

df['type'].value_counts()
# возьмём ваны (van) для примера 

cars=df[df["type"]=="van"]

cars.lat.fillna(0, inplace = True)

cars.long.fillna(0, inplace = True) 

cars = cars[['lat', 'long', 'price']]



CarMap=folium.Map(location=[35,-91],zoom_start=4)

HeatMap(data=cars, radius=12).add_to(CarMap)

CarMap
df.hist(figsize=(15,12), bins=20)
# пример выброса с максимальной ценой на полных данных

df['price'].max()
# численные колонки

def get_num_cols(df):

    num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    return num_cols



def clear_outliers(df):

    for col in get_num_cols(df):

        Q1 = np.percentile(df[col], 25)

        Q3 = np.percentile(df[col],75)



        # Interquartile range

        IQR = Q3 - Q1

        outlier_step = 1.5 * IQR



        # Outliers indices

        outliers_indicies = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        df.drop(outliers_indicies, axis = 0)

    

    return df

    

clear_outliers(df)
# пример выброса с максимальной ценой на полных данных

df['price'].max()
df = df.sample(frac=0.01, random_state=42)

df.shape
clear_outliers(df)
sns.pairplot(df)
# categorical columns

def get_cat_cols(df):

    cat_cols = [col for col in df.columns if pd.api.types.is_object_dtype(df[col])]

    return cat_cols



# not categorical columns

def get_not_cat_cols(df):

    not_cat_cols = [col for col in df.columns if not pd.api.types.is_object_dtype(df[col])]

    return not_cat_cols



def encode_cats(df):

    from sklearn.preprocessing import OneHotEncoder

    

    cat_cols = get_cat_cols(df)

    not_cat_cols = get_not_cat_cols(df)

    

    # Замен NaN на None для categorical столбцов

    for col in cat_cols:

        df[col].fillna('None', inplace=True)

    

    # трансформируем категориальные колонки

    ohe_df = pd.DataFrame(index=df.index)

    ohe = OneHotEncoder(handle_unknown='ignore')



    for col in cat_cols:

        ohe.fit(df[[col]])

        ohe_result = pd.DataFrame(ohe.transform(df[[col]]).toarray(),

                                  columns=ohe.get_feature_names(input_features=[col]),

                                  index=df.index)

        ohe_df = ohe_df.join(ohe_result)

    

    return ohe_df
# дата фрейм с закодированными категориальными признаками

df_cat_encoded = encode_cats(df)

df_cat_encoded.head()
def scale_nums(df):

    from sklearn.preprocessing import StandardScaler



    std_df = pd.DataFrame(index=df.index)

    scaler = StandardScaler()



    for col in get_num_cols(df):

        scaler.fit(df[[col]])

        std_result = pd.DataFrame(scaler.transform(df[[col]]),

                                  columns=[col],

                                  index=df.index)

        std_df = std_df.join(std_result)



    return std_df



df_nums_scaled = scale_nums(df)

df_nums_scaled.head()
# полная выборка со всеми необходимыми столбцами

df_transformed = df_nums_scaled.join(df_cat_encoded)

df_transformed.head()
def split_sample(df):

    from sklearn.model_selection import train_test_split

    

    y = df['price']

    X = df.drop(['price'], axis=1)

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 42)

    

    return X_train, X_test, y_train, y_test



X_train, X_test, y_train, y_test = split_sample(df_transformed)

X_train.head()
import tensorflow as tf

from catboost import CatBoostRegressor, Pool



# TENSORBOARD (не работает на Kaggle)

#if not os.path.isdir('/kaggle/working/logs'):

#    os.mkdir('/kaggle/working/logs')  

#%load_ext tensorboard

#%tensorboard --logdir '/kaggle/working/logs'
# лучший путь запуска Catboost, если использовать Tensorboard

'''for learning_rate in [0.03, 0.1]:

    for depth in [4, 10, 12]:

        for l2_leaf_reg in [3, 5, 7, 9, 15]:

            train_dir = f'/kaggle/working/logs/lr={learning_rate} depth={depth} l2_leaf_reg={l2_leaf_reg}'

            try:

                os.mkdir(train_dir)

            except:

                pass



            model = CatBoostRegressor(iterations=50,

                          loss_function = 'RMSE',

                          # включаем GPU при большом количестве итераций

                          #task_type="GPU",

                          devices='0:1',

                          learning_rate= learning_rate,

                          train_dir = train_dir,

                          depth = depth,

                          l2_leaf_reg = l2_leaf_reg)



            model.fit(X = np.array(X_train, float),

                      y = np.array(y_train, float),

                      eval_set = (np.array(X_test, float), np.array(y_test, float)),

                      silent = True,

                      early_stopping_rounds=10)

'''
# т.к. ресурсы ограничены, то ставим 50 итераций (но в идеале нужно ставить примерно несколько тысяч)

model = CatBoostRegressor(iterations=50,

                          loss_function = 'RMSE',

                          # включаем GPU при большом количестве итераций

                          # task_type="GPU",

                          # devices='0:1',

                          # директория для tensorboard

                          # train_dir = train_dir

                         )



grid = {'learning_rate': [0.03, 0.1],

        'depth': [4, 6, 10],

        'l2_leaf_reg': [1, 3, 5, 7, 9]}



grid_search_result = model.grid_search(grid,

                                        X=X_train, 

                                        y=y_train, 

                                        plot=True,

                                        search_by_train_test_split = False )
grid_search_result['params']
# Обучаем CatBoost с лучшими параметрами

model_cat = CatBoostRegressor(iterations=50,

                          loss_function = 'RMSE',

                          # включаем GPU при большом количестве итераций 

                          # task_type="GPU",

                          # devices='0:1',

                          # директория для tensorboard

                          # train_dir = train_dir,

                          learning_rate = grid_search_result['params']['learning_rate'],

                          depth = grid_search_result['params']['depth'],

                          l2_leaf_reg = grid_search_result['params']['l2_leaf_reg'],

                          verbose = True)



model_cat.fit(X = np.array(X_train, float),

                      y = np.array(y_train, float),

                      eval_set = (np.array(X_test, float), np.array(y_test, float)),

                      silent = True,

                      # больше при большом количестве итерраций (1000 для 2000, например)

                      early_stopping_rounds=20)



y_pred = model_cat.predict(X_test)
def accuracy_report(y_test, y_pred):

    print("Точность предсказания \n")

    from sklearn.metrics import r2_score, mean_squared_error

    print("R^2 score: ", r2_score(y_test, y_pred), "\nMSE: ", mean_squared_error(y_test, y_pred))



    plt.figure(figsize = (10, 8))

    plt.scatter(y_test, y_pred)

    plt.xlim(-1, 8)

    plt.ylim(-1, 8)

    plt.xlabel('y_test')

    plt.ylabel('y_pred')

    plt.title('Точность предсказания: y_predicted vs y_test')

    # линия

    x = np.linspace(-1, 8, 10)

    plt.plot(x, x, '-r')

    # Текстовые боксы

    plt.text(0, 5, "Недооценённые авто", fontsize=14, verticalalignment='top')

    plt.text(5, 0, "Переоценёные авто", fontsize=14, verticalalignment='top')



accuracy_report(y_test, y_pred)
# Для распределённых вычислений стоит работать с Dask

# from dask_ml.xgboost import XGBRegressor

# from dask.distributed import Client, progress

# client = Client(processes=False, threads_per_worker=2, n_workers=8, memory_limit='20GB')

# client



from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV



# т.к. ресурсы ограничены, то ставим n_estimators = 10 (но в идеале нужно ставить значительно больше)

xgb_model = XGBRegressor(n_estimators = 10)



# параметры, которые были оптимальны для CatBoost: learning_rate=0.1, max_depth = 10, reg_lambda = 1

params = {'learning_rate': [0.03, 0.1],

        'max_depth': [6, 10],

        'reg_lambda': [1, 3]}



# перебор параметров

xgb_grid = GridSearchCV(xgb_model,

                        params,

                        n_jobs = 4,

                        cv = 3)



# обучаем с перебором параметров

xgb_grid.fit(X_train, 

            y_train,

            # больше при большом количестве итерраций (1000 для 2000, например)

            early_stopping_rounds = 5,

            eval_set = [(X_test, y_test)], 

            verbose = False)
print("XGBoost лучшие папаметры: ", xgb_grid.best_params_, "\nЛучший score: ", xgb_grid.best_score_)
# предсказываем и строим отчёт

y_pred = xgb_grid.predict(X_test)

accuracy_report(y_test, y_pred)
from xgboost import plot_importance



# запустим модель XGBoost повторно на лучших параметрах, что бы получить важность характеристик

xgb_model_best = XGBRegressor(n_estimators = 10,

                        learning_rate = 0.1, 

                        max_depth = 10, 

                        reg_lambda = 1)



xgb_model_best.fit(X_train, 

                y_train,

                # больше при большом количестве итерраций (1000 для 2000, например)

                early_stopping_rounds = 5,

                eval_set = [(X_test, y_test)], 

                verbose = False)



plot_importance(xgb_model_best, max_num_features = 20)
# сделаем ещё раз оценку на базе CatBoost с лучшими параметрамии, т.к. эта модель отработала лучше всего

y_pred = model_cat.predict(X_test)
# степень недооценённости автомобиля

y_underestim = y_pred - y_test



X_test['y_test'] = y_test

X_test['y_pred'] = y_pred

X_test['y_underestim'] = y_underestim



# добавим данные для анализа

valuation = X_test[['y_test', 'y_pred', 'y_underestim']].join(df[['manufacturer', 'type']], on = X_test.index, how = 'left')

valuation.head()
# закодируем недооценённые авто как 'Недооценён', переоценённые как 'Переоценён', остальные как 'Норма'

valuation['estimation'] = 'Норма'



# недооценённые

valuation.loc[valuation['y_underestim'] >= 0.3, 'estimation'] = 'Недооценён'



# переоценённые

valuation.loc[valuation['y_underestim'] <= -0.3, 'estimation'] = 'Переоценён'
valuation_sorted = valuation.sort_values(by='y_underestim', ascending=False)

valuation_sorted
valuation['estimation'].value_counts()
# уберём нейтральные значения ('Норма') из данных, что бы можно было сравнить недоценённые авто и переоценённые

valuation_to_plot = valuation[(valuation['estimation'] == 'Недооценён') | (valuation['estimation'] == 'Переоценён')]



sns.catplot(x="manufacturer", hue="estimation", data=valuation_to_plot, kind="count", orient = "h", height=6, aspect=2.5)

sns.catplot(x="type", hue="estimation", data=valuation_to_plot, kind="count", orient = "h", height=6, aspect=2.5)