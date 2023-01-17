import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

%matplotlib inline





# Просмотрим наличие файлов в каталоге

import os

PATH="../input/home-credit-default-risk/"

print(os.listdir(PATH))
# Подругажаем файлы и выводим размер обучающей и тестовой выборки

app_train = pd.read_csv(PATH + 'application_train.csv')

app_test = pd.read_csv(PATH + 'application_test.csv')

print ('формат обучающей выборки:', app_train.shape)

print ('формат тестовой выборки:', app_test.shape)
pd.set_option('display.max_columns', None) # иначе pandas не покажет все столбцы

app_train.head()
app_train.info(max_cols=122)
# Распределение целевой переменной

app_train['TARGET'].value_counts()
plt.style.use('fivethirtyeight')

plt.rcParams["figure.figsize"] = [8,5]

plt.hist(app_train.TARGET)

plt.show()
# Проверим недостающие данные

# Функция для подсчета недостающих столбцов

def missing_values_table(df):

    

        # Всего недостает

        mis_val = df.isnull().sum()

        

        # Процент недостающих данных

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Таблица с результатами

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Переименование столбцов

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Сортировка про процентажу

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Инфо

        print ("В выбранном датафрейме " + str(df.shape[1]) + " столбцов.\n"      

            "Всего " + str(mis_val_table_ren_columns.shape[0]) +

              " столбцов с неполными данными.")

        

        # Возврат таблицы с данными

        return mis_val_table_ren_columns

    

missing_values = missing_values_table(app_train)

missing_values.head(10)
# В графическом формате

plt.style.use('seaborn-talk')

fig = plt.figure(figsize=(18,6))

miss_train = pd.DataFrame((app_train.isnull().sum())*100/app_train.shape[0]).reset_index()

miss_test = pd.DataFrame((app_test.isnull().sum())*100/app_test.shape[0]).reset_index()

miss_train["type"] = "тренировочная"

miss_test["type"]  =  "тестовая"

missing = pd.concat([miss_train,miss_test],axis=0)

ax = sns.pointplot("index",0,data=missing,hue="type")

plt.xticks(rotation =90,fontsize =7)

plt.title("Доля отсуствующих значений в данных")

plt.ylabel("Доля в %")

plt.xlabel("Столбцы")
app_train.dtypes.value_counts()
app_train.select_dtypes(include=[object]).apply(pd.Series.nunique, axis = 0)
# кодировать категориальнцые признаки можно с помощью one hot encoding либо label encoding

# можно совершить кодирование с помощью pandas 

app_train = pd.get_dummies(app_train)

app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)

print('Testing Features shape: ', app_test.shape)
#сохраним лейблы, их же нет в тестовой выборке и при выравнивании они потеряются. 

train_labels = app_train['TARGET']

# Выравнивание - сохранятся только столбцы. имеющиеся в обоих датафреймах

app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

print('Формат тренировочной выборки: ', app_train.shape)

print('Формат тестовой выборки: ', app_test.shape)

# Add target back in to the data

app_train['TARGET'] = train_labels
# Корреляция и сортировка

correlations = app_train.corr()['TARGET'].sort_values()

# Отображение

print('Наивысшая позитивная корреляция: \n', correlations.tail(15))

print('\nНаивысшая негативная корреляция: \n', correlations.head(15))
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])

app_train['DAYS_BIRTH'].corr(app_train['TARGET'])
# Гистограмма распределения возраста в годах, всего 25 столбцов

plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)

plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');
# KDE займов, выплаченных вовремя

sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')

# KDE проблемных займов

sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')

# Обозначения

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

ext_data_corrs = ext_data.corr()

ext_data_corrs
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

plt.title('Correlation Heatmap');
plt.figure(figsize = (10, 12))

# итерация по источникам

for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):

    

    # сабплот

    plt.subplot(3, 1, i + 1)

    # отрисовка качественных займов

    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source], label = 'target == 0')

    # отрисовка дефолтных займов

    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source], label = 'target == 1')

    

    # метки

    plt.title('Distribution of %s by Target Value' % source)

    plt.xlabel('%s' % source); plt.ylabel('Density');

    

plt.tight_layout(h_pad = 2.5)
#вынесем данные по возрасту в отдельный датафрейм

age_data = app_train[['TARGET', 'DAYS_BIRTH']]

age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

# копирование данных для графика

plot_data = ext_data.drop(labels = ['DAYS_BIRTH'], axis=1).copy()

# Добавим возраст

plot_data['YEARS_BIRTH'] = age_data['YEARS_BIRTH']

# Уберем все незаполненнные строки и ограничим таблицу в 100 тыс. строк

plot_data = plot_data.dropna().loc[:100000, :]

# Функиця для расчет корреляции

def corr_func(x, y, **kwargs):

    r = np.corrcoef(x, y)[0][1]

    ax = plt.gca()

    ax.annotate("r = {:.2f}".format(r),

                xy=(.2, .8), xycoords=ax.transAxes,

                size = 20)

# Создание объекта pairgrid object

grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False,

                    hue = 'TARGET', 

                    vars = [x for x in list(plot_data.columns) if x != 'TARGET'])

# Сверху - скаттерплот

grid.map_upper(plt.scatter, alpha = 0.2)

# Диагональ - гистограмма

grid.map_diag(sns.kdeplot)

# Внизу - распределение плотности

grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r);

plt.suptitle('Ext Source and Age Features Pairs Plot', size = 32, y = 1.05);
application_train = pd.read_csv(PATH + "application_train.csv")

application_test = pd.read_csv(PATH + "application_test.csv")
def plot_stats(feature,label_rotation=False,horizontal_layout=True):

    temp = application_train[feature].value_counts()

    df1 = pd.DataFrame({feature: temp.index,'Количество займов': temp.values})

    # Расчет доли target=1 в категории

    cat_perc = application_train[[feature, 'TARGET']].groupby([feature],as_index=False).mean()

    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

    

    if(horizontal_layout):

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

    else:

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12,14))

    sns.set_color_codes("pastel")

    s = sns.barplot(ax=ax1, x = feature, y="Количество займов",data=df1)

    if(label_rotation):

        s.set_xticklabels(s.get_xticklabels(),rotation=90)

    

    s = sns.barplot(ax=ax2, x = feature, y='TARGET', order=cat_perc[feature], data=cat_perc)

    if(label_rotation):

        s.set_xticklabels(s.get_xticklabels(),rotation=90)

    plt.ylabel('Доля проблемных', fontsize=10)

    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show();
plot_stats('NAME_CONTRACT_TYPE')
plot_stats('CODE_GENDER')
plot_stats('FLAG_OWN_CAR')

plot_stats('FLAG_OWN_REALTY')
plot_stats('NAME_FAMILY_STATUS',True, True)
plot_stats('CNT_CHILDREN')
application_train.CNT_CHILDREN.value_counts()
plot_stats('CNT_FAM_MEMBERS',True)
plot_stats('NAME_INCOME_TYPE',False,False)
plot_stats('OCCUPATION_TYPE',True, False)
application_train.OCCUPATION_TYPE.value_counts()
plot_stats('NAME_EDUCATION_TYPE',True)
plot_stats('ORGANIZATION_TYPE',True, False)
plt.figure(figsize=(12,5))

plt.title("Распределение AMT_CREDIT")

ax = sns.distplot(app_train["AMT_CREDIT"])
plt.figure(figsize=(12,5))

# KDE займов, выплаченных вовремя

sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'AMT_CREDIT'], label = 'target == 0')

# KDE проблемных займов

sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'AMT_CREDIT'], label = 'target == 1')

# Обозначения

plt.xlabel('Сумма кредитования'); plt.ylabel('Плотность'); plt.title('Суммы кредитования');
plt.figure(figsize=(12,5))

plt.title("Распределение REGION_POPULATION_RELATIVE")

ax = sns.distplot(app_train["REGION_POPULATION_RELATIVE"])
plt.figure(figsize=(12,5))

# KDE займов, выплаченных вовремя

sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'REGION_POPULATION_RELATIVE'], label = 'target == 0')

# KDE проблемных займов

sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'REGION_POPULATION_RELATIVE'], label = 'target == 1')

# Обозначения

plt.xlabel('Плотность'); plt.ylabel('Плотность населения'); plt.title('Плотность населения');
# создадим новый датафрейм для полиномиальных признаков

poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]

poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

# обработаем отуствующие данные



from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy = 'median')

poly_target = poly_features['TARGET']

poly_features = poly_features.drop('TARGET', axis=1)

poly_features = imputer.fit_transform(poly_features)

poly_features_test = imputer.transform(poly_features_test)



from sklearn.preprocessing import PolynomialFeatures

                                  

# Создадим полиномиальный объект степени 3

poly_transformer = PolynomialFeatures(degree = 3)



# Тренировка полиномиальных признаков

poly_transformer.fit(poly_features)



# Трансформация признаков

poly_features = poly_transformer.transform(poly_features)

poly_features_test = poly_transformer.transform(poly_features_test)

print('Формат полиномиальных признаков: ', poly_features.shape)
poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]
# Датафрейм для новых фич 

poly_features = pd.DataFrame(poly_features, 

                             columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 

                                                                           'EXT_SOURCE_3', 'DAYS_BIRTH']))

# Добавим таргет

poly_features['TARGET'] = poly_target

# рассчитаем корреляцию

poly_corrs = poly_features.corr()['TARGET'].sort_values()

# Отобразим признаки с наивысшей корреляцией

print(poly_corrs.head(10))

print(poly_corrs.tail(5))
# загрузим тестовые признаки в датафрейм

poly_features_test = pd.DataFrame(poly_features_test, 

                                  columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 

                                                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))

# объединим тренировочные датафреймы

poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']

app_train_poly = app_train.merge(poly_features, on = 'SK_ID_CURR', how = 'left')

# объединим тестовые датафреймы

poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']

app_test_poly = app_test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')

# Выровняем датафреймы

app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)

# Посмотрим формат

print('Тренировочная выборка с полиномиальными признаками: ', app_train_poly.shape)

print('Тестовая выборка с полиномиальными признаками: ', app_test_poly.shape)
from sklearn.preprocessing import MinMaxScaler

from sklearn.impute import SimpleImputer

# Уберем таргет из тренировочных данных

if 'TARGET' in app_train:

    train = app_train.drop(labels = ['TARGET'], axis=1)

else:

    train = app_train.copy()

features = list(train.columns)

# копируем тестовые данные

test = app_test.copy()

# заполним недостающее по медиане

imputer = SimpleImputer(strategy = 'median')

# Нормализация

scaler = MinMaxScaler(feature_range = (0, 1))

# заполнение тренировочной выборки

imputer.fit(train)

# Трансофрмация тренировочной и тестовой выборок

train = imputer.transform(train)

test = imputer.transform(app_test)

# то же самое с нормализацией

scaler.fit(train)

train = scaler.transform(train)

test = scaler.transform(test)

print('Формат тренировочной выборки: ', train.shape)

print('Формат тестовой выборки: ', test.shape)
from sklearn.linear_model import LogisticRegression

# Создаем модель

log_reg = LogisticRegression(C = 0.0001)

# Тренируем модель

log_reg.fit(train, train_labels)

LogisticRegression(C=0.0001, class_weight=None, dual=False,

          fit_intercept=True, intercept_scaling=1, max_iter=100,

          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,

          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

# Теперь модель можно использовать для предсказаний. Метод prdict_proba даст на выходе массив m x 2, где m - количество наблюдений, первый столбец - вероятность 0, второй - вероятность 1. Нам нужен второй (вероятность невозврата).





log_reg_pred = log_reg.predict_proba(test)[:, 1]
submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = log_reg_pred

submit.head()
from sklearn.ensemble import RandomForestClassifier

# Создадим классификатор

random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50)

# Тренировка на тернировочных данных

random_forest.fit(train, train_labels)

# Предсказание на тестовых данных

predictions = random_forest.predict_proba(test)[:, 1]

# Создание датафрейма для загрузки

#submit = app_test[['SK_ID_CURR']]

#submit['TARGET'] = predictions

# Сохранение

#submit.to_csv('random_forest_baseline.csv', index = False)
from lightgbm import LGBMClassifier



clf = LGBMClassifier()

clf.fit(train, train_labels)



predictions = clf.predict_proba(test)[:, 1]

print(predictions)



# Датафрейм для загрузки

#submit = app_test[['SK_ID_CURR']]

#submit['TARGET'] = predictions



# Сохранение датафрейма

#submit.to_csv('lightgbm_baseline.csv', index = False)
predictions