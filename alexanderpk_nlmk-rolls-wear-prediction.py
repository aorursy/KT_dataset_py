INPUT_DIR = r'../input'

import os

print(os.listdir(INPUT_DIR))
import os

import time

import tqdm

import datetime

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from catboost import Pool, CatBoostRegressor

import hyperopt



import warnings

warnings.filterwarnings("ignore")



%matplotlib inline

%config InlineBackend.figure_format = 'retina'

sns.set()
# Чтение данных

df_ruloni = pd.read_csv(os.path.join(INPUT_DIR, 'Ruloni.csv'))

df_ruloni_copy = df_ruloni.copy()

df_valki = pd.read_csv(os.path.join(INPUT_DIR, 'Valki.csv'))

df_zavalki = pd.read_csv(os.path.join(INPUT_DIR, 'Zavalki.csv'))



df_ruloni.rename({'Время_обработки':'дата_завалки'}, axis=1, inplace=True)

df_test = pd.read_csv(os.path.join(INPUT_DIR, 'Test.csv'))

df_sample_test = pd.read_csv(os.path.join(INPUT_DIR, 'sample_test.csv'))
# Запишем индексы партий

df_ruloni['party'] = 1e10



# По датам завалки трейна

dates = df_zavalki['дата_завалки'].unique().tolist()

# По датам завалки теста

test_dates = df_test['дата_завалки'].unique().tolist()

dates = dates + test_dates



i = 0

for f in tqdm.tqdm_notebook(dates[1:]):

    #print(f'{f} < x <{dates[i]}')  

    df_ruloni.loc[(df_ruloni['дата_завалки'] < f)&

                  (df_ruloni['party'] >= i), 'party'] = i

    i+=1



# Корректировка последней обрабатываемой партии

# <2018-12-31 21:03:39

df_ruloni.loc[(df_ruloni['party']==1e10)&

              (df_ruloni['дата_завалки']<'2018-12-31 21:03:39'), 'party'] = 2413

df_ruloni.loc[df_ruloni['party']==1e10, 'party'] = 2414
df_ruloni.tail()
# Еще добавим для понимания, что шло по порядку в партии

df_ruloni['порядок_прохода'] = None

for i in tqdm.tqdm_notebook(range(2414+1)):

    l = df_ruloni[df_ruloni['party']==i].shape[0]

    df_ruloni.loc[df_ruloni['party']==i, 'порядок_прохода'] = [x for x in range(1, l+1)]
# Для df_ruloni дату завалки добавим начальную, чтобы сопоставить с остальными данными

for i in tqdm.tqdm_notebook(range(2414+1)):

    d = df_ruloni[df_ruloni['party']==i]['дата_завалки'].iloc[0]

    df_ruloni.loc[df_ruloni['party']==i, 'дата_завалки'] = d
df_ruloni.head()
df_ruloni['порядок_прохода'].max()
# Тоннаж 

tonn = pd.DataFrame(df_ruloni.groupby(['party'])['Масса'].sum())

tonn.reset_index(inplace=True, drop=False)

mass_std = tonn['Масса'].std()

mass_mean = tonn['Масса'].mean()

plt.hist(tonn['Масса'], bins = 20)

plt.plot([mass_mean+ 3*mass_std for x in range(100)], [x*2 for x in range(100)])

plt.plot([mass_mean- 3*mass_std for x in range(100)], [x*2 for x in range(100)])

plt.show()
df_ruloni['Ширина'].hist(bins=20)

plt.show()
df_ruloni['Толщина'].hist(bins=20)

plt.show()
# Единственный категориальный признак, который необходимо обработать явным образом - Марка стали

marki = pd.get_dummies(df_ruloni['Марка'])

df_ruloni = pd.concat([df_ruloni, marki], axis=1)

del df_ruloni['Марка']
df_ruloni.head()
def to_dtime(s):

    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

def to_seconds(end, start):

    return (to_dtime(end) - to_dtime(start)).seconds
# Имена прокатываемых марок стали

marki_names = df_ruloni.columns.tolist()[6:]



df_ruloni['Толщина_Ширина'] = df_ruloni['Толщина']*df_ruloni['Ширина']
# Данные по прокатываемым маркам стали

v1 = pd.DataFrame(df_ruloni.groupby(['дата_завалки'])[marki_names].sum())

v1.reset_index(inplace=True)



# Сколько материала прошло

v2 = pd.DataFrame(df_ruloni.groupby(['дата_завалки'])['порядок_прохода'].max())

v2.reset_index(inplace=True)

v2.rename({'порядок_прохода':'прошло_рулонов'}, axis=1)



# Суммарная пройденная масса

v3 = pd.DataFrame(df_ruloni.groupby(['дата_завалки'])['Масса'].sum())

v3.reset_index(inplace=True)

v3.rename({'Масса':'Масса_сумма'}, axis=1, inplace=True)



# Минимум массы

v4 = pd.DataFrame(df_ruloni.groupby(['дата_завалки'])['Масса'].min())

v4.reset_index(inplace=True)

v4.rename({'Масса':'Масса_мин'}, axis=1, inplace=True)



# Максимум массы

v5 = pd.DataFrame(df_ruloni.groupby(['дата_завалки'])['Масса'].max())

v5.reset_index(inplace=True)

v5.rename({'Масса':'Масса_макс'}, axis=1, inplace=True)



# Сигма масса

v6 = pd.DataFrame(df_ruloni.groupby(['дата_завалки'])['Масса'].std())

v6.reset_index(inplace=True)

v6.rename({'Масса':'Масса_сигма'}, axis=1, inplace=True)



# Средняя масса

v7 = pd.DataFrame(df_ruloni.groupby(['дата_завалки'])['Масса'].mean())

v7.reset_index(inplace=True)

v7.rename({'Масса':'Масса_ср'}, axis=1, inplace=True)
# Средняя толщина

v8 = pd.DataFrame(df_ruloni.groupby(['дата_завалки'])['Толщина'].mean())

v8.reset_index(inplace=True)

v8.rename({'Толщина':'Толщина_ср'}, axis=1, inplace=True)



# Сигма толщины

v9 = pd.DataFrame(df_ruloni.groupby(['дата_завалки'])['Толщина'].std())

v9.reset_index(inplace=True)

v9.rename({'Толщина':'Толщина_сигма'}, axis=1, inplace=True)



# Средняя ширина

v10 = pd.DataFrame(df_ruloni.groupby(['дата_завалки'])['Ширина'].mean())

v10.reset_index(inplace=True)

v10.rename({'Ширина':'Ширина_ср'}, axis=1, inplace=True)



# Сигма ширины

v11 = pd.DataFrame(df_ruloni.groupby(['дата_завалки'])['Ширина'].std())

v11.reset_index(inplace=True)

v11.rename({'Ширина':'Ширина_сигма'}, axis=1, inplace=True)



# Средняя Толщина*Ширина

v12 = pd.DataFrame(df_ruloni.groupby(['дата_завалки'])['Толщина_Ширина'].mean())

v12.reset_index(inplace=True)

v12.rename({'Толщина_Ширина':'Толщина_Ширина_ср'}, axis=1, inplace=True)



# Сигма Толщина*Ширина

v13 = pd.DataFrame(df_ruloni.groupby(['дата_завалки'])['Толщина_Ширина'].std())

v13.reset_index(inplace=True)

v13.rename({'Толщина_Ширина':'Толщина_Ширина_сигма'}, axis=1, inplace=True)
# Мин толщины

v14 = pd.DataFrame(df_ruloni.groupby(['дата_завалки'])['Толщина'].min())

v14.reset_index(inplace=True)

v14.rename({'Толщина':'Толщина_мин'}, axis=1, inplace=True)



# Макс толщины

v15 = pd.DataFrame(df_ruloni.groupby(['дата_завалки'])['Толщина'].max())

v15.reset_index(inplace=True)

v15.rename({'Толщина':'Толщина_макс'}, axis=1, inplace=True)



# Мин ширины

v16 = pd.DataFrame(df_ruloni.groupby(['дата_завалки'])['Ширина'].min())

v16.reset_index(inplace=True)

v16.rename({'Ширина':'Ширина_мин'}, axis=1, inplace=True)



# Макс ширины

v17 = pd.DataFrame(df_ruloni.groupby(['дата_завалки'])['Ширина'].max())

v17.reset_index(inplace=True)

v17.rename({'Ширина':'Ширина_макс'}, axis=1, inplace=True)
# Добавим время обработки для пооперационного подсчета

df_ruloni['Время_обработки'] = df_ruloni_copy['Время_обработки']

df_ruloni['Время_обработки_lag'] = df_ruloni['Время_обработки'].shift(1)

df_ruloni['Время_обработки_lag'].fillna(value=0, inplace=True)
t = [116.81109473044754] # Среднее по всем остальным для заполнения пропусков

for i in tqdm.tqdm_notebook(range(1,df_ruloni.shape[0])):

    s = to_seconds(df_ruloni['Время_обработки'].iloc[i],

                   df_ruloni['Время_обработки_lag'].iloc[i])

    t.append(s)

df_ruloni['Операция'] = t
# Средняя операция

v18 = pd.DataFrame(df_ruloni.groupby(['дата_завалки'])['Операция'].mean())

v18.reset_index(inplace=True)

v18.rename({'Операция':'Операция_ср'}, axis=1, inplace=True)



# Сигма операция

v19 = pd.DataFrame(df_ruloni.groupby(['дата_завалки'])['Операция'].std())

v19.reset_index(inplace=True)

v19.rename({'Операция':'Операция_сигма'}, axis=1, inplace=True)
# Добавим значения долей времени, затраченных на каждую марку стали

df_ruloni_newf = df_ruloni.copy()



cls = df_ruloni_newf.columns.tolist()

for i in ['Масса', 'Толщина', 'Ширина', 'дата_завалки', 'party',

          'порядок_прохода', 'Толщина_Ширина','Время_обработки',

          'Время_обработки_lag']:

    cls.remove(i)

df_ruloni_newf = df_ruloni_newf[cls].copy()
cls.remove('Операция')

for i in cls:

    df_ruloni_newf[f'{i}_time'] = df_ruloni_newf[i]*df_ruloni_newf['Операция']

    

marki_times = df_ruloni_newf.columns.tolist()[100:]

df_ruloni_newf['marki_time'] = 0

for m in marki_times:

    df_ruloni_newf['marki_time'] = df_ruloni_newf[m]+df_ruloni_newf['marki_time']
df_ruloni['marki_time'] = df_ruloni_newf['marki_time']

df_ruloni_newf = df_ruloni_newf[marki_times].copy()

df_ruloni_newf['party'] = df_ruloni['party']
tmp = pd.DataFrame(df_ruloni.groupby(['party'])['marki_time'].sum())

tmp.reset_index(inplace=True, drop=False)

tmp.rename({'marki_time':'marki_time_sum'}, axis=1, inplace=True)



df_ruloni_newf = df_ruloni_newf.merge(tmp, how='left', on = ['party'])



for m in marki_times:

    df_ruloni_newf[m] = df_ruloni_newf[m]/df_ruloni_newf['marki_time_sum']

    

time_fractions = df_ruloni_newf[marki_times].copy()
time_fractions['party'] = df_ruloni['party']

time_fractions['дата_завалки'] = df_ruloni['дата_завалки']



time_fr = time_fractions.groupby(['дата_завалки'])[marki_times].sum()

time_fr.reset_index(inplace=True, drop=False)



# Проверка суммы долей по одному прогону партии. 

# 1+- machine eps

time_fr.iloc[0][marki_times].sum()
time_fr.head(1)
df_zavalki.head()
# Расширение df_ruloni

df_ruloni_exp = pd.merge(v1,v2, how='left', on = ['дата_завалки'])

for v in [v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,

          v14,v15,v16,v17,v18,v19,time_fr]:

    df_ruloni_exp = pd.merge(df_ruloni_exp, v, how='left', on = ['дата_завалки'])
df_ruloni_exp.head(1)
df_ruloni_exp.shape
def process_df_tocatb(df_test, df_valki, df_ruloni):



    df_test['номер_клетки'] = df_test['номер_клетки'].astype(str)

    

    time_sec = []

    for i in tqdm.tqdm(range(df_test.shape[0])):

        s = to_seconds(df_test.iloc[i]['дата_вывалки'], df_test.iloc[i]['дата_завалки'])

        time_sec.append(s)

        

    df_test['time_sec'] = time_sec

    df_test = df_test.merge(df_valki, how='left', on = ['номер_валка'])

    

    df_test = df_test.merge(df_ruloni, how='left', on = ['дата_завалки'])



    df_test.rename({'износ':'y'}, axis=1, inplace=True)

    

    return df_test
# Итоговый датасет

zavalki_to_catboost = process_df_tocatb(df_zavalki, df_valki, df_ruloni_exp)
# Тестовый датасет

test_to_catboost = process_df_tocatb(df_test, df_valki, df_ruloni_exp)
zavalki_to_catboost.head()
def y_hist(zavalki_to_catboost):

    # Распределение ответа

    plt.hist(zavalki_to_catboost['y'].values, bins=20)

    y_mean = zavalki_to_catboost['y'].mean()

    y_std = zavalki_to_catboost['y'].std()

    plt.plot([y_mean+ 3*y_std for x in range(1000)], [x*2 for x in range(1000)], color='r')

    plt.plot([y_mean- 3*y_std for x in range(1000)], [x*2 for x in range(1000)], color='r')

    plt.plot()

    plt.show()
y_hist(zavalki_to_catboost)
# Усечение выбросов

zavalki_to_catboost = zavalki_to_catboost[zavalki_to_catboost['y'] <=2]

zavalki_to_catboost_ts= zavalki_to_catboost.copy()

zavalki_to_catboost_ts.reset_index(inplace=True, drop=True)
plt.hist(zavalki_to_catboost['y'].values, bins=20)

plt.show()
# Есть немного данных по 4 кварталу

# Их возьмем для детекции переобучения итоговой модели

print(zavalki_to_catboost.iloc[17825:].shape)

zavalki_to_catboost.iloc[17825:]
# Удаление меток времени

del zavalki_to_catboost['дата_завалки']

del zavalki_to_catboost['дата_вывалки']
# Разбитие данных по кварталом.

# Тестрование результатов с поквартальным окном:

# Train 1q, Test 2q; Train 2q, Test 3q

# Небольшая часть даных по 4 кварталу - для выбора best_model для сабмита.



# Поквартальные адреса ячеек с усеченными по y данными 

# 1 reduced: 0:5869

# 2 reduced: 5869:11806

# 3 reduced: 11806:17825

# 4 small:   17825:end

X_q1nohe, y_q1 = zavalki_to_catboost.iloc[0:5869], zavalki_to_catboost['y'].iloc[0:5869]

X_q2nohe, y_q2 = zavalki_to_catboost.iloc[5870:11806], zavalki_to_catboost['y'].iloc[5870:11806]

X_q3nohe, y_q3 = zavalki_to_catboost.iloc[11807:17825], zavalki_to_catboost['y'].iloc[11807:17825]

X_q4_small_val, y_q4_small_val = zavalki_to_catboost.iloc[17825:], zavalki_to_catboost['y'].iloc[17825:]
zavalki_to_catboost.head(1)
zavalki_to_catboost.shape
# Признаки и их индексы

for i, f in enumerate(X_q1nohe.columns):

    print(i, f)
#------------------------------

#  Сводки из части идей по снижению RMSE для Catboost

#  Квартальная разбивка использована с целью приблизить результат к максимально правдоподобному по сабмиту

#------------------------------

# Добавление всех данных 

#(0.2678169292)/(0.2695692791) - на 2999 итерации



# мин/макс по Толщине и Ширине

# (0.2592309416)/(0.2623674841)- на 2999 итерации



# игнорирование переменных с нулевым весом по итогам

# (0.2592309416)/(0.2623674841)- на 2999 итерации



# мин/макс по Толщина*Ширина

# (0.2592564188)/(0.2630818086)- на 2999 итерации. Убрать



# Убрано мин/макс по Толщина*Ширина

# (0.2592791627)/(0.2626429178)- на 2999 итерации.



# Среднее и сигма по Операции

# (0.2581583367)/(0.2625029269)- на 2999 итерации.



# Масса_сумма/time_sec:

# (0.2603577499)/(0.2630249549)



# Доли времени марок ст

# (0.2567618605)/(0.259612986)



# Доли масс марок ст

# (0.2567835873)/(0.260658636) Убрать



# iterations 3000>3500 

# (0.255353185)/(0.2585162396)  - на 3499 итерации.



# iterations 3500>4000

# (0.2545113556)/(0.2576650744) - на 3999 итерации.



#  Применение hyperopt

# 500 итераций, 10 eval-ов

# {'l2_leaf_reg': 2.0, 'learning_rate': 0.03375590702146598}

# [13:01<00:00, 77.64s/it, best loss: 0.00924935909274656]

# (0.2511933302)/(0.2562225371)



# Отработка hyperopt на colaboratory:



"""

def hyperopt_objective(params):

    model = CatBoostRegressor(

        l2_leaf_reg=int(params['l2_leaf_reg']),

        learning_rate=params['learning_rate'],

        iterations=3000,

        eval_metric='RMSE',

        random_seed=42,

        task_type="GPU",

        logging_level='Silent'

    )

    

    cv_data = cv(

        Pool(zavalki_to_catboost, zavalki_to_catboost['y'],

             cat_features=categorical_features_indices),

        model.get_params()

    )

    best_rmse = np.min(cv_data['test-RMSE-mean'])

    

    return best_rmse # as hyperopt minimises

    

from numpy.random import RandomState

params_space = {

    'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 0, 2, 1),

    'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 1e-1),

}



trials = hyperopt.Trials()



best = hyperopt.fmin(

    hyperopt_objective,

    space=params_space,

    algo=hyperopt.tpe.suggest,

    max_evals=10,

    trials=trials,

    rstate=RandomState(123)

)



print(best)

100%|██████████| 10/10 [51:31<00:00, 307.29s/it, best loss: 0.009103421834351715]

{'l2_leaf_reg': 2.0, 'learning_rate': 0.03375590702146598}

"""



# 3000 итераций, 10 eval-ов

# {'l2_leaf_reg': 2.0, 'learning_rate': 0.03375590702146598}

# [51:31<00:00, 307.29s/it, best loss: 0.009103421834351715]

# (0.2511933302)/(0.2562225371)



# y_lag

# (0.2511000059)/(0.2530664897)  

# Для теста - обучение без y_lag,

# Подтягивание предсказаний и подтягивание их в качестве лагов

# Незначительный вклад, можно исключить.



# iterations 4000>6000

# (0.2511883717)/(0.2562225371) -на 5916/3277 итерации.



# Указание та то, какие еще материалы валков использовались при этом валке

# (0.2524229315)/(0.2532975342)



# Прунинг факторов

"""

Марка стали 10_time

Марка стали 104_time

Марка стали 106_time

Марка стали 107_time

Марка стали 15_time

Марка стали 28_time

Марка стали 29_time

Марка стали 31_time

Марка стали 37_time

Марка стали 40_time

"""

# (0.2508956729)/(0.2545026402)
categorical_features_indices = [0, 1, 2, 5]
# Параметры были отобраны с помощью hyperopt

params = {

    'iterations': 5000,

    'learning_rate': 0.03375590702146598,

    'l2_leaf_reg': 2.0,

    'loss_function': 'RMSE',

    'eval_metric': 'RMSE',

    'ignored_features':['y',

                        'Марка стали 10_time',

                        'Марка стали 104_time',

                        'Марка стали 106_time',

                        'Марка стали 107_time',

                        'Марка стали 15_time',

                        'Марка стали 28_time',

                        'Марка стали 29_time',

                        'Марка стали 31_time',

                        'Марка стали 37_time',

                        'Марка стали 40_time',]}



catb = CatBoostRegressor(**params)
train_pool = Pool(X_q1nohe, y_q1, 

                  cat_features=categorical_features_indices)

validate_pool = Pool(X_q2nohe, y_q2, 

                     cat_features=categorical_features_indices)

catb.fit(train_pool, eval_set=validate_pool)
d1 = dict()

for n, imp in zip(catb.feature_names_, catb.feature_importances_):

    d1.update({n:imp})

    print(n, imp)
df_d1 = pd.DataFrame([d1]).T

df_d1 = df_d1.sort_values(ascending=False, by=0)
# Top-30 признаков

fig = plt.figure(figsize = (10, 8))

xs = [i for i in range(30)]

plt.bar(xs, df_d1[0].values.tolist()[:30])

plt.xticks(xs, df_d1.index.tolist()[:30], rotation=90)

plt.show()
train_pool = Pool(X_q2nohe, y_q2, 

                  cat_features=categorical_features_indices)

validate_pool = Pool(X_q3nohe, y_q3, 

                     cat_features=categorical_features_indices)

catb.fit(train_pool, eval_set=validate_pool)
d2 = dict()

for n, imp in zip(catb.feature_names_, catb.feature_importances_):

    d2.update({n:imp})

    print(n, imp)
df_d2 = pd.DataFrame([d2]).T

df_d2 = df_d2.sort_values(ascending=False, by=0)
# Top-30 признаков

fig = plt.figure(figsize = (10, 8))

xs = [i for i in range(30)]

plt.bar(xs, df_d2[0].values.tolist()[:30])

plt.xticks(xs, df_d2.index.tolist()[:30], rotation=90)

plt.show()
# Для сабмита ищем best_model с использованием небольшого отрезка по 4 кварталу

best_model_params = params.copy()

best_model_params.update({'use_best_model': True})



best_model = CatBoostRegressor(**best_model_params)

train_to_kaggle_pool = Pool(zavalki_to_catboost, zavalki_to_catboost['y'],

                            cat_features=categorical_features_indices)



validate_pool_to_kaggle = Pool(X_q4_small_val, y_q4_small_val, 

                               cat_features=categorical_features_indices)



best_model.fit(train_to_kaggle_pool, eval_set=validate_pool_to_kaggle)
# Добавляем не учитываемую переменную y в трейне

test_to_catboost['y'] = 0

test_to_catboost_for_pred = test_to_catboost[zavalki_to_catboost.columns.tolist()]
preds_catb = best_model.predict(test_to_catboost_for_pred)

output = pd.DataFrame({'id':test_to_catboost['id'].values,

                       'iznos':preds_catb})

output['iznos'].hist()

plt.show()
#output.to_csv('submit_output.csv', index=False)

output.head(10)