# Подгружаем необходимые библиотеки для анализа данных и визуализации

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Вам нужно форкнуть скрипт fetch, чтобы можно было его импортировать. Затем нужно File>Add Utility Script>fetch

import fetch
# Используем скрипт от grwlf чтобы скачать последние данные с яндекса. Будем это делать в 2 ячейки:

data = fetch.fetch_yandex(dump_folder='')
# Вторая ячейка, данные сохранятся в файл с именем времени скачивания. Запомним имя в переменную.

data, filepath = fetch.format_csse2(data, dump_folder='')
# Считываем исторические данные

russia = pd.read_csv('https://raw.githubusercontent.com/grwlf/COVID-19_plus_Russia/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_RU.csv')
# Сфетченные данные считываем в нужном формате

russia_latest = pd.read_csv(filepath)
# Данные на сейчас

russia_latest.head()
# Исторические

russia.head()
# Собираем все данные

rus = russia.set_index('Province_State').join(russia_latest.set_index('Province_State')['Confirmed'])
today = filepath[:10]
today2 = today[3:5]+'/'+today[:2]+'/'+today[-2:]
# Наводим марафет

rus.drop(['UID','iso2','iso3','FIPS','Admin2','Country_Region','Lat','Long_','Combined_Key','code3'], axis=1, inplace=True)
rus[today2] = rus['Confirmed']
del rus['Confirmed']
# Все данные есть:

rus.head()
# Посмотрим на последние 30 дней

plt.figure(figsize=(15,10))

plt.plot(rus.T.iloc[-30:,:])

plt.show()
# Лучше взять логарифт, чтобы не смотреть только на Москву

plt.figure(figsize=(15,10))

plt.plot(np.log(rus + 0.5).T[-30:])

plt.show()
# А что если бы всех поставить с 8+ случаями в начало графика?

y = np.log(rus + 0.5).T
y_m = y[y>2]
print (np.exp(2))
y_gt_7 = pd.DataFrame(data = [[0 for i in range(85)] for j in range(len(y_m.columns))], index = range(len(y_m.columns)), columns = y_m.columns)
for i in range(len(y_m.columns)):
    temp = y_m.iloc[:,i].dropna().reset_index(drop=True)
    y_gt_7.iloc[:temp.shape[0],i] = temp
y_gt_7 = y_gt_7.replace(0,np.nan)
y_gt_7.iloc[0,:].max()
delta = y_gt_7.iloc[0,:].max() - y_gt_7.iloc[0,:]
plt.figure(figsize=(15,10))

plt.plot(y_gt_7 + delta)

plt.show()
# Пусть будет истинное количество случаев, но все вначале.

plt.figure(figsize=(15,10))

plt.plot(y_gt_7)

plt.show()
# Считаем список регионов для которых нужно предсказывать

russia_regions = pd.read_csv('/kaggle/input/russia-regions-in-sber-covid-competition/russia_regions.csv')
russia_regions.head()
# Автономные области не названы, поможем назвать

russia_regions.loc[russia_regions['iso_code'] == 'RU-NEN', 'csse_province_state'] = 'Nenetskiy autonomous oblast'
russia_regions.loc[russia_regions['iso_code'] == 'RU-CHU', 'csse_province_state'] = 'Chukotskiy autonomous oblast'
# Привести в соответствие с конкурсным перечнем регионов

rus['ind'] = rus.index
# Altay republic > Republic of Altay
rus.loc[rus.index == 'Altay republic', 'ind'] = 'Republic of Altay'
rus.set_index('ind', inplace=True)
# Данные могут быть не обновленные

# rus['04/25/20'] = rus['04/26/20']
# del rus['04/26/20']
# i это индекс, который будет меняться по регионам: берем первую область (край и т.п.)

scores = []

for i in range(len(rus.T.columns)):

    obl = pd.DataFrame(rus.T.iloc[:,i].copy(), columns=[rus.T.iloc[:,i].name]) 

    # Исправим данные, где кол-во растет, потом падает: пусть только растет
    prev = 0
    obl_corrected = []

    for one in obl.iloc[:,0]:
        if one < prev:
            obl_corrected[-1] = one

        obl_corrected.append(one)  
        prev = one

    obl.iloc[:,0] = obl_corrected

    # Индекс приводим к формату даты
    obl.index = pd.to_datetime(obl.index) 

    ## Делаем признаки

    # Порядковый нмер дня в году - "дата" для всех
    obl['day_of_year'] = obl.index.dayofyear 

    # Создаем day7 - на 1 возрастают дни (от начала года) со времени, когда кейсов стали больше 7 (после какого-то значения рост более стабильный)
    obl['day7'] = 0 
    idx_day_7 = obl.iloc[:,0]>7
    obl.loc[idx_day_7,'day7'] = obl.loc[idx_day_7,'day7'].index.dayofyear
    day_day7 = obl.loc[obl['day7'] > 0, 'day7'].min() - 1
    obl.loc[obl['day7'] > 0, 'day7'] -= day_day7 

    # Создаем apr6 - на 1 возрастают дни с 6 апреля - когда должны заметить карантин (30.03+7 дней) - траектория изменилась
    obl['apr6'] = 0 
    idx_apr_6 = obl.index >= '2020-04-06'
    obl.loc[idx_apr_6,'apr6'] = obl.loc[idx_apr_6,'apr6'].index.dayofyear
    day_apr6 = obl.loc[obl['apr6'] > 0, 'apr6'].min() - 1
    obl.loc[obl['apr6'] > 0, 'apr6'] -= day_apr6 
    
    # Создаем last14 - на 1 возрастают дни для последних 7 дней - больше учитываем последние данные
    obl['last14'] = 0 
    idx_last_14 = obl.index >= obl.index[-14]
    obl.loc[idx_last_14,'last14'] = obl.loc[idx_last_14,'last14'].index.dayofyear
    day_last14 = obl.loc[obl['last14'] > 0, 'last14'].min() - 1
    obl.loc[obl['last14'] > 0, 'last14'] -= day_last14
    
    # И сделаем квадрат от last14
    obl['last14_2'] = obl['last14'] ** 2

    # Давайте посмотрим на 2 режима - day7 (сверху) и apr6. Что видно?
    plt.figure(figsize=(15,10))
    pos_cases = obl.iloc[:,0]>0
    plt.plot(np.log(obl.loc[pos_cases].iloc[:,0]))
    plt.scatter(obl.loc[pos_cases,'day7'].index,(obl.loc[pos_cases,'day7']>0)+0.1)
    plt.scatter(obl.loc[pos_cases,'apr6'].index,(obl.loc[pos_cases,'apr6']>0))
    plt.title(obl.columns[0])
    plt.show()
    
    # Собираем Х
    X_cols = ['day_of_year','day7','apr6','last14', 'last14_2']
    
    # до 19 обучаем, c 20 будем тестироваться
    train = obl.loc[(pos_cases) & (obl.index<'2020-04-20')]
    test  = obl.loc[obl.index>'2020-04-19']

    # Х в лог
    X_train = np.log(train[X_cols] + 0.5)
    y_train = np.log(train.iloc[:,0] + 0.5)
    X_test = np.log(test[X_cols] + 0.5)
    y_test = np.log(test.iloc[:,0] + 0.5)

    # Скалируем признаки, для регрессии с регуляризацией хорошо
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    # print(scaler.mean_, scaler.var_)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Строим регрессию
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(X_train_scaled, y_train.values)

    # Функции для 2 метрик: первая - конкурсная, вторая - процент отклонения от истины
    def MALE(pred, true):
    #    print(np.log10((pred + 1) / (true + 1)))
        return np.mean(np.abs(np.log10((pred + 1) / (true + 1))))

    def AvgProc(pred, true):
    #    print((pred-true)/true)
        return np.mean(np.abs((pred-true)/true))

    # Приводим у к кол-ву случаев
    y_pred_test_exp = np.round(np.exp(reg.predict(X_test_scaled))-0.5,0)
    y_pred_train_exp = np.round(np.exp(reg.predict(X_train_scaled))-0.5,0)
    y_test_exp = np.round(np.exp(y_test)-0.5,0)
    y_train_exp = np.round(np.exp(y_train)-0.5,0)

    # Обрабатываем "Архангельский прыжок" (резкое увеличение случаев - увеличиваем предсказ на 2/3 от скачка)
    if i==2: 
        y_pred_test_exp = y_pred_test_exp + (y_train_exp[-1]-y_train_exp[-2])*(2/3)

    # Выводим результаты
    print(obl.columns[0])
    print('coefs=', reg.coef_, 'const=', reg.intercept_)
    print('MALE test = ', MALE(y_pred_test_exp, y_test_exp), 'MALE train =', MALE(y_pred_train_exp, y_train_exp))
    print('AvgProc test = ', AvgProc(y_pred_test_exp, y_test_exp), 'AvgProc train =', AvgProc(y_pred_train_exp, y_train_exp))

    # Отрисуем предсказ
    plt.figure(figsize=(15,10))
    for x,y in zip(y_test.index.dayofyear, y_test_exp):
        label = "{:.0f}".format(y)
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,-15), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center

    for x,y in zip(y_test.index.dayofyear, y_pred_test_exp):
        label = "{:.0f}".format(y)
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,-15), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center

    plt.title(obl.columns[0]+' predict')

    plt.plot(y_test.index.dayofyear, y_pred_test_exp, 'ro-')
    plt.plot(y_test.index.dayofyear, y_test_exp, 'bo-')

    plt.grid()
    plt.show()

    # Отрисуем модель
    plt.figure(figsize=(15,10))

    for x,y in zip(y_train.index.dayofyear, y_train_exp):
        label = "{:.0f}".format(y)
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,-15), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center

    for x,y in zip(y_train.index.dayofyear, y_pred_train_exp):
        label = "{:.0f}".format(y)
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,-15), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center

    plt.title(obl.columns[0]+' fit')

    plt.plot(y_train.index.dayofyear, y_pred_train_exp, 'ro-')
    plt.plot(y_train.index.dayofyear, y_train_exp, 'bo-')

    plt.grid()
    plt.show()
    
    scores.append((obl.columns[0], MALE(y_pred_test_exp, y_test_exp)))
# SCORE Avg MALE

np.mean([two for (one,two) in scores])
# Теперь моя любимая модель - регрессия 7 дней

scores = []

for i in range(len(rus.T.columns)):

    obl = pd.DataFrame(rus.T.iloc[:,i].copy(), columns=[rus.T.iloc[:,i].name]) 

    # Исправим данные, где кол-во растет, потом падает: пусть только растет
    prev = 0
    obl_corrected = []

    for one in obl.iloc[:,0]:
        if one < prev:
            obl_corrected[-1] = one

        obl_corrected.append(one)  
        prev = one

    obl.iloc[:,0] = obl_corrected

    # Индекс приводим к формату даты
    obl.index = pd.to_datetime(obl.index) 

    ## Делаем признаки

    # Порядковый номер дня в году - "дата" для всех
    obl['day_of_year'] = obl.index.dayofyear
    
    # Х
    X_cols = ['day_of_year']
    
    # до 19 обучаем, c 20 будем тестироваться
    train = obl.loc[(obl.index>'2020-04-12') & (obl.index<'2020-04-20')]
    test  = obl.loc[obl.index>'2020-04-19']
    
    # Х в лог
    X_train = np.log(train[X_cols] + 0.5)
    y_train = np.log(train.iloc[:,0] + 0.5)
    X_test = np.log(test[X_cols] + 0.5)
    y_test = np.log(test.iloc[:,0] + 0.5)

    # Скалируем признаки, для регрессии с регуляризацией хорошо
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    # print(scaler.mean_, scaler.var_)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Строим регрессию
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(X_train_scaled, y_train.values)

    # Функции для 2 метрик: первая - конкурсная, вторая - процент отклонения от истины
    def MALE(pred, true):
    #    print(np.log10((pred + 1) / (true + 1)))
        return np.mean(np.abs(np.log10((pred + 1) / (true + 1))))

    def AvgProc(pred, true):
    #    print((pred-true)/true)
        return np.mean(np.abs((pred-true)/true))

    # Приводим у к кол-ву случаев
    y_pred_test_exp = np.round(np.exp(reg.predict(X_test_scaled))-0.5,0)
    y_pred_train_exp = np.round(np.exp(reg.predict(X_train_scaled))-0.5,0)
    y_test_exp = np.round(np.exp(y_test)-0.5,0)
    y_train_exp = np.round(np.exp(y_train)-0.5,0)

    # Обрабатываем "Архангельский прыжок" (резкое увеличение случаев - увеличиваем предсказ на 2/3 от скачка)
    if i==2: 
        y_pred_test_exp = y_pred_test_exp + (y_train_exp[-1]-y_train_exp[-2])*(2/3)

    # Выводим результаты
    print(obl.columns[0])
    print('coefs=', reg.coef_, 'const=', reg.intercept_)
    print('MALE test = ', MALE(y_pred_test_exp, y_test_exp), 'MALE train =', MALE(y_pred_train_exp, y_train_exp))
    print('AvgProc test = ', AvgProc(y_pred_test_exp, y_test_exp), 'AvgProc train =', AvgProc(y_pred_train_exp, y_train_exp))

    # Отрисуем предсказ
    plt.figure(figsize=(15,10))
    for x,y in zip(y_test.index.dayofyear, y_test_exp):
        label = "{:.0f}".format(y)
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,-15), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center

    for x,y in zip(y_test.index.dayofyear, y_pred_test_exp):
        label = "{:.0f}".format(y)
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,-15), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center

    plt.title(obl.columns[0])

    plt.plot(y_test.index.dayofyear, y_pred_test_exp, 'ro-')
    plt.plot(y_test.index.dayofyear, y_test_exp, 'bo-')

    plt.grid()
    plt.show()

    # Отрисуем модель
    plt.figure(figsize=(15,10))

    for x,y in zip(y_train.index.dayofyear, y_train_exp):
        label = "{:.0f}".format(y)
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,-15), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center

    for x,y in zip(y_train.index.dayofyear, y_pred_train_exp):
        label = "{:.0f}".format(y)
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,-15), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center

    plt.title(obl.columns[0])

    plt.plot(y_train.index.dayofyear, y_pred_train_exp, 'ro-')
    plt.plot(y_train.index.dayofyear, y_train_exp, 'bo-')

    plt.grid()
    plt.show()
    
    scores.append((obl.columns[0], MALE(y_pred_test_exp, y_test_exp)))
    


print('Avg MALE = ', np.mean([two for (one,two) in scores]))
scores = []

for i in range(len(rus.T.columns)):

    obl = pd.DataFrame(rus.T.iloc[:,i].copy(), columns=[rus.T.iloc[:,i].name]) 

    # Исправим данные, где кол-во растет, потом падает: пусть только растет
    prev = 0
    obl_corrected = []

    for one in obl.iloc[:,0]:
        if one < prev:
            obl_corrected[-1] = one

        obl_corrected.append(one)  
        prev = one

    obl.iloc[:,0] = obl_corrected

    # Индекс приводим к формату даты
    obl.index = pd.to_datetime(obl.index) 

    ## Делаем признаки

    # Порядковый нмер дня в году - "дата" для всех
    obl['day_of_year'] = obl.index.dayofyear 

    # Создаем day7 - на 1 возрастают дни (от начала года) со времени, когда кейсов стали больше 7 (после какого-то значения рост более стабильный)
    obl['day7'] = 0 
    idx_day_7 = obl.iloc[:,0]>7
    obl.loc[idx_day_7,'day7'] = obl.loc[idx_day_7,'day7'].index.dayofyear
    day_day7 = obl.loc[obl['day7'] > 0, 'day7'].min() - 1
    obl.loc[obl['day7'] > 0, 'day7'] -= day_day7 

    # Создаем apr6 - на 1 возрастают дни с 6 апреля - когда должны заметить карантин (30.03+7 дней) - траектория изменилась
    obl['apr6'] = 0 
    idx_apr_6 = obl.index >= '2020-04-06'
    obl.loc[idx_apr_6,'apr6'] = obl.loc[idx_apr_6,'apr6'].index.dayofyear
    day_apr6 = obl.loc[obl['apr6'] > 0, 'apr6'].min() - 1
    obl.loc[obl['apr6'] > 0, 'apr6'] -= day_apr6 
    
    # Создаем last7 - на 1 возрастают дни для последних 7 дней - больше учитываем последние данные
    obl['last7'] = 0 
    idx_last_7 = obl.index >= obl.index[-7]
    obl.loc[idx_last_7,'last7'] = obl.loc[idx_last_7,'last7'].index.dayofyear
    day_last7 = obl.loc[obl['last7'] > 0, 'last7'].min() - 1
    obl.loc[obl['last7'] > 0, 'last7'] -= day_last7
    
    # И сделаем квадрат от last7
    obl['last7_2'] = obl['last7'] ** 2

    # Давайте посмотрим на 2 режима - day7 (сверху) и apr6. Что видно?
#     plt.figure(figsize=(15,10))
    pos_cases = obl.iloc[:,0]>0
#     plt.plot(np.log(obl.loc[pos_cases].iloc[:,0]))
#     plt.scatter(obl.loc[pos_cases,'day7'].index,(obl.loc[pos_cases,'day7']>0)+0.1)
#     plt.scatter(obl.loc[pos_cases,'apr6'].index,(obl.loc[pos_cases,'apr6']>0))
#     plt.title(obl.columns[0])
#     plt.show()
    
    # Собираем Х
    X_cols = ['day_of_year','day7','apr6','last7', 'last7_2']
    
    # до конца обучаем, c 27 будем предиктить
    train = obl.loc[(pos_cases)]
    
    wk3 = ['2020-04-27','2020-04-28','2020-04-29','2020-04-30','2020-05-01','2020-05-02','2020-05-03']
    test  = pd.DataFrame(index = pd.to_datetime(wk3), data = [], columns = X_cols)
    test['day_of_year'] = test.index.dayofyear
    count_down = test.index.dayofyear - test.index.dayofyear.min()
    test['day7'] = count_down + train['day7'].max() + 1
    test['apr6'] = count_down + train['apr6'].max() + 1
    test['last7'] = count_down + 8
    test['last7_2'] = test['last7'] ** 2

    # Х в лог
    X_train = np.log(train[X_cols] + 0.5)
    y_train = np.log(train.iloc[:,0] + 0.5)
    X_test = np.log(test[X_cols] + 0.5)
#    y_test = np.log(test.iloc[:,0] + 0.5)

    # Скалируем признаки, для регрессии с регуляризацией хорошо
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    # print(scaler.mean_, scaler.var_)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Строим регрессию
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(X_train_scaled, y_train.values)

    # Функции для 2 метрик: первая - конкурсная, вторая - процент отклонения от истины
    def MALE(pred, true):
    #    print(np.log10((pred + 1) / (true + 1)))
        return np.mean(np.abs(np.log10((pred + 1) / (true + 1))))

    def AvgProc(pred, true):
    #    print((pred-true)/true)
        return np.mean(np.abs((pred-true)/true))

    # Приводим у к кол-ву случаев
    y_pred_test_exp = np.round(np.exp(reg.predict(X_test_scaled))-0.5,0)
    y_pred_train_exp = np.round(np.exp(reg.predict(X_train_scaled))-0.5,0)
#    y_test_exp = np.round(np.exp(y_test)-0.5,0)
    y_train_exp = np.round(np.exp(y_train)-0.5,0)

    # Обрабатываем "Архангельский прыжок" (резкое увеличение случаев - увеличиваем предсказ на 2/3 от скачка)
    if i==2: 
        y_pred_test_exp = y_pred_test_exp + (y_train_exp[-1]-y_train_exp[-2])*(2/3)

    # Выводим результаты
    print(obl.columns[0])
    print('coefs=', reg.coef_, 'const=', reg.intercept_)
    print('MALE train =', MALE(y_pred_train_exp, y_train_exp))
    print('AvgProc train =', AvgProc(y_pred_train_exp, y_train_exp))

    # Отрисуем предсказ
    plt.figure(figsize=(15,10))
#     for x,y in zip(y_test.index.dayofyear, y_test_exp):
#         label = "{:.0f}".format(y)
#         plt.annotate(label, # this is the text
#                      (x,y), # this is the point to label
#                      textcoords="offset points", # how to position the text
#                      xytext=(0,-15), # distance from text to points (x,y)
#                      ha='center') # horizontal alignment can be left, right or center

    for x,y in zip(wk3, y_pred_test_exp):
        label = "{:.0f}".format(y)
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,-15), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center

    plt.title(obl.columns[0]+' predict')

    plt.plot(wk3, y_pred_test_exp, 'ro-')
#     plt.plot(y_test.index.dayofyear, y_test_exp, 'bo-')

    plt.grid()
    plt.show()

    # Отрисуем модель
    plt.figure(figsize=(15,10))

    for x,y in zip(y_train.index, y_train_exp):
        label = "{:.0f}".format(y)
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,-15), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center

    for x,y in zip(y_train.index, y_pred_train_exp):
        label = "{:.0f}".format(y)
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,-15), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center

    plt.title(obl.columns[0]+' fit')

    plt.plot(y_train.index, y_pred_train_exp, 'ro-')
    plt.plot(y_train.index, y_train_exp, 'bo-')

    plt.grid()
    plt.show()
    
    scores.append((obl.columns[0], y_pred_test_exp))
scores
sub_reg = pd.DataFrame([one[0] for one in scores], columns=['reg'])
preds = pd.DataFrame([one[1] for one in scores], columns=wk3).astype(int)
sub_reg = pd.concat([sub_reg, preds], axis = 1)
sub_reg = sub_reg.set_index('reg').join(russia_regions.set_index('csse_province_state')['iso_code'])
sub = sub_reg.set_index('iso_code').T 
sub
submit = pd.DataFrame()

for col in sub.columns:
    submit = pd.concat([submit,pd.DataFrame(data = {'region':7*[col], 'prediction_confirmed':sub[col]})])
    
submit.reset_index(inplace=True)
submit['date'] = submit['index']
del submit['index']
sample = pd.read_csv('/kaggle/input/prediction-format/sample_submission_JgJvhOF.csv')
sample
submit_joined = sample.set_index(['date','region']).join(submit.set_index(['date','region']), rsuffix = 'r')
submit_joined
submit_joined.loc[submit_joined['prediction_confirmedr'].notnull(),'prediction_confirmed'] = \
    submit_joined.loc[submit_joined['prediction_confirmedr'].notnull(),'prediction_confirmedr'] 
del submit_joined['prediction_confirmedr']
submit_joined.reset_index(inplace=True)
submit_joined['prediction_confirmed'] = submit_joined['prediction_confirmed'].astype(int)
submit_joined.to_csv('submit_wk3.csv', index=None)