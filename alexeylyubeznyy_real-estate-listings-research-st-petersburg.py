import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

data = pd.read_csv('../input/real_estate_data.csv', index_col=0)

pd.set_option('display.max_columns', 500)

data.head()
data.info()
# функция, получающая на вход название признака, и возвращающая число пропущенных значений

def print_nan_ratio(column):

    return print('Пропущено {:.1%}'. format(data[column].isna().value_counts()[1] / len(data), 2) + ' значений')
print_nan_ratio('ceiling_height')

ceiling_medians = data.pivot_table(index='floors_total', values='ceiling_height', aggfunc=['median', 'count'])

ceiling_medians.columns = ['median', 'count']

ceiling_medians.head()
# заменяем значения высоты потолков на общую медиану для отсутствующих значений этажности дома

for floor in ceiling_medians.index:

    data.loc[(data['floors_total'].isna()) | 

             ((data['floors_total'] == floor) & (ceiling_medians.loc[floor, 'count'] == 0)), 

             'ceiling_height'] = data['ceiling_height'].describe()[5]



# медианы из сгруппированной таблицы вставляем на места пропущенных значений высоты потолков для дома соответствующей этажности

for floor in ceiling_medians.index:

    data.loc[(data['ceiling_height'].isna()) & 

             (data['floors_total'] == floor), 

             'ceiling_height'] = ceiling_medians.loc[floor, 'median']



# посчитаем количество пропущенных значений после проделанной замены

data['ceiling_height'].isna().value_counts()
print_nan_ratio('floors_total')

data.dropna(subset=['floors_total'], inplace=True)
print_nan_ratio('living_area')



from scipy.optimize import minimize



# сначала очистим данные от пропущенных значений

living_area_notna = data[data['living_area'].notna()].reset_index(drop=True)



# зададим квадратичную функцию ошибки, зависящую от вектора параметров

def error_function(w):

    sum = 0

    for i in range(len(living_area_notna)):

        sum += (living_area_notna.loc[i, 'living_area'] - (w[0] + w[1] * living_area_notna.loc[i, 'total_area'])) ** 2

    return sum



# решим задачу минимизации заданной функции с некоторым начальным приближением

result = minimize(error_function, np.array([0, 0]))



# зададим линейную функцию, отображающую наше решение

def linear_function(w0, w1, x):

    return (w0 + w1 * x)



# Изобразим решение на графике

sns.relplot(x='total_area', y='living_area', data=data)



total_area = np.linspace(0, 800)

line = plt.plot(total_area, linear_function(result.x[0], result.x[1], total_area), color='crimson', linewidth=3)

plt.title('Зависимость жилой площади от общей площади недвижимости')

plt.xlabel('Общая площадь')

plt.ylabel('Жилая площадь')

plt.show()
# заменим все пропущенные значения на основе оцененной зависимости

data.loc[data['living_area'].isna(), 

         'living_area'] = result.x[0] + result.x[1] * data.loc[data['living_area'].isna(), 'total_area']



# проверим, все ли пропущенные значения мы обработали

data['living_area'].isna().value_counts()
# вычислим долю пропущенных значений

print_nan_ratio('kitchen_area')



# сначала очистим данные от пропущенных значений

kitchen_area_notna = data[data['kitchen_area'].notna()].reset_index(drop=True)



# зададим квадратичную функцию ошибки, зависящую от вектора параметров

def error_function(w):

    sum = 0

    for i in range(len(kitchen_area_notna)):

        sum += (kitchen_area_notna.loc[i, 'kitchen_area'] - (w[0] + w[1] * kitchen_area_notna.loc[i, 'total_area'])) ** 2

    return sum



# решим задачу минимизации заданной функции с некоторым начальным приближением

result = minimize(error_function, np.array([0, 0]))



# Изобразим решение на графике

sns.relplot(x='total_area', y='kitchen_area', data=data)



total_area = np.linspace(0, 800)

line = plt.plot(total_area, linear_function(result.x[0], result.x[1], total_area), color='crimson', linewidth=3)

plt.title('Зависимость площади кухни от общей площади недвижимости')

plt.xlabel('Общая площадь')

plt.ylabel('Площадь кухни')

plt.show()
# заменим пропущенные значения площади кухнина в квартирах-студиях на нули

data.loc[data['studio'] == True, 'kitchen_area'] = 0



# заполним пропущенные значения для квартир, которые не являются студиями

data.loc[data['kitchen_area'].isna(), 

         'kitchen_area'] = result.x[0] + result.x[1] * data.loc[data['kitchen_area'].isna(), 'total_area']



# проверим, все ли пропущенные значения мы обработали

data['kitchen_area'].isna().value_counts()
print_nan_ratio('balcony')
data['balcony'].fillna(0, inplace=True)

data['balcony'].isna().value_counts()
print_nan_ratio('locality_name')

data.dropna(subset=['locality_name'], inplace=True)
print_nan_ratio('cityCenters_nearest')
round(len(data[(data['locality_name'] == 'Санкт-Петербург') & (data['cityCenters_nearest'].isna())]) / len(data.reset_index(drop=True).query('locality_name == "Санкт-Петербург"')), 3)
print_nan_ratio('is_apartment')
print_nan_ratio('airports_nearest')
print_nan_ratio('parks_around3000')
print_nan_ratio('parks_nearest')
print_nan_ratio('ponds_around3000')
print_nan_ratio('ponds_nearest')
print_nan_ratio('days_exposition')
# заменим пропуски на медианные значения

data.loc[data['days_exposition'].isna(), 'days_exposition'] = data['days_exposition'].describe()[5]
# заменим тип данных по дате публикации со строчного на datetime

data['first_day_exposition'] = pd.to_datetime(data['first_day_exposition'], format='%Y-%m-%dT%H:%M:%S')
data['price_per_sqm'] = round(data['last_price'] / data['total_area'], 1)
data['weekday'] = data['first_day_exposition'].dt.weekday

data['month'] = data['first_day_exposition'].dt.month

data['year'] = data['first_day_exposition'].dt.year
data.loc[data['floor'] == 1, 'floor_category'] = 'первый'

data.loc[data['floor'] == data['floors_total'], 'floor_category'] = 'последний'

data.loc[data['floor_category'].isna(), 'floor_category'] = 'другой'
data['living_area_ratio'] = round(data['living_area'] / data['total_area'], 3)

data['kitchen_area_ratio'] = round(data['kitchen_area'] / data['total_area'], 3)
# функция, получающая на вход название признака, и возвращающая границы "усов"

def det_whiskers(column):

    stat = data[column].describe()

    

    # межквартирльный размах

    iqr = stat[6] - stat[4]

    

    # левый и правый "ус"    

    left_whisker = round(stat[4] - 1.5 * iqr, 2)

    right_whisker = round(stat[6] + 1.5 * iqr, 2)



    # левый "ус" не должен быть меньше минимального значения

    if left_whisker < stat[3]: left_whisker = stat[3] 

        

    # правый "ус" не должен быть больше максимального значения

    if right_whisker > stat[7]: right_whisker = stat[7]

        

    return [left_whisker, right_whisker]
det_whiskers('total_area')
data['total_area'].describe()
# построим диаграмму размаха, ограничив площадь 150 кв.м

sns.boxplot(data['total_area'], color='lightsteelblue')

plt.title('Диаграмма размаха')

plt.xlabel('Общая площадь, кв.м')

plt.xlim(0, 150)

plt.show()
# Построим диаграмму для объектов с площадью менее 120 кв.м.

sns.distplot(data['total_area'], kde=False, bins=45)

plt.xlim(0, 120)

plt.title('Гистограмма общей площади')

plt.xlabel('Общая площадь, кв.м')

plt.ylabel('Частота')

plt.show()
# зададим функцию, вычисляющую долю аномальных значений

def print_anomalies_ratio(column):

    return 'Аномальные значения составляют {:.1%}'. format(len(data.loc[data[column] > det_whiskers(column)[1]]) / len(data)) + ' от всей выборки'

print_anomalies_ratio('total_area')
data['too_large_area'] = data['total_area'] > det_whiskers('total_area')[1]
det_whiskers('last_price')
data['last_price'].describe()
sns.boxplot(data['last_price'], color='tab:red')

plt.title('Диаграмма размаха')

plt.xlabel('Цены, 10 млн рублей')

_ = plt.xlim(0, 13000000)
sns.distplot(data['last_price'], kde=False, bins=450, color='tab:red')

plt.xlim(0, 13000000)

plt.title('Гистограмма цен на момент снятия публикации')

plt.xlabel('Цены, 10 млн рублей')

_ = plt.ylabel('Частота')
print_anomalies_ratio('last_price')
data['too_expensive'] = data['last_price'] > det_whiskers('last_price')[1]
det_whiskers('rooms')
data['rooms'].describe()
sns.boxplot(data['rooms'], color='tab:cyan')

plt.title('Диаграмма размаха')

plt.xlabel('Число комнат, шт.')

_ = plt.xlim(0, 10)
sns.distplot(data['rooms'], kde=False, bins=20, color='tab:cyan')

plt.xlim(0, 6)

plt.title('Гистограмма числа комнат')

plt.xlabel('Число комнат, шт.')

_ = plt.ylabel('Частота')
print_anomalies_ratio('rooms')
data['too_many_rooms'] = data['rooms'] > det_whiskers('rooms')[1]
det_whiskers('ceiling_height')
data['ceiling_height'].describe()
sns.boxplot(data['ceiling_height'], color='tab:purple')

plt.title('Диаграмма размаха')

plt.xlabel('Высота потолков, м')

_ = plt.xlim(1, 4)
sns.distplot(data['ceiling_height'], kde=False, bins=750, color='tab:purple')

plt.xlim(det_whiskers('ceiling_height')[0], det_whiskers('ceiling_height')[1])

plt.title('Гистограмма высоты потолков')

plt.xlabel('Высота потолков, м')

_ = plt.ylabel('Частота')
print_anomalies_ratio('ceiling_height')
data['unusual_ceiling'] = (data['ceiling_height'] > det_whiskers('ceiling_height')[1]) | (data['ceiling_height'] < det_whiskers('ceiling_height')[0])
det_whiskers('days_exposition')
sns.boxplot(data['days_exposition'], color='thistle')

plt.title('Диаграмма размаха')

plt.xlabel('Срок размещения публикации, дней')

_ = plt.xlim(0, 450)
data.plot(y='days_exposition', 

          kind='hist', 

          color='thistle', 

          range=(det_whiskers('days_exposition')[0], det_whiskers('days_exposition')[1]), 

          bins=8)

plt.title('Гистограмма срока публикации')

plt.xlabel('Срок размещения публикации, дней')

_ = plt.ylabel('Частота')
data['days_exposition'].describe()
data['too_slow'] = data['days_exposition'] > det_whiskers('days_exposition')[1]

data['too_fast'] = data['days_exposition'] < 30
good_data = data[(data['too_large_area'] == False) & 

     (data['too_expensive'] == False) &

     (data['too_many_rooms'] == False) &

     (data['unusual_ceiling'] == False) &

     (data['too_slow'] == False) &

     (data['too_fast'] == False)].reset_index(drop=True)



print('Доля чистых значений составила: {:.1%}'. format(len(good_data) / len(data)))
_ = sns.pairplot(good_data, 

                 vars=['total_area', 'last_price', 'rooms', 'ceiling_height'],

                 height=3)
fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True, figsize=(17, 6), gridspec_kw={'wspace': 0})

fig.suptitle('Зависимость цены от следующих факторов:')



# построим график зависимости цены от площади

ax1.set_title('Площадь')

ax1.set_ylabel('Цена, 10 млн рублей')

ax1.set_xlabel('Общая площадь квартиры, кв.м')

ax1.text(60, 11000000, 

         'коэффициент корреляции: ' + str(round(good_data['last_price'].corr(good_data['total_area']), 2)), 

         style='italic')

hb1 = ax1.hexbin(good_data['total_area'], good_data['last_price'], cmap='BuPu')

cb1 = fig.colorbar(hb1, ax=ax1)

cb1.set_label('Частота')



# построим график зависимости цены от расстояния до центра города

ax2.set_title('Расстояние до центра города')

ax2.set_ylabel('Цена, 10 млн рублей')

ax2.set_xlabel('Расстояние до центра города, м')

ax2.text(30000, 11000000, 

         'коэффициент корреляции: ' + str(round(good_data['last_price'].corr(good_data['cityCenters_nearest']), 2)), 

         style='italic')

hb2 = ax2.hexbin(good_data['cityCenters_nearest'], good_data['last_price'], cmap='BuPu')

cb2 = fig.colorbar(hb2, ax=ax2)

cb2.set_label('Частота')



plt.tight_layout(pad=3)

plt.show()
for column in ['rooms', 'floor_category', 'weekday', 'month', 'year']:

    sns.catplot(x=column, y="last_price", kind="box", data=good_data, palette='husl')
median_price_per_sqm = good_data.pivot_table(index='locality_name', values='price_per_sqm', aggfunc=['count', 'median'])

median_price_per_sqm.columns = ['count', 'median']

median_price_per_sqm.sort_values('count', ascending=False).head(10)
data['citycenters_km'] = round(data['cityCenters_nearest'] / 1000, 0)

data['citycenters_km'] = data['citycenters_km'].astype('int', errors='ignore')

good_data = data[(data['too_large_area'] == False) & 

     (data['too_expensive'] == False) &

     (data['too_many_rooms'] == False) &

     (data['unusual_ceiling'] == False) &

     (data['too_slow'] == False) &

     (data['too_fast'] == False)].reset_index(drop=True)



good_data.loc[(good_data['locality_name'] == 'Санкт-Петербург') & (good_data['citycenters_km'].notna()), 'citycenters_km'].apply(round)

spb_data = good_data.query('locality_name == "Санкт-Петербург"')

spb_center_nearest = spb_data.pivot_table(index='citycenters_km', values=['price_per_sqm', 'last_price'], aggfunc='median')

spb_center_nearest.head()
sns.relplot(x=spb_center_nearest.index, y='last_price', data=spb_center_nearest, height=4, aspect=1.4)

plt.annotate('Разница средних значений в 1 млн рублей', xy=(8, 6110000), xytext=(15, 7000000),

            arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('', xy=(9, 5000000), xytext=(15, 7000000),

            arrowprops=dict(facecolor='black', shrink=0.05))

plt.title('Зависимость средней цены от расстояния до центра Санкт-Петербурга')

plt.ylabel('Цена недвижимости, руб.')

plt.xlabel('Расстояние до центра, км')

plt.show()
spb_center = good_data[(good_data['citycenters_km'] <= 8) & (good_data['locality_name'] == 'Санкт-Петербург')]
print(spb_center['total_area'].describe())

sns.boxplot(spb_center['total_area'], color='lightsteelblue')

plt.title('Диаграмма размаха')

plt.xlabel('Общая площадь, кв. м')

plt.show()
print(spb_center['last_price'].describe())

sns.boxplot(spb_center['last_price'], color='tab:red')

plt.title('Диаграмма размаха')

plt.xlabel('Цена, 10 млн рублей')

plt.show()
print(spb_center['rooms'].describe())

sns.boxplot(spb_center['rooms'], color='tab:cyan')

plt.title('Диаграмма размаха')

plt.xlabel('Число комнат, шт.')

plt.show()
print(spb_center['ceiling_height'].describe())

sns.boxplot(spb_center['ceiling_height'], color='tab:purple')

plt.title('Диаграмма размаха')

plt.xlabel('Цена, 10 млн рублей')

plt.show()
print(spb_center['days_exposition'].describe())

sns.boxplot(spb_center['days_exposition'], color='thistle')

plt.title('Диаграмма размаха')

plt.xlabel('Срок размещения публикации, дней')

plt.xlim(0, 450)

plt.show()
for column in ['rooms', 'floor_category', 'citycenters_km', 'weekday', 'month', 'year']:

    sns.catplot(x=column, y="last_price", kind="box", data=spb_center, palette='husl')