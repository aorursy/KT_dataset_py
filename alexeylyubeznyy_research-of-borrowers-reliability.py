import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import colors

data = pd.read_csv('../input/borrowers_reliability.csv', index_col=0)

data.info()

data.head(15)
!pip install pymystem3
ax1 = plt.hist(data['total_income'], bins=int(np.sqrt(len(data))))



# N — счетчик в каждом баре (столбце), bins — нижняя граница бара

N, bins, patches = plt.hist(data['total_income'], bins=int(np.sqrt(len(data))))

                               

# Для кодирования цвета будем использовать относительную высоту

fracs = N / N.max()



# нормализуем fracs до промежутка между 0 и 1 для полноценного цветового диапазона

norm = colors.Normalize(fracs.min(), fracs.max())



# нужно пройтись циклом по полученным объектам и установить цвет для каждого в отдельности

for thisfrac, thispatch in zip(fracs, patches):

    color = plt.cm.viridis(norm(thisfrac))

    thispatch.set_facecolor(color)



ax1 = plt.xlabel('$Ежемесячный$ $доход, руб.$', fontsize=11)

ax1 = plt.ylabel('$Клиенты, чел.$', fontsize=11)

plt.title('Гистограмма ежемесячных доходов', fontsize=13)

plt.text(1000000, 1500, '$\mu=167422,\sigma=102972$', fontsize=13)

plt.show()

print(round(data['total_income'].describe()))
plt.xticks(np.arange((data['dob_years'].min()), (data['dob_years'].max() + 5), 5.0))

ax2 = plt.hist(data['dob_years'], bins=15, color='tab:purple')



# N — счетчик в каждом баре (столбце), bins — нижняя граница бара

N, bins, patches = plt.hist(data['dob_years'], bins=15)

                               

# Для кодирования цвета будем использовать относительную высоту

fracs = N / N.max()



# нормализуем fracs до промежутка между 0 и 1 для полноценного цветового диапазона

norm = colors.Normalize(fracs.min(), fracs.max())



# нужно пройтись циклом по полученным объектам и установить цвет для каждого в отдельности

for thisfrac, thispatch in zip(fracs, patches):

    color = plt.cm.viridis(norm(thisfrac))

    thispatch.set_facecolor(color)



ax2 = plt.xlabel('$Возраст, лет$', fontsize=11)

ax2 = plt.ylabel('$Клиенты, чел.$', fontsize=11)

plt.title('Гистограмма возраста клиентов', fontsize=13)

plt.show()
plt.xticks(np.arange((data['children'].min()-1), (data['children'].max() + 1), 1.0))

ax3 = plt.hist(data['children'], bins=25)



# N — счетчик в каждом баре (столбце), bins — нижняя граница бара

N, bins, patches = plt.hist(data['children'], bins=25)

                               

# Для кодирования цвета будем использовать относительную высоту

fracs = N / N.max()



# нормализуем fracs до промежутка между 0 и 1 для полноценного цветового диапазона

norm = colors.Normalize(fracs.min(), fracs.max())



# нужно пройтись циклом по полученным объектам и установить цвет для каждого в отдельности

for thisfrac, thispatch in zip(fracs, patches):

    color = plt.cm.viridis(norm(thisfrac))

    thispatch.set_facecolor(color)



ax3 = plt.xlabel('$Дети, чел.$', fontsize=11)

ax3 = plt.ylabel('$Клиенты, чел.$', fontsize=11)

plt.title('Гистограмма численности детей', fontsize=13)

plt.show()
ax4 = plt.hist(data['days_employed'], bins=30, color='tab:blue')



# N — счетчик в каждом баре (столбце), bins — нижняя граница бара

N, bins, patches = plt.hist(data['days_employed'], bins=30)

                               

# Для кодирования цвета будем использовать относительную высоту

fracs = N / N.max()



# нормализуем fracs до промежутка между 0 и 1 для полноценного цветового диапазона

norm = colors.Normalize(fracs.min(), fracs.max())



# нужно пройтись циклом по полученным объектам и установить цвет для каждого в отдельности

for thisfrac, thispatch in zip(fracs, patches):

    color = plt.cm.viridis(norm(thisfrac))

    thispatch.set_facecolor(color)



ax4 = plt.xlabel('$Стаж, дней$', fontsize=11)

ax4 = plt.ylabel('$Клиенты, чел.$', fontsize=11)

plt.title('Гистограмма дней трудового стажа', fontsize=13)

plt.show()
# сделаем выборку из нулевого возраста и соотвествующих значений трудового стажа

zero_age = data.loc[data['dob_years'] == 0, 'dob_years']

days_employed_for_age_null = data.loc[data['dob_years'] == 0, 'days_employed']



# сделаем выборку из положительных значений трудового стажа и возраста клиентов (без нулевых возрастов)

age_for_positive_days_employed = data.loc[(data['days_employed'] > 0) & (data['dob_years'] != 0), 'dob_years']

positive_days_employed = data.loc[(data['days_employed'] > 0) & (data['dob_years'] != 0), 'days_employed']



# сделаем выборку из отрицательных значений трудового стажа и возраста клиентов (без нулевых возрастов)

age_for_negative_days_employed = data.loc[(data['days_employed'] < 0) & (data['dob_years'] != 0), 'dob_years']

negative_days_employed = data.loc[(data['days_employed'] < 0) & (data['dob_years'] != 0), 'days_employed']



# строим диаграмму

plt.title('Точечная диаграмма зависимости трудового стажа от возраста клиента')

plt.xlabel('Возраст клиента, лет')

plt.ylabel('Трудовой стаж, дней')

plt.scatter(zero_age, days_employed_for_age_null, 

            c = 'c', marker = 'x', label = 'Стаж у клиентов с нулевым возрастом')

plt.scatter(age_for_positive_days_employed, positive_days_employed, 

            c = 'g', marker = '+', label = 'Положительный трудовой стаж')

plt.scatter(age_for_negative_days_employed, negative_days_employed, 

            c = 'r', marker = '|', label = 'Отрицательный трудовой стаж')

plt.legend()

plt.show()
# Для этого проверим, в каких типах занятости встречаются пропущенные значения доходов.

data.loc[data['total_income'].isnull(), 'income_type'].value_counts()
# Найдем медианные уровни дохода для каждого типа занятости

medians_total_income = data.groupby('income_type')['total_income'].median()

round(medians_total_income)
# Заменим пропущенные значения ежемесячного дохода на медианы по типам занятости.

data.loc[(data['total_income'].isnull()) & (data['income_type'] == 'сотрудник'), 'total_income'] = medians_total_income[6]

data.loc[(data['total_income'].isnull()) & (data['income_type'] == 'компаньон'), 'total_income'] = medians_total_income[3]

data.loc[(data['total_income'].isnull()) & (data['income_type'] == 'пенсионер'), 'total_income'] = medians_total_income[4]

data.loc[(data['total_income'].isnull()) & (data['income_type'] == 'госслужащий'), 'total_income'] = medians_total_income[2]

data.loc[(data['total_income'].isnull()) & (data['income_type'] == 'предприниматель'), 'total_income'] = medians_total_income[5]
# Посмотрим, какой тип занятости имеют клиенты с ошибочным возрастом.

data.loc[data['dob_years'] == 0, 'income_type'].value_counts()
# Найдем медианные значения возрастов клиентов для каждого типа занятости:

age_medians = data.groupby('income_type')['dob_years'].median()

age_medians
# Теперь заменим нулевые значения на медианы по каждому типу занятости.

data.loc[(data['dob_years'] == 0) & (data['income_type'] == 'сотрудник'), 'dob_years'] = age_medians[6]

data.loc[(data['dob_years'] == 0) & (data['income_type'] == 'пенсионер'), 'dob_years'] = age_medians[4]

data.loc[(data['dob_years'] == 0) & (data['income_type'] == 'компаньон'), 'dob_years'] = age_medians[3]

data.loc[(data['dob_years'] == 0) & (data['income_type'] == 'госслужащий'), 'dob_years'] = age_medians[2]



# Проверим, все ли нули мы исправили.

data.loc[data['dob_years'] == 0, 'dob_years'].value_counts()
# Проверим, является ли отсутствие данных по трудовому стажу причиной отсутствия данных по доходам, и наоборот.

count = 0

for i in range(len(data)):

    if pd.isna(data.loc[i, 'days_employed']) == pd.isna(data.loc[i, 'total_income']) == True:

        count += 1

count
age_statistics = data['dob_years'].describe()

age_statistics[4:7]
# напишем функцию, которая принимает на вход возраст клиента и возвращает возрастную категорию

def age_group(age):

    if age <= age_statistics[4]: return 'до 34'

    elif age_statistics[4] < age <= age_statistics[5]: return '34-43'

    elif age_statistics[5] < age <= age_statistics[6]: return '43-53'

    else: return '53+'

    

data['age_group'] = data['dob_years'].apply(age_group)

data.head(2)
# Найдем медианное значение трудового стажа для каждой возрастной группы.

medians_of_days_employed = data.groupby('age_group')['days_employed'].median()

round(medians_of_days_employed, 0)
# заменим пропущенные значения трудового стажа на медиану из возрастных категорий

data.loc[(data['days_employed'].isnull()) & (data['age_group'] == '34-43'), 'days_employed'] = medians_of_days_employed[0]

data.loc[(data['days_employed'].isnull()) & (data['age_group'] == '43-53'), 'days_employed'] = medians_of_days_employed[1]

data.loc[(data['days_employed'].isnull()) & (data['age_group'] == '53+'), 'days_employed'] = medians_of_days_employed[2]

data.loc[(data['days_employed'].isnull()) & (data['age_group'] == 'до 34'), 'days_employed'] = medians_of_days_employed[3]
data.info()
data.dtypes
data['dob_years'] = data['dob_years'].astype('int')
data.dtypes
# сосчитаем всех клиентов по уровням образования

data['education'].value_counts()
print('Количество неправильно заполненных ячеек со средним уровнем образования:', 772 + 711)

print('Доля ошибок от общего числа клиентов со средним образованием: {:.2%}'. format((772 + 711) / (13750 + 772 + 711)))
# сделаем все буквы в значениях образования строчными

data['education'] = data['education'].str.lower()



# посмотрим, что получилось, одновременно посчитая количество клиентов по каждому уровню образования

data['education'].value_counts()
# Посчитаем количество дубликатов в массиве.

print('Число дубликатов в массиве данных:', data.duplicated().sum())

print('Число полностью идентичных строк:', data.duplicated(keep = False).sum())

print('Доля дубликатов из общей длины массива: {:.2%}'.format(data.duplicated().sum() / len(data)))
# Для наглядности выведем первые 5 строк полностью "идентичных" клиентов.

data[data.duplicated(keep = False)].sort_values('dob_years', ascending = False).head()
# Удалим одинаковые строки.

data = data.drop_duplicates().reset_index(drop = True)



# Проверим число дубликатов.

print('Число дубликатов в массиве данных:', data.duplicated().sum())
# Импортируем библиотеку *pymystem3* и *collections*

from pymystem3 import Mystem

from collections import Counter

m = Mystem()



# Посчитаем все варианты целей кредита.

data['purpose'].value_counts()
# основные ключевые слова можно выделить вручную

categories = ["сдача", "коммерческий", "жилье", "образование", "свадьба", "недвижимость", "автомобиль"]



# проведем лемматизацию,

# одновременно заменив полученный список лемм в каждой строке на главное ключевое слово из списка категорий



def lemmatize(text):

    lemma = m.lemmatize(text)

    for word in categories:

        if word in lemma:

            lemma = word

    return lemma



data['purpose_group'] = data['purpose'].apply(lemmatize)        

data.head(2)
# Посчитаем все возможные категории целей кредита.

data['purpose_group'].value_counts()
data.loc[data['purpose_group'] == 'жилье', 'purpose_group'] = 'недвижимость'

data.loc[(data['purpose_group'] == 'коммерческий') | (data['purpose_group'] == 'сдача'), 'purpose_group'] = 'инвестиционная цель'

data['purpose_group'].value_counts()
statistics = data['total_income'].describe()

statistics[4:7]
# функция для определения категории доходов

def determine_income_group(income):

    if income <= statistics[4]: return 1

    elif statistics[4] < income <= statistics[5]: return 2

    elif statistics[5] < income <= statistics[6]: return 3

    else: return 4



# применим функцию к столбцу доходов

data['income_group'] = data['total_income'].apply(determine_income_group)

data.head()
data['age_group'].value_counts()
data['purpose_group'].value_counts()
data['income_group'].value_counts()
# Заменим ошибочные значения в количестве детей из следующего предположения:

# вместо 20 детей укажем 2-х, вместо -1 укажем 1.

data.loc[data['children'] == 20, 'children'] = 2

data.loc[data['children'] == -1, 'children'] = 1



# напишем функцию для категоризации клиентов: 1 — есть дети, 0 — нет детей.

def determine_children(children):

    if children > 0: return 1

    else: return 0



# добавим новый столбец с бинарным признаком в наш исходный массив

data['child_exist'] = data['children'].apply(determine_children)
# построим сводную таблицу

data_pivot = data.pivot_table(index = ['child_exist'], values = 'debt').round(3)

#data_pivot['ratio'] = data_pivot[1] / data_pivot[0]

data_pivot.head()
data_pivot = data.pivot_table(index = ['family_status'], columns = 'debt', values = 'gender', aggfunc = 'count')



# посчитаем вероятность задолженности для каждого вида семейного положения

data_pivot['ratio'] = round(data_pivot[1] / (data_pivot[0] + data_pivot[1]), 3)

data_pivot.sort_values('ratio', ascending = False)
data_pivot = data.pivot_table(index = ['income_group'], columns = 'debt', values = 'gender', aggfunc = 'count')



# посчитаем вероятность задолженности для каждой группы доходов

data_pivot['ratio'] = round(data_pivot[1] / (data_pivot[0] + data_pivot[1]), 3)

data_pivot.sort_values('ratio', ascending = False)
data_pivot = data.pivot_table(index = ['purpose_group'], columns = 'debt', values = 'gender', aggfunc = 'count')



# посчитаем вероятность задолженности для каждой группы доходов

data_pivot['ratio'] = round(data_pivot[1] / (data_pivot[0] + data_pivot[1]), 3)

data_pivot.sort_values('ratio', ascending = False)