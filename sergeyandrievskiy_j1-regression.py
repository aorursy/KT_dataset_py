import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pylab as plt

from pylab import rcParams

rcParams['figure.figsize'] = 20, 10
df = pd.read_csv(r'../input/j-data-regression-for-task/j_data_regression_for_task.csv')
df.head(5)
df.describe()
# В датасете есть строки с пропущенными значениями

# Выведем строки с такими значениями



df[df['CONTRAGENT'].isnull() | df['ARTICLE_GROUP'].isnull()]



# Известно, что пропущенные значения есть только в столбцах CONTRAGENT и ARTICLE_GROUP

# Поскольку всего строк в датасете больше 128к, а битых строк 25, то проще их удалить
# Удаляем строки с пропущенными значениями



df = df.dropna();
from sklearn.preprocessing import OneHotEncoder
# Поскольку данных за 2017 год преимущественно больше, чем за 2018, то будем обучать модель на данных за 2017 год, а тестировать на данных за 2018
# Создаем дополнительное поле Период вида"ГодМесяц"



df['MONTH'] = df['MONTH'].apply(str)

df['YEAR'] = df['YEAR'].apply(str)



df['MONTH'] = df['MONTH'].apply(lambda x: '0' + x if len(x) == 1 else x)



df['PERIOD'] = df['YEAR'] + df['MONTH']
# Группируем данные до разреза ARTICLE_GROUP-PERIOD



#gr_data_period = pd.crosstab(df['ARTICLE_GROUP'], df['PERIOD'], aggfunc="sum", margins=True)

#gr_data_period



gr_data_period = pd.pivot_table(df, values='SALES', index=['PERIOD'], columns='ARTICLE_GROUP', aggfunc=np.sum)

gr_data_period
# В марте 2017 года данные о продажах либо нулевые, либо отсутствуют, поэтому можно удалить их



df = df[df.MONTH != '04']
gr_data_period = pd.pivot_table(df, values='SALES', index=['PERIOD'], columns='ARTICLE_GROUP', aggfunc=np.sum)

gr_data_period.plot();