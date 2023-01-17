# импортируем нужные библиотеки
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
import math
from scipy import stats
from scipy.stats import skew, norm

import matplotlib.pyplot as plt
import matplotlib
palette = plt.get_cmap('Set2')
import seaborn as sns
plt.style.use('seaborn-darkgrid')
# %matplotlib inline
import missingno as msno

from datetime import datetime, timedelta
import os

import warnings
warnings.filterwarnings("ignore")
telecom_df = pd.read_csv('/kaggle/input/telecom-users/telecom_users.csv', index_col=0)
telecom_df.head(5)
print(telecom_df.isnull().sum())

# создадим один объект Figure и два объекта Axes
fig, axes = plt.subplots(1, 2, figsize=(16, 9))
# теперь отрисуем в axes[0] диаграмму пропущенных значений
msno.bar(telecom_df, figsize=(10, 7), ax=axes[0])
# а в axes[1] отрисуем матрицу пропущенных значений
msno.matrix(telecom_df, figsize=(10, 7), ax=axes[1])
plt.show()
print(telecom_df.dtypes)
### метод describe для числовых данных
print(telecom_df.describe(include=[np.number]))
### метод describe для текстовых данных
print(telecom_df.describe(include=[np.object]))

# глянем на уникальные данные каждого столбца
for column in telecom_df.columns:
    print('Уникальные значения столбца {}:\n{}'.format(column, np.sort(telecom_df[column].unique())))
# столбец SeniorCitizen имеет бинарную шкалу, но из нуля и единиц, приведем к однообразному виду
telecom_df['SeniorCitizen'].replace({0:'No', 1:'Yes'}, inplace=True)
# столбец TotalCharges имеет в своем составе такие значения как ' '. Что это? Пропуск или недосчет? Выясним
print(telecom_df.loc[telecom_df['TotalCharges'] == ' ', :])
# все понятно - это вновь прибывшие клиенты. Заменим их значения на 0
telecom_df.loc[telecom_df['TotalCharges'] == ' ', 'TotalCharges'] = 0
# столбец TotalCharges имеет тип данных - object. Непорядок, сменим на float64
telecom_df['TotalCharges'] = telecom_df['TotalCharges'].astype(np.float64)
# доход компании за весь период датасета
print('Доход компании за весь период датасета:', telecom_df['TotalCharges'].sum())

# выведем наиболее часто встречающиеся значения в столбцах
for column in telecom_df.columns:
    print('Column name:', column)
    print(telecom_df[column].value_counts().nlargest(5), '\n')
# оставим только столбцы с данными типа int64 и float64
telecom_df_for_dist = telecom_df.loc[:, telecom_df.dtypes!='object']
# а теперь отобразим все на одном графике
fig, axes = plt.subplots(1, len(telecom_df_for_dist.columns), figsize=(20, 14))
for i, column in enumerate(telecom_df_for_dist.columns):
    sns.distplot(telecom_df_for_dist[column], ax=axes[i])
    axes[i].set_title(column)
    axes[i].axvline(telecom_df_for_dist[column].median(), c='g', label='Медиана')
    axes[i].legend(loc='best')
plt.show()
main_columns_1 = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'InternetService', 'Contract',
'PaperlessBilling', 'PaymentMethod']

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
# пройдемся по каждому основному столбцу
for ax, column in zip(axes.ravel(), main_columns_1):
    # посчитаем распределение по уникальным значениям столбца, сразу же нормализуем их
    data = (telecom_df[column].value_counts() / len(telecom_df)).to_frame().reset_index()
    sns.barplot(y=data.columns[1], x='index', data=data, alpha=0.6, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set(ylabel=None, xlabel=None)
    ax.set_title(f'Распределение по {column}')
    ax.legend()

    # добавим значения столбцов на диаграммы
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{height:.0%}', (x+width/2, y+height*1.02), ha='center')

plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize=(16,9))
telecom_df['Churn'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', shadow=True)
plt.ylabel(None)
plt.title('Распределение клиентов по оттоку')
plt.show()
fig, axes = plt.subplots(3, 3, figsize=(16, 9))
for ax, column in zip(axes.ravel(), main_columns_1):
    # нормализация данных
    data = telecom_df\
    .groupby(['Churn', column])[column]\
    .count()\
    .groupby(column).apply(lambda x: 100 * x / x.sum())\
    .to_frame().stack().reset_index()

    sns.barplot(x=column, y=0, hue='Churn', data=data, ax=ax, alpha=0.6)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set(ylabel=None, xlabel=None)
    ax.set_title(f'Распределение по оттоку по {column}')
    ax.legend(title='Договор рассторжен', loc=2, bbox_to_anchor=(1, 1), fontsize=10)
    # добавим значения столбцов на диаграммы
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{height/100:.0%}', (x+width/2, y+height*1.02), ha='center')
    # отразим на графике прямую, символизирующую пропорцию между глобальным churn/not churn
    ax.axhline(26.5, c='r')
plt.tight_layout()
plt.show()
# выберем колонки с услугами
services_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
# заменим текстовые бинарные данные на числовые
telecom_df_for_services_calculation = telecom_df.copy()
telecom_df_for_services_calculation[services_columns] = telecom_df_for_services_calculation[services_columns]\
.replace({'No':0, 'Yes':1, 'No phone service':0, 'Fiber optic':1, 'DSL':1, 'No internet service':0})
# заведем новую колонку в основном датасете
telecom_df['services'] = telecom_df_for_services_calculation[services_columns].sum(axis=1)
# визуализируем результаты
fig, axes = plt.subplots(figsize=(16,9))
sns.barplot(x=telecom_df['services'].value_counts().reset_index()['index'],
            y=telecom_df['services'].value_counts().reset_index()['services'],
            alpha=0.6)
plt.xlabel('Кол-во подключенных услуг')
plt.ylabel('Кол-во клиентов')
plt.title('Распределение клиентов по кол-ву подключенных услуг')
##plt.close()
plt.show()
fig, axes = plt.subplots(figsize=(16,9))
plt.scatter(telecom_df['services'], telecom_df['MonthlyCharges'], c='r', alpha=0.2)
# определим линию тренда для большей показательности
# создадим для дальнейшего удобства функцию по определнию линии тренда
def plot_trendline(x, y, degree, color, name):
    trend = np.polyfit(x, y, degree)
    trendline = np.poly1d(trend)
    # вычислим коэффициент корреляции
    coef_corr = x.corr(y)
    return plt.plot(x, trendline(x), f'{color}--', label=f'{name} trendline\ncorrelation = {coef_corr:.2}')

plot_trendline(telecom_df['services'], telecom_df['MonthlyCharges'], 1, 'r', 'Services')
plt.xlabel('Кол-во подключенных услуг')
plt.ylabel('Месячные расходы клиента')
plt.title('Зависимость между подключенными опциями и суммой чека')
plt.legend()
plt.show()
# нормализация данных
data = telecom_df\
.groupby(['Churn', 'services'])['services']\
.count()\
.groupby('services').apply(lambda x: 100 * x / x.sum())\
.to_frame().stack().reset_index()

fig, ax = plt.subplots(figsize=(16, 9))
sns.barplot(x='services', y=0, hue='Churn', data=data, alpha=0.6)
ax.set_xlabel('Кол-во подключенных опций')
ax.set_ylabel(None)
ax.set_title(f'Распределение оттока клиентов по кол-ву подключенных опций')
ax.legend(title='Договор рассторжен', loc=2, bbox_to_anchor=(1, 1), fontsize=10)
# добавим значения столбцов на диаграммы
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height/100:.0%}', (x+width/2, y+height*1.02), ha='center')
# отразим на графике прямую, символизирующую пропорцию между глобальным churn/not churn
ax.axhline(26.5, c='r')
plt.tight_layout()
plt.show()
main_columns_2 = ['gender', 'SeniorCitizen', 'Dependents']
fig, axes = plt.subplots(1, 3, figsize=(16, 9))
for ax, column in zip(axes.ravel(), main_columns_2):
    # нормализация данных
    data = telecom_df\
    .groupby([column])[column, 'TotalCharges']\
    .sum()\
    .apply(lambda x: 100 * x / x.sum())\
    .reset_index()

    sns.barplot(x=column, y='TotalCharges', data=data, ax=ax, alpha=0.6)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set(ylabel=None, xlabel=None)
    ax.set_title(f'Распределение дохода по {column}')
    # добавим значения столбцов на диаграммы
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{height/100:.0%}', (x+width/2, y+height*1.02), ha='center')

plt.tight_layout()
plt.show()
# сгруппируем клиентов по месяцам и найдем медиану каждой группы
telecom_df_for_monthlycharges_by_tenure = telecom_df[['tenure', 'MonthlyCharges']]\
                                          .groupby('tenure').median().reset_index()
x = telecom_df_for_monthlycharges_by_tenure['tenure']
y = telecom_df_for_monthlycharges_by_tenure['MonthlyCharges']
# нанесём результат каждого месяца
fig, ax = plt.subplots(figsize=(16, 9))
sns.scatterplot(x=x, y=y)
# определим линию тренда для большей показательности
plot_trendline(x, y, 1, 'b', 'MonthlyCharges')
plt.title('Влияние времени пребывания клиентом компании на размер месячного чека')
plt.xlabel('Кол-во месяцев')
plt.ylabel('Месячная сумма расходов, USD')
plt.legend()
plt.show()
# теперь уже сгруппируем клиентов по месяцам и наличию подключенной услуги телефонной связи, затем найдем медиану каждой группы
telecom_df_for_monthlycharges_by_tenure_phoneservice = telecom_df[['PhoneService', 'tenure', 'MonthlyCharges']]\
                                          .groupby(['PhoneService', 'tenure']).median().reset_index()
x_0 = telecom_df_for_monthlycharges_by_tenure_phoneservice\
[telecom_df_for_monthlycharges_by_tenure_phoneservice['PhoneService'] == 'No']\
['tenure']
y_0 = telecom_df_for_monthlycharges_by_tenure_phoneservice\
[telecom_df_for_monthlycharges_by_tenure_phoneservice['PhoneService'] == 'No']\
['MonthlyCharges']
x_1 = telecom_df_for_monthlycharges_by_tenure_phoneservice\
[telecom_df_for_monthlycharges_by_tenure_phoneservice['PhoneService'] == 'Yes']\
['tenure']
y_1 = telecom_df_for_monthlycharges_by_tenure_phoneservice\
[telecom_df_for_monthlycharges_by_tenure_phoneservice['PhoneService'] == 'Yes']\
['MonthlyCharges']
# нанесём результат каждого месяца
fig, ax = plt.subplots(figsize=(16, 9))
sns.scatterplot(x=x_0, y=y_0)
sns.scatterplot(x=x_1, y=y_1)
# определим линию тренда для большей показательности
plot_trendline(x_0, y_0, 1, 'b', 'No PhoneService')
plot_trendline(x_1, y_1, 1, 'r', 'Yes PhoneService')
##plot_trendline(x, y, 1, 'g', 'Base')
plt.title('Влияние времени пребывания клиентом компании на размер месячного\
чека в зависимости от наличия услуги телефонной связи')
plt.xlabel('Кол-во месяцев')
plt.ylabel('Месячная сумма расходов, USD')
plt.legend()
plt.show()
# теперь уже сгруппируем клиентов по месяцам и наличию подключенной услуги интернет-провайдера, затем найдем медиану каждой
# группы
telecom_df_for_monthlycharges_by_tenure_internetservice =\
telecom_df[['InternetService', 'tenure', 'MonthlyCharges']].groupby(['InternetService', 'tenure']).median().reset_index()
x_0 = telecom_df_for_monthlycharges_by_tenure_internetservice\
[telecom_df_for_monthlycharges_by_tenure_internetservice['InternetService'] == 'No']['tenure']
y_0 = telecom_df_for_monthlycharges_by_tenure_internetservice\
[telecom_df_for_monthlycharges_by_tenure_internetservice['InternetService'] == 'No']['MonthlyCharges']
x_1 = telecom_df_for_monthlycharges_by_tenure_internetservice\
[telecom_df_for_monthlycharges_by_tenure_internetservice['InternetService'] == 'DSL']['tenure']
y_1 = telecom_df_for_monthlycharges_by_tenure_internetservice\
[telecom_df_for_monthlycharges_by_tenure_internetservice['InternetService'] == 'DSL']['MonthlyCharges']
x_2 = telecom_df_for_monthlycharges_by_tenure_internetservice\
[telecom_df_for_monthlycharges_by_tenure_internetservice['InternetService'] == 'Fiber optic']['tenure']
y_2 = telecom_df_for_monthlycharges_by_tenure_internetservice\
[telecom_df_for_monthlycharges_by_tenure_internetservice['InternetService'] == 'Fiber optic']['MonthlyCharges']
# нанесём результат каждого месяца
fig, ax = plt.subplots(figsize=(16, 9))
sns.scatterplot(x=x_0, y=y_0)
sns.scatterplot(x=x_1, y=y_1)
sns.scatterplot(x=x_2, y=y_2)
# определим линию тренда для большей показательности
plot_trendline(x_0, y_0, 1, 'b', 'No InternetService')
plot_trendline(x_1, y_1, 1, 'r', 'DSL InternetService')
plot_trendline(x_2, y_2, 1, 'g', 'Fiber optic InternetService')
plt.title('Влияние времени пребывания клиентом компании на размер месячного чека в зависимости от типа услуги интернет-провайдера')
plt.xlabel('Кол-во месяцев')
plt.ylabel('Месячная сумма расходов, USD')
plt.legend()
plt.show()
# теперь уже сгруппируем клиентов по месяцам и оттоку и найдем медиану каждой группы
telecom_df_for_monthlycharges_by_tenure_churn = telecom_df[['Churn', 'tenure', 'MonthlyCharges']]\
                                          .groupby(['Churn', 'tenure']).median().reset_index()
x_0 =\
telecom_df_for_monthlycharges_by_tenure_churn[telecom_df_for_monthlycharges_by_tenure_churn['Churn'] == 'No']['tenure']
y_0 =\
telecom_df_for_monthlycharges_by_tenure_churn[telecom_df_for_monthlycharges_by_tenure_churn['Churn'] == 'No']['MonthlyCharges']
x_1 =\
telecom_df_for_monthlycharges_by_tenure_churn[telecom_df_for_monthlycharges_by_tenure_churn['Churn'] == 'Yes']['tenure']
y_1 =\
telecom_df_for_monthlycharges_by_tenure_churn[telecom_df_for_monthlycharges_by_tenure_churn['Churn'] == 'Yes']['MonthlyCharges']
# нанесём результат каждого месяца
fig, ax = plt.subplots(figsize=(16, 9))
sns.scatterplot(x=x_0, y=y_0)
sns.scatterplot(x=x_1, y=y_1)
# определим линию тренда для большей показательности
plot_trendline(x_0, y_0, 1, 'b', 'Churn No')
plot_trendline(x_1, y_1, 1, 'r', 'Churn Yes')
plt.title('Влияние времени пребывания клиентом компании на размер месячного чека')
plt.xlabel('Кол-во месяцев')
plt.ylabel('Месячная сумма расходов, USD')
plt.legend()
# plt.close()
plt.show()
# сгруппируем клиентов по месяцам и найдем медиану каждой группы
telecom_df_for_services_by_tenure = telecom_df[['tenure', 'services']]\
                                          .groupby('tenure').median().reset_index()
x = telecom_df_for_services_by_tenure['tenure']
y = telecom_df_for_services_by_tenure['services']
# нанесём результат каждого месяца
fig, ax = plt.subplots(figsize=(16, 9))
sns.scatterplot(x=x, y=y)
# определим линию тренда для большей показательности
plot_trendline(x, y, 1, 'b', 'services')
plt.title('Влияние времени пребывания клиентом компании на кол-во подключенных опций')
plt.xlabel('Кол-во месяцев')
plt.ylabel('Кол-во подключенных опций')
plt.legend()
plt.show()
# сгруппируем клиентов по месяцам и найдем медиану каждой группы
telecom_df_for_services_by_tenure_phoneservice = telecom_df[['PhoneService', 'tenure', 'services']]\
                                          .groupby(['PhoneService', 'tenure']).median().reset_index()
x_0 =\
telecom_df_for_services_by_tenure_phoneservice\
[telecom_df_for_services_by_tenure_phoneservice['PhoneService'] == 'No']\
['tenure']
y_0 =\
telecom_df_for_services_by_tenure_phoneservice\
[telecom_df_for_services_by_tenure_phoneservice['PhoneService'] == 'No']\
['services']
x_1 =\
telecom_df_for_services_by_tenure_phoneservice\
[telecom_df_for_services_by_tenure_phoneservice['PhoneService'] == 'Yes']\
['tenure']
y_1 =\
telecom_df_for_services_by_tenure_phoneservice\
[telecom_df_for_services_by_tenure_phoneservice['PhoneService'] == 'Yes']\
['services']
# нанесём результат каждого месяца
fig, ax = plt.subplots(figsize=(16, 9))
sns.scatterplot(x=x_0, y=y_0)
sns.scatterplot(x=x_1, y=y_1)
# определим линию тренда для большей показательности
plot_trendline(x_0, y_0, 1, 'b', 'PhoneService No')
plot_trendline(x_1, y_1, 1, 'r', 'PhoneService Yes')
plt.title('Влияние времени пребывания клиентом компании на размер месячного чека')
plt.xlabel('Кол-во месяцев')
plt.ylabel('Месячная сумма расходов, USD')
plt.legend()
plt.show()
# сгруппируем клиентов по месяцам и найдем медиану каждой группы
telecom_df_for_services_by_tenure_internetservice = telecom_df[['InternetService', 'tenure', 'services']]\
                                          .groupby(['InternetService', 'tenure']).median().reset_index()
x_0 = telecom_df_for_services_by_tenure_internetservice\
[telecom_df_for_services_by_tenure_internetservice['InternetService'] == 'No']\
['tenure']
y_0 = telecom_df_for_services_by_tenure_internetservice\
[telecom_df_for_services_by_tenure_internetservice['InternetService'] == 'No']\
['services']
x_1 = telecom_df_for_services_by_tenure_internetservice\
[telecom_df_for_services_by_tenure_internetservice['InternetService'] == 'DSL']\
['tenure']
y_1 = telecom_df_for_services_by_tenure_internetservice\
[telecom_df_for_services_by_tenure_internetservice['InternetService'] == 'DSL']\
['services']
x_2 = telecom_df_for_services_by_tenure_internetservice\
[telecom_df_for_services_by_tenure_internetservice['InternetService'] == 'Fiber optic']\
['tenure']
y_2 = telecom_df_for_services_by_tenure_internetservice\
[telecom_df_for_services_by_tenure_internetservice['InternetService'] == 'Fiber optic']\
['services']
# нанесём результат каждого месяца
fig, ax = plt.subplots(figsize=(16, 9))
sns.scatterplot(x=x_0, y=y_0)
sns.scatterplot(x=x_1, y=y_1)
sns.scatterplot(x=x_2, y=y_2)
# определим линию тренда для большей показательности
plot_trendline(x_0, y_0, 1, 'b', 'No InternetService')
plot_trendline(x_1, y_1, 1, 'r', 'DSL InternetService')
plot_trendline(x_2, y_2, 1, 'g', 'Fiber optic InternetService')
plt.title('Влияние времени пребывания клиентом компании на размер месячного чека')
plt.xlabel('Кол-во месяцев')
plt.ylabel('Месячная сумма расходов, USD')
plt.legend()
plt.show()
# теперь уже сгруппируем клиентов по месяцам и оттоку и найдем медиану каждой группы
telecom_df_for_services_by_tenure_churn = telecom_df[['Churn', 'tenure', 'services']]\
                                          .groupby(['Churn', 'tenure']).median().reset_index()
x_0 =\
telecom_df_for_services_by_tenure_churn[telecom_df_for_services_by_tenure_churn['Churn'] == 'No']['tenure']
y_0 =\
telecom_df_for_services_by_tenure_churn[telecom_df_for_services_by_tenure_churn['Churn'] == 'No']['services']
x_1 =\
telecom_df_for_services_by_tenure_churn[telecom_df_for_services_by_tenure_churn['Churn'] == 'Yes']['tenure']
y_1 =\
telecom_df_for_services_by_tenure_churn[telecom_df_for_services_by_tenure_churn['Churn'] == 'Yes']['services']
# нанесём результат каждого месяца
fig, ax = plt.subplots(figsize=(16, 9))
sns.scatterplot(x=x_0, y=y_0)
sns.scatterplot(x=x_1, y=y_1)
# определим линию тренда для большей показательности
plot_trendline(x_0, y_0, 1, 'b', 'Churn No')
plot_trendline(x_1, y_1, 1, 'r', 'Churn Yes')
plt.title('Влияние времени пребывания клиентом компании на размер месячного чека')
plt.xlabel('Кол-во месяцев')
plt.ylabel('Месячная сумма расходов, USD')
plt.legend()
plt.show()
fig, ax = plt.subplots(figsize=(16, 9))
sns.heatmap(telecom_df[['tenure', 'MonthlyCharges', 'TotalCharges', 'services']].corr(), annot=True, cmap='RdYlGn',\
            linewidths=0.2, annot_kws={'size':20})
plt.show()
from sklearn.metrics import matthews_corrcoef


# создадим отдельный датасет
telecom_df_for_corr = telecom_df.copy()
# сначала заменим в колонке InternetService, 'Fiber optic' и 'DSL' на 'Yes'
telecom_df_for_corr['InternetService'].replace({'Fiber optic':'Yes', 'DSL':'Yes'}, inplace=True)
# давайте выберем столбцы для дальнейшего анализа
matthew_corr_columns_1 = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'InternetService', 'PaperlessBilling']
# создадим функцию, которая составляет корреляционный датасет
def matthews_corr(df):
    corr_dict = {column:[] for column in df.columns}
    for column_1 in df.columns:
        for column_2 in df.columns:
            corr_dict[column_1] += [matthews_corrcoef(df[column_1], df[column_2])]
    corr_df = pd.DataFrame(data=corr_dict, index=df.columns, columns=df.columns)
    return corr_df

fig, ax = plt.subplots(figsize=(16, 9))
sns.heatmap(\
matthews_corr(telecom_df_for_corr[matthew_corr_columns_1]),
annot=True, cmap='RdYlGn', linewidths=0.2, annot_kws={'size':20})
plt.show()
telecom_df_for_corr_IS = telecom_df_for_corr[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                              'TechSupport', 'StreamingTV', 'StreamingMovies']]
telecom_df_for_corr_IS = telecom_df_for_corr_IS[telecom_df_for_corr_IS != 'No internet service']
telecom_df_for_corr_IS.dropna(inplace=True)
telecom_df_for_corr_IS
# matthew_corr_columns_2 = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

fig, ax = plt.subplots(figsize=(16, 9))
sns.heatmap(\
matthews_corr(telecom_df_for_corr_IS),
annot=True, cmap='RdYlGn', linewidths=0.2, annot_kws={'size':20})
plt.show()
# взглянем на данные в столбцах
for column in telecom_df.columns:
    print(f'Колонка {column}, {len(telecom_df[column].unique())} уникальных значений')
    print(telecom_df[column].unique())
telecom_df['tenure_years'] = pd.cut(telecom_df['tenure'], 6, labels=range(1, 7))
sns.distplot(telecom_df['MonthlyCharges'])
plt.show()
from sklearn.cluster import KMeans


fig, axes = plt.subplots(1, 2, figsize=(16, 9))
sns.distplot(telecom_df['MonthlyCharges'], ax=axes[0])
km_ = KMeans(n_clusters=2, random_state=0)
km_model = km_.fit(np.array(telecom_df['MonthlyCharges']).reshape(-1, 1))
telecom_df['MonthlyCharges_group'] = km_model.labels_
cluster_centers = km_model.cluster_centers_.ravel()
for i in cluster_centers:
    plt.axvline(i)
for i, (group, cluster_center) in enumerate(zip(telecom_df.groupby('MonthlyCharges_group')['MonthlyCharges'], cluster_centers)):
    sns.distplot(group[1], ax=axes[1], fit=norm)
    axes[1].axvline(cluster_center, c='g')
plt.show()
# сначала пометим второй кластер как третий (ведь их скоро станет три)
telecom_df.loc[telecom_df['MonthlyCharges_group']==1, 'MonthlyCharges_group'] = 2
telecom_df_for_dist_1 = telecom_df[telecom_df['MonthlyCharges_group']==0]
km__ = KMeans(n_clusters=2, random_state=0)
km_model_ = km__.fit(np.array(telecom_df_for_dist_1['MonthlyCharges']).reshape(-1, 1))
telecom_df.loc[telecom_df['MonthlyCharges_group']==0, 'MonthlyCharges_group'] = km_model_.labels_
cluster_centers_ = np.r_[cluster_centers[1], km_model_.cluster_centers_.ravel()]
print(cluster_centers_)
# отобразим результат
fig, axes = plt.subplots(1, 2, figsize=(16, 9))
sns.distplot(telecom_df['MonthlyCharges'], ax=axes[0])
for i in cluster_centers_:
    plt.axvline(i)
for i, (group, cluster_center) in enumerate(zip(telecom_df.groupby('MonthlyCharges_group')['MonthlyCharges'], cluster_centers_)):
    sns.distplot(group[1], ax=axes[1], fit=norm)
    axes[1].axvline(cluster_center, c='g')
plt.show()
# сначала поднимем столбец с медианами
telecom_df_for_monthlycharges_by_tenure_churn.rename(columns={'MonthlyCharges': 'median_by_monthlycharges'}, inplace=True)
# затем посмотрим на сгруппированые значения
telecom_df_for_monthlycharges_by_tenure_churn['values'] = telecom_df[['Churn', 'tenure', 'MonthlyCharges']]\
                                                .groupby(['Churn', 'tenure'])['MonthlyCharges'].apply(list)\
                                                .reset_index()['MonthlyCharges']
# распределим по группам оттока
telecom_df_for_monthlycharges_by_tenure_churn['clusters_centers'] = 0
for churn in ['Yes', 'No']:
    # заведем переменные, куда внесём группы по оттоку
    data = telecom_df_for_monthlycharges_by_tenure_churn[telecom_df_for_monthlycharges_by_tenure_churn['Churn'] == churn]
    x = data['tenure']
    y = data['median_by_monthlycharges']
    # расчитаем тренд каждой группы
    trend_model = np.polyfit(x, y, 1)
    trendline = np.poly1d(trend_model)
    # заведём новый столбец со значенями центров кластеров
    telecom_df_for_monthlycharges_by_tenure_churn\
    .loc[telecom_df_for_monthlycharges_by_tenure_churn['Churn'] == churn, 'clusters_centers'] = trendline(x)
    # отобразим результат для проверки
    sns.scatterplot(x, y)
    plt.plot(x, trendline(x))
plt.show()
# а теперь сгруппируем просто по tenure, таким образом объединим центры двух класетров
telecom_df_for_high_risk_groupby_1 = telecom_df_for_monthlycharges_by_tenure_churn\
                                  .groupby('tenure')['clusters_centers']\
                                  .apply(list)\
                                  .reset_index()
print(telecom_df_for_high_risk_groupby_1)
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score


# первая строчка осталась без второго значения, дадим второе значение 75
telecom_df_for_high_risk_groupby_1.iloc[0, 1].append(75)
# теперь объединим полученный датасет с основным
telecom_df_for_high_risk = telecom_df.merge(telecom_df_for_high_risk_groupby_1)
# пора посмотреть к какому кластеру значения столбца MonthlyCharges каждого объекта ближе
telecom_df_for_high_risk['high_risk_by_monthlycharges'] = telecom_df_for_high_risk[['MonthlyCharges', 'clusters_centers']]\
.apply(lambda x: np.argmin([euclidean(x['MonthlyCharges'], i) for i in x['clusters_centers']]), axis=1)
telecom_df_for_high_risk['Churn'].replace({'Yes':1, 'No':0}, inplace=True)
# посчитаем процент правильных ответов
print('Процент совпадений -',\
      accuracy_score(telecom_df_for_high_risk['Churn'], telecom_df_for_high_risk['high_risk_by_monthlycharges']))
# добавим получившийся столбец к основному датасету
telecom_df = pd.merge(telecom_df, telecom_df_for_high_risk[['customerID', 'high_risk_by_monthlycharges']])
# сначала поднимем столбец с медианами
telecom_df_for_services_by_tenure_churn.rename(columns={'services': 'median_by_services'}, inplace=True)
# затем посмотрим на сгруппированые значения
telecom_df_for_services_by_tenure_churn['values'] = telecom_df[['Churn', 'tenure', 'services']]\
                                                .groupby(['Churn', 'tenure'])['services'].apply(list)\
                                                .reset_index()['services']
# распределим по группам оттока
telecom_df_for_services_by_tenure_churn['clusters_centers'] = 0
for churn in ['Yes', 'No']:
    # заведем переменные, куда внесём группы по оттоку
    data = telecom_df_for_services_by_tenure_churn[telecom_df_for_services_by_tenure_churn['Churn'] == churn]
    x = data['tenure']
    y = data['median_by_services']
    # расчитаем тренд каждой группы
    trend_model = np.polyfit(x, y, 1)
    trendline = np.poly1d(trend_model)
    # заведём новый столбец со значенями центров кластеров
    telecom_df_for_services_by_tenure_churn\
    .loc[telecom_df_for_services_by_tenure_churn['Churn'] == churn, 'clusters_centers'] = trendline(x)
    # отобразим результат для проверки
    sns.scatterplot(x, y)
    plt.plot(x, trendline(x))
plt.show()
# а теперь сгруппируем просто по tenure, таким образом объединим центры двух класетров
telecom_df_for_high_risk_groupby_2 = telecom_df_for_monthlycharges_by_tenure_churn\
                                  .groupby('tenure')['clusters_centers']\
                                  .apply(list)\
                                  .reset_index()
print(telecom_df_for_high_risk_groupby_2)
# первая строчка осталась без второго значения, дадим второе значение 75
telecom_df_for_high_risk_groupby_2.iloc[0, 1].append(75)
# теперь объединим полученный датасет с основным
telecom_df_for_high_risk = telecom_df.merge(telecom_df_for_high_risk_groupby_2)
# пора посмотреть к какому кластеру значения столбца services каждого объекта ближе
telecom_df_for_high_risk['high_risk_by_services'] = telecom_df_for_high_risk[['services', 'clusters_centers']]\
.apply(lambda x: np.argmin([euclidean(x['services'], i) for i in x['clusters_centers']]), axis=1)
telecom_df_for_high_risk['Churn'].replace({'Yes':1, 'No':0}, inplace=True)
# посчитаем процент правильных ответов
print('Процент совпадений -',\
      accuracy_score(telecom_df_for_high_risk['Churn'], telecom_df_for_high_risk['high_risk_by_services']))
# добавим получившийся столбец к основному датасету
telecom_df = pd.merge(telecom_df, telecom_df_for_high_risk[['customerID', 'high_risk_by_services']])
telecom_df
print(len(telecom_df[(telecom_df['PhoneService'] == 'No')\
    & ((telecom_df['MultipleLines'] == 'Yes') | (telecom_df['MultipleLines'] == 'No'))]))
print(len(telecom_df[(telecom_df['PhoneService'] == 'Yes') & (telecom_df['MultipleLines'] == 'No phone service')]))
internetservices = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for column in internetservices:
    print(len(telecom_df[(telecom_df['InternetService'] == 'No')
        & ((telecom_df[column] == 'Yes') | (telecom_df[column] == 'No'))]))
telecom_df_for_ml =\
telecom_df.drop(['customerID', 'tenure', 'TotalCharges', 'MonthlyCharges', 'services'], axis=1)
from sklearn.preprocessing import LabelEncoder


# создадим функцию, которая будет кодировать данные в столбцах
def transform(df):
    transform_df = df.copy()
    le_dict = {column:LabelEncoder() for column in df.columns}
    for column, le_model in le_dict.items():
        le_dict[column] = le_model.fit(df[column])
        transform_df[column] = le_dict[column].transform(df[column])
    return transform_df, le_dict

# а эта функция будет декодировать данные в столбцах на основе передаваемого датасета и словаря енкодеров
def inverse_transform(transform_df, le_dict):
    inverse_transform_df = transform_df.copy()
    for column, le_model in le_dict.items():
        inverse_transform_df[column] = le_dict[column].inverse_transform(transform_df[column])
    return inverse_transform_df, le_dict

df_for_learning = transform(telecom_df_for_ml)[0]
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# разделим датасет на фичи и цели
x = df_for_learning.drop('Churn', axis=1)
y = df_for_learning['Churn']
# а теперь разделим фичи и цели на тренировочные и тестовые
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# также будем применять кросс-валидацию
skf = StratifiedKFold(n_splits=5, random_state=0)
# для некоторых алгоритмов нам будет необходимо иметь признаки и цели одновременно
train, test = train_test_split(df_for_learning, test_size=0.2, random_state=0)
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier

from sklearn.svm import SVC, NuSVC, OneClassSVM, LinearSVC

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, CategoricalNB, ComplementNB

from sklearn.linear_model import LogisticRegressionCV, ElasticNetCV, RidgeClassifierCV

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import recall_score, roc_auc_score
# составим список классификаторов
base_classifiers = [ExtraTreeClassifier, DecisionTreeClassifier, SVC, NuSVC, OneClassSVM, LinearSVC, MLPClassifier,
                    KNeighborsClassifier, NearestCentroid, BernoulliNB, GaussianNB, MultinomialNB, CategoricalNB,
                    ComplementNB, LogisticRegressionCV, RidgeClassifierCV,
                    LinearDiscriminantAnalysis]
# результаты "прогона" базовых классификаторов
classifier_names, recall_scores, roc_auc_scores = [], [], []
for classifier in base_classifiers:
    # "завернём" в оболочку try-except алгоритмы и зададим параметр random_state=0 для репрезентативности результатов
    try:
        estimator = classifier(random_state=0)
    except TypeError:
        estimator = classifier()
    # обучим модель
    model = estimator.fit(x_train, y_train)
    # получим предсказанные значения для тестовых фич
    y_pred = model.predict(x_test)
    # специально для классификатора OneClassSVM, который даёт предсказания в виде -1, 1, создадим условие замены
    if -1 in y_pred:
        y_pred[y_pred == -1] = 0
    # специально для классов ElasticNet и ElasticNetCV, которые дают предсказания в виде вероятности, создадим условие замены
    y_pred = [0 if i <= 0.5 else 1 for i in y_pred]
    # посчитаем метрики и занесём результаты в соответствующие списки
    classifier_names += [classifier.__name__]
    roc_auc_scores += [roc_auc_score(y_test, y_pred)]
    recall_scores += [recall_score(y_test, y_pred)]
# создадим датафрейм с результатами
base_classifiers_df = pd.DataFrame(data=zip(classifier_names, roc_auc_scores, recall_scores),
                                   columns=['classifier', 'roc_auc', 'recall'])
# отсортируем датафрейм по roc-auc метрике в порядке убывания, а также проиндексируем по порядку
base_classifiers_df = base_classifiers_df.sort_values(['roc_auc', 'recall'], ascending=False).reset_index(drop=True)
base_classifiers_df
classifier_names_GS, roc_auc_scores_GS, recall_scores_GS  = [], [], []
%%time

from sklearn.model_selection import GridSearchCV


cnb = CategoricalNB(alpha=0.52)
# cnb = CategoricalNB()
cnb_params = {
# 'alpha':np.arange(0, 1, 0.01),
}
cnb_GS = GridSearchCV(estimator=cnb, param_grid=cnb_params, cv=skf, n_jobs=-1)
cnb_GS_model = cnb_GS.fit(x_train, y_train)
cnb_y_pred = cnb_GS_model.predict(x_test)
classifier_names_GS += ['CategoricalNB']
roc_auc_scores_GS += [roc_auc_score(y_test, cnb_y_pred)]
recall_scores_GS += [recall_score(y_test, cnb_y_pred)]
print('cnb_GS_model.best_params_', cnb_GS_model.best_params_)
print('roc_auc', roc_auc_score(y_test, cnb_y_pred))
print('recall', recall_score(y_test, cnb_y_pred))
%%time

gnb = GaussianNB(var_smoothing=1e-10)
# gnb = GaussianNB()
gnb_params = {
# 'var_smoothing':np.arange(1e-10, 1e-8, 1e-10),
}
gnb_GS = GridSearchCV(estimator=gnb, param_grid=gnb_params, cv=skf, n_jobs=-1)
gnb_GS_model = gnb_GS.fit(x_train, y_train)
gnb_y_pred = gnb_GS_model.predict(x_test)
classifier_names_GS += ['GaussianNB']
roc_auc_scores_GS += [roc_auc_score(y_test, gnb_y_pred)]
recall_scores_GS += [recall_score(y_test, gnb_y_pred)]
print('gnb_GS_model.best_params_', gnb_GS_model.best_params_)
print('roc_auc', roc_auc_score(y_test, gnb_y_pred))
print('recall', recall_score(y_test, gnb_y_pred))
%%time

lda = LinearDiscriminantAnalysis(solver='svd', store_covariance=True, tol=10e-08)
# lda = LinearDiscriminantAnalysis()
lda_params = {
# 'solver':['svd', 'lsqr'],
# 'store_covariance':[True, False],
# 'tol':np.arange(1.0e-8, 2.0e-6, 5.0e-8)
}
lda_GS = GridSearchCV(estimator=lda, param_grid=lda_params, cv=skf, n_jobs=-1)
lda_GS_model = lda_GS.fit(x_train, y_train)
lda_y_pred = lda_GS_model.predict(x_test)
classifier_names_GS += ['LinearDiscriminantAnalysis']
roc_auc_scores_GS += [roc_auc_score(y_test, lda_y_pred)]
recall_scores_GS += [recall_score(y_test, lda_y_pred)]
print('lda_GS_model.best_params_', lda_GS_model.best_params_)
print('roc_auc', roc_auc_score(y_test, lda_y_pred))
print('recall', recall_score(y_test, lda_y_pred))
%%time

lrcv = LogisticRegressionCV(Cs=11, cv=skf, dual=True, multi_class='auto', penalty='l2', solver='liblinear',
    refit=False, random_state=0, n_jobs=-1)
# lrcv = LogisticRegressionCV()
lrcv_params = {
# 'Cs':range(5, 14),
# 'cv':[range(2, 5), skf, None],
# 'dual':[True, False],
# 'penalty':['l1', 'l2', 'elasticnet'],
# 'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
# 'refit':[True, False],
# 'multi_class':['auto', 'ovr', 'multinomial']
}
lrcv_GS = GridSearchCV(estimator=lrcv, param_grid=lrcv_params, cv=skf, n_jobs=-1)
lrcv_GS_model = lrcv_GS.fit(x_train, y_train)
lrcv_y_pred = lrcv_GS_model.predict(x_test)
classifier_names_GS += ['LogisticRegressionCV']
roc_auc_scores_GS += [roc_auc_score(y_test, lrcv_y_pred)]
recall_scores_GS += [recall_score(y_test, lrcv_y_pred)]
print('lrcv_GS_model.best_params_', lrcv_GS_model.best_params_)
print('roc_auc', roc_auc_score(y_test, lrcv_y_pred))
print('recall', recall_score(y_test, lrcv_y_pred))
%%time

rccv = RidgeClassifierCV(cv=skf, fit_intercept=False, normalize=True, store_cv_values=False)
# rccv = RidgeClassifierCV()
rccv_params = {
# 'fit_intercept':[True, False],
# 'normalize':[True, False],
# 'cv':[range(2, 5), skf, None],
# 'class_weight':['balanced', None],
# 'store_cv_values':[True, False],
}
rccv_GS = GridSearchCV(estimator=rccv, param_grid=rccv_params, cv=skf, n_jobs=-1)
rccv_GS_model = rccv_GS.fit(x_train, y_train)
rccv_y_pred = rccv_GS_model.predict(x_test)
classifier_names_GS += ['RidgeClassifierCV']
roc_auc_scores_GS += [roc_auc_score(y_test, rccv_y_pred)]
recall_scores_GS += [recall_score(y_test, rccv_y_pred)]
print('rccv_GS_model.best_params_', rccv_GS_model.best_params_)
print('roc_auc', roc_auc_score(y_test, rccv_y_pred))
print('recall', recall_score(y_test, rccv_y_pred))
%%time

dtc = DecisionTreeClassifier(criterion='entropy', max_depth=6, splitter='best', random_state=0)
# dtc = DecisionTreeClassifier()
dtc_params = {
# 'criterion':["gini", "entropy"],
# 'splitter':["best", "random"],
# 'max_depth':range(2, 7),
}
dtc_GS = GridSearchCV(estimator=dtc, param_grid=dtc_params, cv=skf, n_jobs=-1)
dtc_GS_model = dtc_GS.fit(x_train, y_train)
dtc_y_pred = dtc_GS_model.predict(x_test)
classifier_names_GS += ['DecisionTreeClassifier']
roc_auc_scores_GS += [roc_auc_score(y_test, dtc_y_pred)]
recall_scores_GS += [recall_score(y_test, dtc_y_pred)]
print('dtc_GS_model.best_params_', dtc_GS_model.best_params_)
print('roc_auc_score', roc_auc_score(y_test, dtc_y_pred))
print('recall_score', recall_score(y_test, dtc_y_pred))
%%time

knc = KNeighborsClassifier(algorithm='auto', n_neighbors=86, n_jobs =-1)
# knc = KNeighborsClassifier()
knc_params = {
# 'n_neighbors':range(50, 100),
# 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
}
knc_GS = GridSearchCV(estimator=knc, param_grid=knc_params, cv=skf, n_jobs=-1)
knc_GS_model = knc_GS.fit(x_train, y_train)
knc_y_pred = knc_GS_model.predict(x_test)
classifier_names_GS += ['KNeighborsClassifier']
roc_auc_scores_GS += [roc_auc_score(y_test, knc_y_pred)]
recall_scores_GS += [recall_score(y_test, knc_y_pred)]
print('knc_GS_model.best_params_', knc_GS_model.best_params_)
print('roc_auc_score', roc_auc_score(y_test, knc_y_pred))
print('recall_score', recall_score(y_test, knc_y_pred))
# создадим датафрейм с результатами
base_classifiers_GS_df = pd.DataFrame(data=zip(classifier_names_GS, roc_auc_scores_GS, recall_scores_GS),
                                   columns=['classifier', 'roc_auc', 'recall'])
# отсортируем датафрейм по roc-auc метрике в порядке убывания, а также проиндексируем по порядку
base_classifiers_GS_df = base_classifiers_GS_df.sort_values(['roc_auc', 'recall'], ascending=False).reset_index(drop=True)
base_classifiers_GS_df
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier


bc = BaggingClassifier(base_estimator=dtc, bootstrap_features=True, n_estimators=4,
    max_features=20, oob_score=True, n_jobs=-1, random_state=0)
# bc = BaggingClassifier(n_jobs=-1, random_state=0)
bc_params = {
# 'base_estimator':[lda, lrcv, rccv, dtc],
# 'n_estimators':range(1, 9),
# 'max_features':range(16, 22),
# 'bootstrap_features':[True, False],
# 'oob_score':[True, False],
# 'warm_start':[True, False],
}
bc_GS = GridSearchCV(estimator=bc, param_grid=bc_params, cv=skf, n_jobs=-1)
bc_GS_model = bc_GS.fit(x_train, y_train)
bc_y_pred = bc_GS_model.predict(x_test)
classifier_names_GS += ['BaggingClassifier']
roc_auc_scores_GS += [roc_auc_score(y_test, bc_y_pred)]
recall_scores_GS += [recall_score(y_test, bc_y_pred)]
print('bc_GS_model.best_params_', bc_GS_model.best_params_)
print('roc_auc_score', roc_auc_score(y_test, bc_y_pred))
print('recall_score', recall_score(y_test, bc_y_pred))
rfc = RandomForestClassifier(n_estimators=186,  max_depth=7, oob_score=True, warm_start=True, n_jobs=-1, random_state=0)
# rfc = RandomForestClassifier(n_jobs=-1, random_state=0)
rfc_params = {
# 'n_estimators':range(180, 200, 2),
# 'max_depth':range(6, 8),
# 'criterion':['gini', 'entropy'],
# 'oob_score':[True, False],
# 'warm_start':[True, False]
}
rfc_GS = GridSearchCV(estimator=rfc, param_grid=rfc_params, cv=skf, n_jobs=-1)
rfc_GS_model = rfc_GS.fit(x_train, y_train)
rfc_y_pred = rfc_GS_model.predict(x_test)
classifier_names_GS += ['RandomForestClassifier']
roc_auc_scores_GS += [roc_auc_score(y_test, rfc_y_pred)]
recall_scores_GS += [recall_score(y_test, rfc_y_pred)]
print('rfc_GS_model.best_params_', rfc_GS_model.best_params_)
print('roc_auc_score', roc_auc_score(y_test, rfc_y_pred))
print('recall_score', recall_score(y_test, rfc_y_pred))
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


xgbc = XGBClassifier(
    objective='binary:logistic',
    n_jobs=-1,
    random_state=0,
    learning_rate=0.1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1
)
xgbc_params = {
    'max_depth':range(1,10),
    'min_child_weight':range(8,20)
}
xgbc_GS = GridSearchCV(
    estimator=xgbc,
    param_grid=xgbc_params,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1)
xgbc_GS_model = xgbc_GS.fit(x_train, y_train)
y_pred = xgbc_GS_model.predict(x_test)
print('xgbc_GS_model.best_params_', xgbc_GS_model.best_params_)
print('roc_auc_score', roc_auc_score(y_test, y_pred))
print('recall_score', recall_score(y_test, y_pred))
xgbc = XGBClassifier(
    objective='binary:logistic',
    n_jobs=-1,
    random_state=0,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    max_depth=3,
    min_child_weight=14
)
xgbc_params = {
    'gamma':np.arange(0, 0.1, 0.01),
    'scale_pos_weight':np.arange(0.5, 3, 0.1)
}
xgbc_GS = GridSearchCV(
    estimator=xgbc,
    param_grid=xgbc_params,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1)
xgbc_GS_model = xgbc_GS.fit(x_train, y_train)
y_pred = xgbc_GS_model.predict(x_test)
print('xgbc_GS_model.best_params_', xgbc_GS_model.best_params_)
print('roc_auc_score', roc_auc_score(y_test, y_pred))
print('recall_score', recall_score(y_test, y_pred))
xgbc = XGBClassifier(
    objective='binary:logistic',
    n_jobs=-1,
    random_state=0,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=14,
    gamma=0.01,
    scale_pos_weight=2.4,

)
xgbc_params = {
    'subsample':np.arange(0.1, 1, 0.05),
    'colsample_bytree':np.arange(0.1, 1, 0.05),
}
xgbc_GS = GridSearchCV(
    estimator=xgbc,
    param_grid=xgbc_params,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1)
xgbc_GS_model = xgbc_GS.fit(x_train, y_train)
xgbc_y_pred = xgbc_GS_model.predict(x_test)
print('xgbc_GS_model.best_params_', xgbc_GS_model.best_params_)
print('roc_auc_score', roc_auc_score(y_test, xgbc_y_pred))
print('recall_score', recall_score(y_test, xgbc_y_pred))
xgbc = XGBClassifier(
    objective='binary:logistic',
    n_jobs=-1,
    random_state=0,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=14,
    gamma=0.01,
    scale_pos_weight=2.4,
    subsample=0.8,
    colsample_bytree=0.8)

xgbc_model = xgbc.fit(x_train, y_train)
xgbc_y_pred = xgbc_model.predict(x_test)
classifier_names_GS += ['XGBClassifier']
roc_auc_scores_GS += [roc_auc_score(y_test, xgbc_y_pred)]
recall_scores_GS += [recall_score(y_test, xgbc_y_pred)]
print('roc_auc_score', roc_auc_score(y_test, xgbc_y_pred))
print('recall_score', recall_score(y_test, xgbc_y_pred))
xgb.plot_importance(xgbc_model)
# создадим список наших обученных алгоритмов
estimators = [cnb, gnb, knc, lda, rccv, lrcv, dtc]
# функция получения матриц метапризнаков
def meta_matrix(estimators, x_train, x_test, y_train, cv=5):
    from sklearn.model_selection import cross_val_predict

    
    meta_mtrx_train = np.empty((len(x_train), len(estimators)))
    meta_mtrx_test = np.empty((len(x_test), len(estimators)))
    for n, estimator in enumerate(estimators):
        meta_mtrx_train[:, n] = cross_val_predict(estimator, x_train, y_train, cv=cv, method='predict')
        meta_mtrx_test[:, n] = estimator.fit(x_train, y_train).predict(x_test)
    return meta_mtrx_train, meta_mtrx_test

# две матрицы метапризнаков (тренировочная и тестовая)
meta_mtrx_train, meta_mtrx_test = meta_matrix(estimators, x_train, x_test, y_train)
xgb_stacking = XGBClassifier(
    objective='binary:logistic',
    n_jobs=-1,
    random_state=0,
    learning_rate=0.1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1
)
xgb_stacking_params = {
    'max_depth':range(2, 10),
    'min_child_weight':range(2, 10)
}
xgb_stacking_GS = GridSearchCV(
    estimator=xgb_stacking,
    param_grid=xgb_stacking_params,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1)
xgb_stacking_GS_model = xgb_stacking_GS.fit(meta_mtrx_train, y_train)
stacking_y_pred = xgb_stacking_GS_model.predict(meta_mtrx_test)
print('xgb_stacking_GS_model.best_params_', xgb_stacking_GS_model.best_params_)
print('roc_auc_score', roc_auc_score(y_test, stacking_y_pred))
print('recall_score', recall_score(y_test, stacking_y_pred))
xgb_stacking = XGBClassifier(
    objective='binary:logistic',
    n_jobs=-1,
    random_state=0,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    max_depth=2,
    min_child_weight=8
)
xgb_stacking_params = {
    'gamma':np.arange(0, 4, 0.1),
    'scale_pos_weight':np.arange(0, 4, 0.1)
}
xgb_stacking_GS = GridSearchCV(
    estimator=xgb_stacking,
    param_grid=xgb_stacking_params,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1)
xgb_stacking_GS_model = xgb_stacking_GS.fit(meta_mtrx_train, y_train)
stacking_y_pred = xgb_stacking_GS_model.predict(meta_mtrx_test)
print('xgb_stacking_GS_model.best_params_', xgb_stacking_GS_model.best_params_)
print('roc_auc_score', roc_auc_score(y_test, stacking_y_pred))
print('recall_score', recall_score(y_test, stacking_y_pred))
xgb_stacking = XGBClassifier(
    objective='binary:logistic',
    n_jobs=-1,
    random_state=0,
    learning_rate=0.1,
    max_depth=2,
    min_child_weight=8,
    scale_pos_weight=2.7,
    gamma=3
)
xgb_stacking_params = {
    'subsample':np.arange(0.1, 1, 0.05),
    'colsample_bytree':np.arange(0.1, 1, 0.05),
}
xgb_stacking_GS = GridSearchCV(
    estimator=xgb_stacking,
    param_grid=xgb_stacking_params,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1)
xgb_stacking_GS_model = xgb_stacking_GS.fit(meta_mtrx_train, y_train)
stacking_y_pred = xgb_stacking_GS_model.predict(meta_mtrx_test)
print('xgb_GS_model.best_params_', xgb_stacking_GS_model.best_params_)
print('roc_auc_score', roc_auc_score(y_test, stacking_y_pred))
print('recall_score', recall_score(y_test, stacking_y_pred))
xgb_stacking = XGBClassifier(
    objective='binary:logistic',
    n_jobs=-1,
    random_state=0,
    learning_rate=0.1,
    max_depth=2,
    min_child_weight=8,
    scale_pos_weight=2.7,
    gamma=3,
    colsample_bytree=0.3,
    subsample=0.35
    )

xgb_stacking_model = xgb_stacking.fit(meta_mtrx_train, y_train)
stacking_y_pred = xgb_stacking_model.predict(meta_mtrx_test)
classifier_names_GS += ['Stacking']
roc_auc_scores_GS += [roc_auc_score(y_test, stacking_y_pred)]
recall_scores_GS += [recall_score(y_test, stacking_y_pred)]
print('roc_auc_score', roc_auc_score(y_test, stacking_y_pred))
print('recall_score', recall_score(y_test, stacking_y_pred))
xgb.plot_importance(xgb_stacking_model)
# создадим датафрейм с результатами
base_classifiers_GS_df = pd.DataFrame(data=zip(classifier_names_GS, roc_auc_scores_GS, recall_scores_GS),
                                   columns=['classifier', 'roc_auc', 'recall'])
# отсортируем датафрейм по roc-auc метрике в порядке убывания, а также проиндексируем по порядку
base_classifiers_GS_df = base_classifiers_GS_df.sort_values(['roc_auc', 'recall'], ascending=False).reset_index(drop=True)
base_classifiers_GS_df
from sklearn.metrics import plot_confusion_matrix


plot_confusion_matrix(xgbc_model, x_test, y_test)
plt.show()
plot_confusion_matrix(xgb_stacking_model, meta_mtrx_test, y_test)
plt.show()
plot_confusion_matrix(cnb_GS_model, x_test, y_test)
plt.show()
from sklearn.metrics import plot_roc_curve

fig, ax = plt.subplots(figsize=(16, 9))
plot_roc_curve(xgbc_model, x_test, y_test, ax=ax, name='XGBClassifier')
plot_roc_curve(xgb_stacking_model, meta_mtrx_test, y_test, color='g', ax=ax, name='Stacking')
plot_roc_curve(cnb_GS_model, x_test, y_test, color='r', ax=ax, name='CategoricalNB')
plt.plot([0, 1], [0, 1], 'k--')
plt.show()
cnb = CategoricalNB(alpha=0.52)
cnb_model = cnb.fit(x_train, y_train)
cnb_y_pred = cnb_model.predict(x_test)
print('roc_auc', roc_auc_score(y_test, cnb_y_pred))
print('recall', recall_score(y_test, cnb_y_pred))