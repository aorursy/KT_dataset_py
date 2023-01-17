# Импорт нужных библиотек

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(); # более красивый внешний вид графиков по умолчанию
df = pd.read_csv('../input/cardio_train.csv', sep=';')
# Метод head(n) предназначен для просмотра первых n строк таблицы (по умолчанию n=5)

# Аналогично метод tail(n) возвращает последние n строк

df.head()
# Если признаков (столбцов) слишком много, полезно будет транспонировать вывод

df.head(10).T
# Метод info() позволяет вывести общую информацию о датасете

# Мы можем узнать тип каждого признака, а также есть ли в данных пропуски

df.info()
# Метод describe() позволяет собрать некоторую статистику по каждому числовому признаку

# Для более удобного прочтения полученную таблицу можно транспонировать

df.describe().T
df['cardio'].value_counts()
# Параметр normalize позволяет узнать процентное соотношение

df['cardio'].value_counts(normalize=True)
df['height'].hist();
df['height'].hist(bins=20);
sns.boxplot(df['height']);
# Обратите внимание - мы применяем здесь метод, а не функцию round. Это значительно ускоряет вычисления

# Операция "деления столбца на число" работает интуитивно понятно - 

# каждый элемент делится на это число. Магия NumPy в действии!

df['age'] = (df['age'] / 365).round()
# Синтаксис предельно прост, лаконичен и интуитивно понятен

df.groupby('cardio')['age'].mean()
df.groupby('cardio')['age'].mean().plot(kind='bar') 

plt.ylabel('Age') # добавляем подпись на оси Оу

plt.show();
plt.figure(figsize=(15, 8)) # увеличим размер картинки

sns.countplot(y='age', hue='cardio', data=df);
plt.scatter(df['age'], df['height']);
sns.jointplot(x='height', y='weight', data=df);
# values - признаки, по которым вычисляются значения функции aggfunc

# index - признаки, по которым выполняется группировка

df.pivot_table(values=['age', 'cardio'], index=['smoke', 'alco'], aggfunc='mean')
pd.crosstab(df['smoke'], df['alco'])
h = df['height'] # сохраним всю колонку "рост" в отдельную переменную для экспериментов

type(h) # посмотрим тип 
first_patient = df[0]
first_patient = df.iloc[0]

print(first_patient)
print(df.loc[0, 'age'])
h_meters = h / 100 # предельно просто!

h_meters[:10] # в отдельных столбцах уже можно применять "обычные" срезы, как в списках
%%timeit

lilliputs = 0

for value in h:

    if value < 125:

        lilliputs = lilliputs + 1
%%timeit

h[h < 125].shape[0]
# Вычислим средний возраст людей, склонных к курению

df[df['smoke'] == 1]['age'].mean()
# Условие может быть составным

df[(df['smoke'] == 1) & (df['cardio'] == 1)]['age'].mean()
# Удалим целевой признак cardio

dummy_df = df.drop('cardio', axis=1)

dummy_df.head()
# Удалим первые 100 пациентов

dummy_df = df.drop(np.arange(100), axis=0)

dummy_df.head()
# Удалим всех пацентов с ростом ниже 125 см, а также выше 200 см

dummy_df = df.drop(df[(df['height'] < 125) | (df['height'] > 200)].index)

dummy_df.shape[0] / df.shape[0]
df['height_cm'] = df['height'] / 100

df.head()
new_values = {1:'low', 2:'normal', 3:'high'} # обычный словарь Python

df['dummy_cholesterol'] = df['cholesterol'].map(new_values)

df.head()
df['cardio'] = df['cardio'].astype(bool)

df.head()
df.groupby('gender')['height'].mean()
df.pivot_table(values=['alco'], index=[ 'gender'], aggfunc='sum')
df.groupby('gender')['smoke'].value_counts(normalize=True)
df[df['smoke'] == 1]['age'].mean()-df[df['smoke'] == 0]['age'].mean()
h = df['height'] # сохраним всю колонку "рост" в отдельную переменную для экспериментов

type(h) # посмотрим тип 

h_meters = h / 100 # предельно просто!

w = df['weight']

q=(w/(h_meters*h_meters))

df['bmi'] = q

print(df['bmi'].mean())#Средний ИМТ находится в диапазоне нормальных значений ИМТ.

print(df[df['gender'] ==2 ]['bmi'].mean())#ИМТ мужчины

print(df[df['gender'] ==1]['bmi'].mean())#ИМТ женщины

print(df[(df['cardio'] == 0) ]['bmi'].mean())#здоровые

print(df[(df['cardio'] == 1) ]['bmi'].mean())#не здоровые

print(df[(df['gender'] == 1 )&(df['cardio'] == 0)&(df['alco'] == 0)]['bmi'].mean())#женщины

print(df[(df['gender'] == 2)&(df['cardio'] == 0)&(df['alco'] == 0)]['bmi'].mean())#мужчины
dummy_df = df.drop(df[(df['ap_hi'] > df['ap_lo'])].index)

dummy_df.shape[0] / df.shape[0]
plt.figure(figsize=(15, 8)) # увеличим размер картинки

sns.countplot(y='age', hue='cholesterol', data=df);
df['bmi'].hist(bins=60);
sns.boxplot(df['bmi']);
df.groupby('cardio')['bmi'].mean().plot(kind='bar') 

plt.ylabel('bmi') 

plt.show();
df.pivot_table(values=['bmi'], index=['age','cardio'], aggfunc='mean')