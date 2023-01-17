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

a = (df['age'] / 365).round()

df['age'] = a
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
a = df[df['gender'] == 1]['height'].mean()

b =  df[df['gender'] == 2]['height'].mean()

print( a, b)

print('Количество женщин:', df[df['gender'] == 1]['height'].size ,'Количество мужчин:', df[df['gender'] == 2]['height'].size)
a = df[(df['gender'] == 1) & (df['alco'] == 1)].shape[0]

b = df[(df['gender'] == 2) & (df['alco'] == 1)].shape[0]

print(a<b, a, b)

a = df[(df['gender'] == 1) & (df['smoke'] == 1)].shape[0]

b = df[(df['gender'] == 2) & (df['smoke'] == 1)].shape[0]

w = df[df['gender'] == 1]

m = df[df['gender'] == 2]

ap = (a  / w.shape[0])*100

bp = (b  / m.shape[0])*100

print(ap,bp)

s1 = df[df['smoke'] == 0]['age'].mean()

s2 =  df[df['smoke'] == 1]['age'].mean()

print(s1, s2,  abs(s1-s2))
x = df.groupby('smoke')['age'].mean()

type(x)
np.abs(x[0]-x[1])
h = df['height']/100

m = df['weight']/(h**2)

df['BMI'] = m

df.head()
BMI_ =  df['BMI']

mean_BMI = BMI_.mean()

print(18.5<mean_BMI<25 , mean_BMI)
a_w = df[df['gender'] == 1]['BMI'].mean()

b_m =  df[df['gender'] == 2]['BMI'].mean()

print(a_w > b_m , a_w, b_m)
a_h = df[df['cardio'] == 0]['BMI'].mean()

b_n =  df[df['cardio'] == 1]['BMI'].mean()

print(a_h>b_n , a_h, b_n)
x=df.pivot_table(values=['BMI'], index=['gender','cardio' ,'alco'], aggfunc='mean')
x
x.index
#6

a = df.drop(df[df['ap_lo'] > df['ap_hi']].index)

a.shape[0] / df.shape[0]



#7

x=df.pivot_table(values=['smoke'], index=['age','ap_hi' ,'cholesterol'], aggfunc='mean')

x

#Визуализируйте распределение уровня холестерина для различных возрастных категорий.

df.groupby('cholesterol')['age'].mean().plot(kind='bar') 

plt.ylabel('Age') 

plt.show();
#9

fig, ax = plt.subplots(figsize=(20, 10))

sns.boxplot(df['BMI']);

#отдельные точки на графике соответствуют выбросам --- нетипичным для данной выборки значениям. 

#Как видим, их оказалось довольно много.
#10

fig, ax = plt.subplots(figsize=(20, 10))

df.groupby('cardio')['BMI'].mean().plot(kind='bar') 

plt.ylabel('BMI') 

plt.show();

#BMI и cardio соотносятся следующим образом: у людей с ССЗ ИМТ выше, чем у тех, которых нет ССЗ. 