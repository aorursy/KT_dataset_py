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

#получается, что "1" - женщины, "2" - мужчины
df.head()
#для удобства, перекадируем признак gender:

new_values = {1:'woman', 2:'man'}

df['new_gender'] = df['gender'].map(new_values)

df.head(10)
df['new_gender'].value_counts()
#2

pd.crosstab(df['alco'], df['new_gender'])

#да, мужчины более склонны к употреблению алкоголя
df[df['new_gender'] == 'man']['alco'].value_counts(normalize=True)
df[df['new_gender'] == 'woman']['alco'].value_counts(normalize=True)
#3

df[df['smoke'] == 1]['new_gender'].value_counts(normalize=True)

#процент курящих мужчин выше, чем курящих женщин
#4

df.pivot_table(values=['age'], index=['smoke'], aggfunc='mean')
#5

BMI = df['weight']/(df['height_cm']*df['height_cm'])

BMI[:10]
#5.a

BMI.mean()
B = BMI.mean()

if B <= 18.5 or B >= 25:

    print('False')

#не верно
df['BMI'] = BMI

df.head(10)
#5.b

df.groupby('new_gender')['BMI'].mean()

#верно
#5.с

df.groupby('cardio')['BMI'].mean()

#не верно
#5.d

# 18.5 <= BMI <= 25

df.pivot_table(values=['BMI'], index=['cardio', 'alco', 'new_gender'], aggfunc='mean')

#верно
#6

dummy_df = df.drop(df[(df['ap_lo'] > df['ap_hi'])].index)

dummy_df.shape[0] / df.shape[0]
#7

df[(df['smoke']==1)&(df['new_gender']=='man')&(df['age']>=60)&(df['age']<=65)&(df['ap_hi']<=120)&(df['cholesterol']==1)]['cardio'].value_counts()
df[(df['smoke']==1)&(df['new_gender']=='man')&(df['age']>=60)&(df['age']<=65)&(df['ap_hi']>=160)&(df['ap_hi']<=180)&(df['cholesterol']==3)]['cardio'].value_counts()
#8

plt.figure(figsize=(15, 8)) 

sns.countplot(y='age', hue='cholesterol', data=df);

#9

plt.figure(figsize=(15, 8)) 

sns.boxplot(df['BMI']);
#10

df.groupby('cardio')['BMI'].mean().plot(kind='bar')

plt.ylabel('BMI')

plt.show();

#ССЗ повышает ИМТ