# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set();

df = pd.read_csv('../input/cardio_train.csv', sep=';')

df.head()
# 1. Определите количество мужчин и женщин среди испытуемых. Обратите внимание, что способ кодирования переменной gender мы не знаем. 

# Воспользуемся медицинским фактом, а именно: мужчины в среднем выше женщин.

new_df = pd.DataFrame()

new_df['mean_height'] = df.groupby('gender')['height'].mean()

new_df['count'] = df['gender'].value_counts()

new_df

# женщин - 45530

# мужчин - 24470
# 2. Верно ли, что мужчины более склонны к употреблению алкоголя, чем женщины?

pd.crosstab(df['gender'], df['alco'])

# Да, мужчины более склонны к употреблению алкоголя, чем женщины. При меньшем кол-ве мужчин число употребляющих алкоголь - больше.
# 3. Каково различие между процентами курящих мужчин и женщин?

pd.crosstab(df['gender'], df['smoke'], normalize=True)
# Процент курящих женщин

df[df['gender'] == 1]['smoke'].sum() / df[df['gender'] == 1].shape[0]
# Процент курящих мужчин

df[df['gender'] == 2]['smoke'].sum() / df[df['gender'] == 1].shape[0]

# Курящие мужчины составляют 7.6% от всех людей и 11.7% от мужчин. Женщины же всего 1.1% от всех людей и 1.7% от женщин. . 
# 4. Какова разница между средними значениями возраста для курящих и некурящих?

df['age'] = (df['age'] / 365).round()

df.groupby('smoke')['age'].mean()

# Курящие люди в срднем на 1 год младше, чем некурящие
# 5. Создайте новый признак --- BMI (body mass index, индекс массы тела). Для этого разделите вес в килограммах на квадрат роста в метрах. 

# Считается, что нормальные значения ИМТ составляют от 18.5 до 25. Выберите верные утверждения:



#(a) Средний ИМТ находится в диапазоне нормальных значений ИМТ.

#(b) ИМТ для женщин в среднем выше, чем для мужчин.

#(c) У здоровых людей в среднем более высокий ИМТ, чем у людей с ССЗ.

#(d) Для здоровых непьющих мужчин ИМТ ближе к норме, чем для здоровых непьющих женщин



df['BMI'] = df['weight'] / ((df['height']/100)**2)

df['BMI'].mean()

# Средний BMI - 27.5 т.е. больше нормы
df.groupby('gender')['BMI'].mean()

# В среднем BMI для женщин выше, чем для мужчин на 1.2
df.groupby('cardio')['BMI'].mean()

# В среднем у людей с ССЗ BMI больше на 2, чем у здоровых.
df.pivot_table(values=['BMI'], index=['cardio', 'alco', 'gender'], aggfunc='mean')

# BMI ближе к норме у здоровых непьющих мужчин. ,

# Ответ:b,d
# 6. Удалите пациентов, у которых диастолическое давление выше систолического. Какой процент от общего количества пациентов они составляли?

df[df['ap_hi'] < df['ap_lo']].shape[0] / df.shape[0]

# Пациенты у которых диастолическое давление выше систолического - 1.7 % от общего кол-ва пациентов
df.drop(df[df['ap_hi'] < df['ap_lo']].index)
# 7. На сайте Европейского общества кардиологов представлена шкала SCORE. Она используется для расчёта риска смерти от сердечно-сосудистых заболеваний в ближайшие 10 лет.

# Вычислите аналогичное соотношение для наших данных.

platform_genre_sales = df[(df['ap_hi'] > 120) & (df['ap_hi'] < 180) & (df['age'] > 60) & (df['age'] < 65) & (df['gender'] == 2)].pivot_table(

                        index='ap_hi', 

                        columns='cholesterol', 

                        values='cardio', 

                        aggfunc=sum).fillna(0).applymap(float)

f, ax = plt.subplots(figsize=(10, 17))

sns.heatmap(platform_genre_sales, annot=True, linewidths=.5, fmt=".1f", ax=ax, yticklabels=5)

# Двннвя зависимость не сохранилась на наших данных.
# 8. Визуализируйте распределение уровня холестерина для различных возрастных категорий.

plt.figure(figsize=(25, 15))

sns.countplot(y='age', hue='cholesterol', data=df);
# 9. Как распределена переменная BMI? Есть ли выбросы

sns.boxplot(df['BMI']);

# Достаточно много выбросов нетипичных для данной выборки значений
# 10. Как соотносятся ИМТ и наличие ССЗ? Придумайте подходящую визуализацию.

df.groupby(['cardio', 'gender'])['BMI'].mean().plot(kind='bar') 

plt.ylabel('BMI') 

plt.show();

# В среднем средний BMI ближе к норме у людей без ССЗ.