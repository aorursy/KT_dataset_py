# The binary classification goal is to predict if the client will subscribe a bank term deposit (variable y).

# Целью бинарной класификации является предсказать возьмёт ли клиент банка срочный депозит (Целевая переменная y).



import os

print(os.listdir("../input"))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)})

from scipy.stats import chi2_contingency, fisher_exact

from scipy.stats import pearsonr, spearmanr, kendalltau
df = pd.read_csv("../input/bank-additional-full.csv", sep = ';')
df.info()

# Информация о типах данных, важна при их обработке
# Оghtltkbv выбросы

sns.boxplot(df['age'])

# Все точки находящиеся вне границ являются выбросами
def outliers_indices(feature):

    '''

    Будем считать выбросами все точки, выходящие за пределы трёх сигм.

    '''

    mid = df[feature].mean()

    sigma = df[feature].std()

    return df[(df[feature] < mid - 3*sigma) | (df[feature] > mid + 3*sigma)].index
wrong_age = outliers_indices("age")

out = set(wrong_age)

print(len(out))

# Из набора данных будет удалено 369 объектов-выбросов, что не является существенным в данном случае.
df.drop(out, inplace=True)
df[(df["marital"] == "single")]["age"].mean() 

# Выбрали всех незамужних клиентов, а затем по полю "age" нашли среднее значение.
df[(df["y"] == "no")]["day_of_week"].value_counts().idxmax() 

# Выбрали всех клиентов, что отказались от депозита и за тем расчитали по полю "day_of_week"

# количество клиентов по каждому дню и нашли день с максимальным значением.
sns.countplot(y='marital', hue='y', data=df)

#Наилучший показатель взятых кредитов у замужних клиентов, за ним идут одинокие, а после разведённые и те о которых информации нет.
print("p-value: ", chi2_contingency(pd.crosstab(df['default'], df['y']))[1])

# Малое значение p-value говорит о том что связь статистически подтверждается (p-value < 0,5).
sns.heatmap(pd.crosstab(df['default'], df['y']))
df.groupby('education')['age'].mean().plot(kind='bar')
print("p-value: ", chi2_contingency(pd.crosstab(df['education'], df['age']))[1])

# Малое значение p-value говорит о том что связь статистически подтверждается (p-value < 0,5).
rezult = pearsonr(df['duration'], df['age'])

print('Pearson correlation:', rezult[0], 'p-value:', rezult[1])

# p-value > 0.5 а значит вероятность того что, есть взаимосвязь между этими параметрами статистически незначима
print("p-value: ", chi2_contingency(pd.crosstab(df['education'], df['housing']))[1])

# Малое значение p-value говорит о том что связь статистически подтверждается (p-value < 0,5).
print("job and age:", "p-value: ", chi2_contingency(pd.crosstab(df['job'], df['age']))[1])

# Малое значение p-value говорит о том что связь статистически подтверждается (p-value < 0,5).

print("job and y:", "p-value: ", chi2_contingency(pd.crosstab(df['job'], df['y']))[1])

# Малое значение p-value говорит о том что связь статистически подтверждается (p-value < 0,5).

print("age and y:", "p-value: ", chi2_contingency(pd.crosstab(df['age'], df['y']))[1])

# Малое значение p-value говорит о том что связь статистически подтверждается (p-value < 0,5).

print("marital and y:", "p-value: ", chi2_contingency(pd.crosstab(df['marital'], df['y']))[1])

# Малое значение p-value говорит о том что связь статистически подтверждается (p-value < 0,5).

print("education and y:", "p-value: ", chi2_contingency(pd.crosstab(df['education'], df['y']))[1])

# Малое значение p-value говорит о том что связь статистически подтверждается (p-value < 0,5).
#Вывод:

#Мы почистили данные от выбрасов и с большой долей вероятности выявили статистическую связь между такими фичами:

#Возрастом, работой, обраованием и жильём.

#А так же выявили факторы наиболее влияющие на результат:

#Семейное положение, наличе кредита, работа, возраст и образование.

#Статистически выяснили что связь между временем разговора и возрастом незначима.