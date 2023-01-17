# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np #  для работы с многомерными массивами

import pandas as pd # загрузка, препроцессинг и разведочный анализ данных

import matplotlib.pyplot as plt # предоставляет множество низкоуровневых графических инструментов

import seaborn as sns # содержит больше высокоуровневых графических инструментов

sns.set(rc={'figure.figsize':(10, 4)});

#sns.set()

sns.set_palette("Set3") 
df = pd.read_csv('../input/bank-additional-full.csv', sep=';')
# Метод head(n) предназначен для просмотра первых n строк таблицы

# Признаков довольно много, поэтому для удобства транспонируем вывод

df.head(10).T
df.info() # выведем общую информацию о датасете, узнаем тип каждого признака, и есть ли в данных пропуски
sns.boxplot(df['age']); # в годах
sns.boxplot(df['duration']); # в секундах
# Вычислим средний возраст холостых/незамужних клиентов

#df[(df['marital'] == 'divorced') | (df['marital'] == 'single')]['age'].mean() 

#36.5053152039555
# Вычислим и выведем средний возраст



# холостых клиентов

m_single = df[df['marital'] == 'single']['age'].mean() 

print('Mean age of single clients:', m_single)



# незамужних клиентов

m_divorced = df[df['marital'] == 'divorced']['age'].mean()

print('Mean age of divorced clients:', m_divorced)



# холостых/незамужних клиентов

print('Mean age of single/divorced clients:', (m_single + m_divorced) / 2)
df.groupby('marital')['age'].mean()
df.groupby('marital')['age'].mean().plot(kind='barh', figsize=(10, 5)) 

plt.ylabel('Marital')

plt.xlabel('Age')

plt.show();
df[df['y'] == 'no']['day_of_week'].value_counts().keys()[0] #выводит элемент, у которого max значение
sns.heatmap(pd.crosstab(df['day_of_week'], df['y']), 

            cmap="PuRd", annot=True, cbar=True);
# Построим кросс-таблицу

pd.crosstab(df['y'], df['marital'])
# Построим хитмап

sns.heatmap(pd.crosstab(df['marital'], df['y']), 

            cmap="RdPu", annot=True, cbar=True);
from scipy.stats import chi2_contingency

chi2_contingency(pd.crosstab(df['y'], df['marital']))
pd.crosstab(df['poutcome'], df['default'])
sns.heatmap(pd.crosstab(df['default'], df['poutcome']), 

            cmap="PuBu", annot=True, cbar=True);
from scipy.stats import chi2_contingency, fisher_exact

chi2_contingency(pd.crosstab(df['default'], df['poutcome']))
df.pivot_table(values=['age'], index=['education'], aggfunc='mean') # сводная таблица для наглядности 
df.groupby('education')['age'].mean().plot(kind='barh') 

plt.ylabel('Age')

plt.show();
sns.heatmap(pd.crosstab(df['education'], df['age']), 

            cmap="RdPu", annot=False, cbar=True);
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(df.education)

df['education_le'] = le.transform(df.education)  
#df.head(10).T # посмотрим
from scipy.stats import pearsonr, pointbiserialr

r = pearsonr(df['education_le'], df['age']) # коэффициент корелляции Пирсона

print('Pearson correlation:', r[0], 'p-value:', r[1])
r = pearsonr(df['duration'], df['age']) 

print('Pearson correlation:', r[0], 'p-value:', r[1])
new_values = {'yes':1, 'no':0, 'unknown':-1} # обычный словарь Python

df['dummy_housing'] = df['housing'].map(new_values)

#df.tail(10).T
#pd.crosstab(df['education'], df['dummy_housing'])
sns.heatmap(pd.crosstab(df['education'], df['housing']), 

            cmap="PuRd", annot=True, cbar=True);
#pearsonr(df['dummy_housing'], df['education_le'])
pointbiserialr(df['dummy_housing'], df['education_le'])
for i in [0, 1, 2]:

    print( "Top -",i + 1, df[df['education'] == 'university.degree']['job'].value_counts().keys()[i])
# Построим хитмап

sns.heatmap(pd.crosstab(df['job'], df['marital']), 

            cmap="BuPu", annot=True, cbar=True);
from scipy.stats import chi2_contingency

chi2_contingency(pd.crosstab(df['job'], df['marital']))
df.groupby('education')['duration'].mean().plot(kind='barh') 

plt.ylabel('Education level')

plt.xlabel('Call duration')

plt.show();
r = pearsonr(df['duration'], df['education_le'])

print('Pearson correlation:', r[0], 'p-value:', r[1])
new_values = {'yes':1, 'no':0} # обычный словарь Python

df['dummy_y'] = df['y'].map(new_values)

#df.head(10).T
df.corr(method='spearman')
sns.heatmap(df.corr(method='spearman'));