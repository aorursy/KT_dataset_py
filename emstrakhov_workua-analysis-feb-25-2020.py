import numpy as np

import pandas as pd
df = pd.read_csv('../input/dou-jobs/df_26-12-2019.csv', index_col=0)

# index_col=0 означает, что первый столбец исходного файла 

# используется как идентификатор строк в датафрейме

df.head()
df.index
df.columns
df[ ['name', 'description', 'company'] ]
df.iloc[:10, -2:]
df.loc[:, 'name':'salary']
df[ (df['salary']>20000) & (df['town']=='Одеса') ]
df.info()
df['date'].value_counts()
# Столбец с датой можно удалить, так как все значения в нём одинаковые

df1 = df.drop('date', axis=1) 

df1.head()
# Описательные статистики для переменной "зарплата"

df1['salary'].describe()
# Популярные города: простейшая инфографика

df['town'].value_counts().head(10).plot(kind='bar');
import seaborn as sns
# Распределение значений з/п: простейшая инфографика

sns.boxplot(df['salary']);
df['salary'].hist();
df1[ (df1['salary'] > 20000) & ((df1['town'] == 'Одеса') | (df1['town'] == 'Одесса')) ]
df1[ df1['company'].isna() ]
def func(s):

    '''

    Функция возвращает первое слово в описании вакансии.

    '''

    return s.split(' ')[0]
func('Майстер манікюру')
df1['type'] = df['name'].map(func)

df1.head()
# То же самое с помощью анонимной функции

df1['type'] = df['name'].map(lambda s: s.split(' ')[0])

df1.head()
df1['type'].value_counts().head(20)
df1.groupby('town')['salary'].mean().sort_values(ascending=False).head(10)
# Вопрос-ДЗ: как исправить ситуацию и что тут не так?
df[ df.company == 'Rakuten' ].head()