import numpy as np
import pandas as pd
df = pd.read_csv('../input/dou-jobs/df_26-12-2019.csv', index_col=0)
# index_col=0 означает, что первый столбец исходного файла 
# используется как идентификатор строк в датафрейме
df.head()
df.index
df.columns
df.tail(5)
df.index
df.columns
df[['name','description','company']]
df.iloc[:10,-2:]
df.loc[:10,'name':'salary']
df.info()
df['date'].value_counts()
df['salary'].value_counts()
df1=df.drop('date',axis=1)
df.head()
df1['salary'].describe()
df['town'].value_counts().head(10).plot(kind='bar')
import seaborn as sns
sns.boxplot(df['salary'])
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
df1[ (df1['company'] == 'Sky Union Holdings') ]