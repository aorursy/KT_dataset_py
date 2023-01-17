texto = 'machine learning python IESB'
texto.lower()
texto.upper()
texto
texto[:4]
texto[10:14]
texto[::-1]
texto.split()
texto_dividido = texto.split()
texto_dividido
texto_dividido[0]
texto_dividido[-1]
' | '.join(texto_dividido)
texto.replace('machine', 'deep')
len(texto)
lista = [1, 3, 5, 7]
lista_quadrado = []

for x in lista:

    lista_quadrado.append(x ** 2)
lista_quadrado
lista_quadrado[:2]
lista_quadrado[-2:]
lista_quadrado[::-1]
len(lista_quadrado)
lista_quadrado = [numero ** 2 for numero in lista]
lista_quadrado
def eleva_quadrado(lista):

    return [numero ** 2 for numero in lista]
eleva_quadrado(lista)
0 == False
1 == True
lista[False]
lista[True]
lista = []

lista_par = []

for i in range(8):

    if i % 2:

        lista.append(i)

    else:

        lista_par.append(i)
lista
lista_par
a = [1,2,3,4,5]
a[::-1]
alfabeto = {'A': 1, 'B': 2, 'C' : 3}
alfabeto['A']
lista = [1,2,3]
a,b,c = lista
a
for k in alfabeto:

    print(k)
for v in alfabeto.values():

    print(v)
for k, v in alfabeto.items():

    print(k, v)
len(alfabeto)
import pandas as pd
df = pd.read_csv('../input/train.csv')
df.shape
df.head()
df.tail()
df.sample(5)
df.sample(5).T
df.info()
df.describe()
df[20:30]
df.iloc[20:30]
df.loc[20:30]
df['humidity']
df.humidity
df.loc[:,'humidity']
df.iloc[:, 7]
type(df)
type(df['temp'])
df['temp'].describe()
df['temp'].value_counts()
df[['workingday','humidity']]
df.loc[20:30, 'workingday':'humidity']
df.at[20, 'humidity']
df.dtypes
df['datetime'] = pd.to_datetime(df['datetime'])
df.dtypes
df = pd.read_csv('../input/train.csv', parse_dates=[0])
df['datetime'].dt.month
df['month'] = df['datetime'].dt.month
df['month'] == 1
df[df['month'] == 1]
df[(df['month'] == 1) & (df['temp'] < 14)]
df[(df['month'] == 1) | (df['temp'] < 14)]
df[(df['month'] == 1) & (df['temp'] < 14)].shape, df[(df['month'] == 1) | (df['temp'] < 14)].shape 
df.nunique()
df['temp'].hist()
df['temp'].plot.box()
import seaborn as sns
sns.distplot(df['temp'], bins=10)
sns.boxplot(y='temp', data=df)
sns.boxplot(y='temp', x='season', data=df)
sns.violinplot(y='temp', data=df)
sns.violinplot(y='temp', x='season', data=df)
sns.violinplot(y='temp', x='season', data=df, hue='weather')
df.groupby('workingday')['count'].mean()
df.groupby('workingday')['count'].mean().plot.bar()
sns.barplot(y='count', x='workingday', data=df)
sns.barplot(y='count', x='season', data=df)
sns.barplot(y='count', x='season', hue='workingday', data=df)
df.groupby('month')['count'].describe()
df.groupby('month')['count'].describe()['mean'].sort_index(ascending=False).plot()
df.groupby('month')['count'].describe()['50%'].sort_index(ascending=False).plot()
sns.pairplot(x_vars='temp', y_vars='count', data=df, size=7)
sns.pairplot(x_vars='temp', y_vars='count', data=df, hue='season', size=7)
sns.pairplot(x_vars='humidity', y_vars='count', data=df, hue='season', size=7, kind='reg')
df[['humidity', 'count']].corr()
df.groupby('season')[['humidity', 'count']].corr()
df.sort_index()
df.sort_index(inplace=True)
df.sort_values(by='count', ascending=False)
df.sort_values(['count', 'registered'])
df['count'].shift(1)
df['last_count_1'] = df['count'].shift(1)
df
for i in range(1, 6):

    df['last_count_'+str(i)] = df['count'].shift(i)
df
df.info()
df.dropna()
df.fillna(-1)
df.groupby('month')['count'].mean().plot.barh()
df['month'] = df['month'].astype('category')
df['month'].cat.categories
df['month'].cat.codes
df['month'].cat.categories = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
df.groupby('month')['count'].mean().plot.barh()
df['month'].cat.as_ordered(True)
df.groupby('month')['count'].mean().sort_index(ascending=False).plot.barh()
df.set_index('datetime', inplace=True)
df.head()
df.resample('M')['count'].mean()
df.resample('M')['count'].mean().plot.barh(figsize=(20,10))
df.resample('2M')['count'].mean().plot.barh()
df.resample('Q')['count'].mean().plot.barh()
df.resample('Y')['count'].mean().plot.barh()
df.groupby(['month'])['count'].transform('mean')
df['media_mensal'] = df.groupby(['month'])['count'].transform('mean')