import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import warnings



warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))



%matplotlib inline
df = pd.read_csv('../input/Consumo_cerveja.csv')

df.columns = df.columns.str.replace(' ','_')

df.columns = df.columns.str.replace('[)(]' ,'')
df.head(10)
df.shape
df.isnull().sum()
# remoção de dados faltantes

df.dropna(subset=['Data','Final_de_Semana','Consumo_de_cerveja_litros'],how='all', inplace=True)
df['Temperatura_Media_C'] = df['Temperatura_Media_C'].str.replace(',' ,'.').astype(float)

df['Temperatura_Minima_C'] = df['Temperatura_Minima_C'].str.replace(',' ,'.').astype(float)

df['Temperatura_Maxima_C'] = df['Temperatura_Maxima_C'].str.replace(',' ,'.').astype(float)

df['Precipitacao_mm'] = df['Precipitacao_mm'].str.replace(',' ,'.').astype(float)

df['Final_de_Semana'] = df['Final_de_Semana'].astype(int)
df['Data'] = pd.to_datetime(df['Data'])

df['mes'] = pd.to_datetime(df['Data']).dt.month

df['dia_semana'] = pd.to_datetime(df['Data']).dt.dayofweek

df['dia'] = pd.to_datetime(df['Data']).dt.day
df.head()
df[['Consumo_de_cerveja_litros']].describe()
print(df.corr()['Consumo_de_cerveja_litros'].sort_values())
plt.figure(figsize=(10,10));

sns.heatmap(df.corr(), square=True ,annot=True, linewidths=1, linecolor='k');
plt.figure(figsize=(12,6))

sns.distplot(df['Consumo_de_cerveja_litros'],kde=False, bins=20);

plt.xlabel('Consumo de cerveja litros')

plt.title('\nDistribuição de frequência\n'.upper())

plt.grid(ls='-.', lw=.5)
plt.figure(figsize=(12,3))

sns.boxplot(df['Consumo_de_cerveja_litros']);

plt.xlabel('Consumo de cerveja litros')

plt.xlim(10,40)

plt.grid(ls='-.', lw=.5);
g = sns.FacetGrid(df, col='Final_de_Semana', hue='Final_de_Semana',

                  height=5, aspect=1.5)



g.map(sns.distplot,'Consumo_de_cerveja_litros', kde=False, bins=20);



g.set_xlabels('Consumo de cerveja litros');
dcc = df.groupby(['Final_de_Semana']).agg({'Consumo_de_cerveja_litros':['count','sum',

                                                                  'mean','std',

                                                                  'min','median',

                                                                  'max']}).round(3)



dcc
f, ax = plt.subplots(1,2,figsize=(16, 6))



a = sns.boxplot(x = 'Final_de_Semana', y= 'Consumo_de_cerveja_litros', data=df, ax=ax[0]);

a.set_xlabel('Final de semana')

a.set_ylabel('Consumo de cerveja litros')

a.set_title('\nConsumo de litros de cerveja\n')

a.grid(ls='-.', lw=.5)



b = sns.countplot(df['Final_de_Semana'], ax=ax[1]);

b.set_xlabel('Final de semana')

b.set_ylabel('quantidade')

b.set_title('\nQuantidade de dias\n')

b.set_ylim(0,300)

b.grid(ls='-.', lw=.5)
plt.figure(figsize=(6, 6))



plt.pie(dcc['Consumo_de_cerveja_litros']['sum'],

        labels=['Semama','Final de semana'],

        autopct='%.2f%%',

        explode=[0,0.05]);
plt.figure(figsize=(12, 6))

sns.boxplot(x='mes', y='Consumo_de_cerveja_litros', data=df);

plt.xlabel('Mês')

plt.ylabel('Consumo de cerveja litros')

plt.title('Consumo de cerveja por mês')

plt.grid(ls='-.', lw=0.2, c='k');
df.groupby(['mes','Final_de_Semana']).agg({'Consumo_de_cerveja_litros':['count','sum',

                                                                  'mean','std',

                                                                  'min','median',

                                                                  'max']}).round(3)
sns.lmplot(x='Temperatura_Media_C', y='Consumo_de_cerveja_litros', hue='Final_de_Semana',

           data=df, aspect=1.5, height=6);



plt.xlabel('Temperatura Media °C')

plt.ylabel('Consumo de cerveja litros')

plt.grid(ls='-.', lw=.5);
f, ax = plt.subplots(1,2,figsize=(16, 6))



a = sns.boxplot(x= 'mes', y='Consumo_de_cerveja_litros', hue='Final_de_Semana', data=df, ax=ax[0])

a.set_ylabel('Consumo de cerveja litros')

a.set_xlabel('Mês')

a.set_title('\nConsumo de cerveja litros por mês\n')

a.grid(ls='-.', lw=.5)



b = sns.boxplot(x= 'mes', y='Temperatura_Media_C', data=df,hue='Final_de_Semana',ax=ax[1],);

b.set_ylabel('Temperatura Media °C')

b.set_xlabel('Mês')

b.set_title('\nTemperatura Media °C por mês\n')

b.grid(ls='-.', lw=.5);

sns.lmplot(x='Precipitacao_mm', y='Consumo_de_cerveja_litros', hue='Final_de_Semana',

           data=df, aspect=1.5, height=6);



plt.xlabel('Precipitacao mm')

plt.ylabel('Consumo de cerveja litros')

plt.xlim(-5,100)

plt.grid(ls='-.', lw=.5);
plt.figure(figsize=(16,6))



sns.lineplot(x='Data',y='Consumo_de_cerveja_litros', data=df,alpha=.5);

sns.lineplot(x='Data',y=df['Consumo_de_cerveja_litros'].rolling(15).mean(), data=df,alpha=.5);



plt.ylabel('Consumo de cerveja litros')



plt.grid(ls='-.', lw=.5);
df.columns
plt.figure(figsize=(16, 6))

sns.heatmap(df.pivot_table(values='Consumo_de_cerveja_litros', index='mes', columns='dia_semana'),

            annot=True, fmt='.3f');



plt.yticks(rotation=0);
plt.figure(figsize=(16, 6))

sns.boxplot(x='dia', y='Consumo_de_cerveja_litros', data=df , hue='Final_de_Semana');

plt.xlabel('Dia')

plt.ylabel('Consumo de cerveja litros')

plt.title('Consumo de cerveja por dia')

plt.grid(ls='-.', lw=0.2, c='k');
plt.figure(figsize=(16, 6))

sns.heatmap(df.pivot_table(values='Consumo_de_cerveja_litros', index='mes', columns='dia'),

            annot=False, fmt='.3f');



plt.yticks(rotation=0);
g = sns.FacetGrid(df, col='mes', col_wrap=2,

                  aspect=1.5, height=5,hue='mes',

                  sharex=False)



g.map(sns.barplot,'dia','Consumo_de_cerveja_litros');
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics  import mean_squared_error,r2_score
X = df.drop(['Data','Temperatura_Media_C','Consumo_de_cerveja_litros'],axis=1)

y = df['Consumo_de_cerveja_litros'].values
X.head()
oneH = OneHotEncoder(categorical_features=[3,4,5,6])

sSC = StandardScaler()
X = oneH.fit_transform(X).toarray()

X = sSC.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42,shuffle=True )
def info(yTest_, pred_,sg = 'X'):

    ''' Return resultados '''

    

    print(f'''

{sg}    

    

{'-' * 65}

          

Mean squared error | {mean_squared_error(yTest_,pred_)}

r2 Score           | {r2_score(yTest_, pred_)}        

          

{'.' * 65}

y-test max         | {y_test.max()}

y-test min         | {y_test.min()}

y-test mean        | {y_test.mean()} 

y-test var         | {y_test.var()}

{'.' * 65}

                

Prediction {sg} max  | {pred_.max()}

Prediction {sg} min  | {pred_.min()}

Prediction {sg} Mean | {pred_.mean()}

Prediction {sg} Var  | {pred_.var()}



          

{'-' * 65}

''')
def graficos(y, pred):

    

    # scatter

    plt.figure(figsize=(12, 6))



    plt.plot(y,y)

    plt.scatter(pred,y, c='r', marker='o')

    plt.legend(['Real','Previsão'])

    plt.grid(ls='-.', lw=0.2, c='k');

    

    # distplot

    plt.figure(figsize=(12, 6))    

    sns.distplot(y)

    sns.distplot(pred)

    plt.legend(['Real','Previsão'])

    plt.grid(ls='-.', lw=0.2, c='k')
from sklearn.linear_model import LinearRegression
lR = LinearRegression()
%time lR.fit(X_train,y_train)
%time pred_lR = lR.predict(X_test)
info(y_test, pred_lR, 'LinearRegression') 
graficos(y_test, pred_lR)
from sklearn.neighbors import KNeighborsRegressor
kNR = KNeighborsRegressor(n_neighbors=5)
%time kNR.fit(X_train,y_train)
%time pred_kNR = kNR.predict(X_test)
info(y_test, pred_kNR, 'KNeighborsRegressor') 
graficos(y_test, pred_kNR)
from sklearn.neural_network import MLPRegressor
mLPR = MLPRegressor(hidden_layer_sizes=(100,100,100,),tol=0.000000001,

                         max_iter=1000,

                         verbose=2)
%time mLPR.fit(X_train,y_train)
%time pred_mLPR =mLPR.predict(X_test)
info(y_test, pred_mLPR, 'MLPRegressor')
graficos(y_test, pred_mLPR)