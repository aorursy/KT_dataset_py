import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn import model_selection

from sklearn import preprocessing
#df_treino = pd.read_csv('data/dataset_treino.csv')

#df_teste = pd.read_csv('data/dataset_teste.csv')

#df_loja = pd.read_csv('data/lojas.csv')



df_treino = pd.read_csv('../input/dataset_treino.csv')

df_teste = pd.read_csv('../input/dataset_teste.csv')

df_loja = pd.read_csv('../input/lojas.csv')
df_treino.head(5)
df_teste.head(5)
df_loja.head(5)
print(df_treino.shape, df_teste.shape)
df_treino.isnull().sum()
# Ajusta os valores aprensentados no describe()

pd.options.display.float_format = '{:.2f}'.format
df_treino.describe()
fig = plt.figure(figsize =(9,7))

correlations = df_treino.corr()

sns.heatmap(correlations, cmap=sns.diverging_palette(141,20,as_cmap=True), annot=True)
#sns.pairplot(df_treino)

#pd.scatter_matrix(df_treino, figsize=(6,6))
fig = plt.figure(figsize=(15,6))

print(sns.distplot(df_treino[df_treino['Open'] == 1]['Sales'], bins=80, color='darkblue'))
df_treino.dtypes
df_treino['StateHoliday'].unique()

df_treino['StateHoliday'].replace([0,'0'])

columns = {'StateHoliday':{'0':0, 'a':1, 'b':2, 'c':2}}

df_treino.replace(columns, inplace = True)

print(df_treino['StateHoliday'].unique())
df_loja['PromoInterval'].fillna('None', inplace = True)
df_loja.head(4)
df_loja_nulos = df_loja.isnull().any()[df_loja.isnull().any() == True].index

df_treino_nulos = df_treino.isnull().any()[df_treino.isnull().any() == True].index

df_teste_nulos = df_teste.isnull().any()[df_teste.isnull().any() == True].index
print('{}:\n {}\n\n'.format('df_loja',df_loja[df_loja_nulos].isnull().sum()));

print('{}:\n {}\n\n'.format('df_treino',df_treino[df_treino_nulos].isnull().sum()));

print('{}:\n {}\n'.format('df_teste',df_teste[df_teste_nulos].isnull().sum()));
print(df_teste['Open'].mode()[0])

print(df_treino['Open'].mode()[0])

# O valor mais frequente em ambos os datasets é 0. Esse será o valor de interpolação
df_teste.loc[:,'Open'].fillna(0, inplace = True)

print('Total NA {}'.format(df_teste.loc[:,'Open'].isnull().sum()))
print(round(df_loja.loc[:,'CompetitionDistance'].mean(), 2))

print(df_loja.loc[:,'CompetitionOpenSinceMonth'].mode()[0])

print(df_loja.loc[:,'CompetitionOpenSinceYear'].mode()[0])

print(df_loja.loc[:,'Promo2SinceWeek'].mode()[0])

print(df_loja.loc[:,'Promo2SinceYear'].mode()[0])
df_loja.loc[:,'CompetitionDistance'].fillna(5404.9, inplace = True)

df_loja.loc[:,'CompetitionOpenSinceMonth'].fillna(9.0, inplace = True)

df_loja.loc[:,'CompetitionOpenSinceYear'].fillna(2013.0, inplace = True)

df_loja.loc[:,'Promo2SinceWeek'].fillna(14.0, inplace = True)

df_loja.loc[:,'Promo2SinceYear'].fillna(2011.0, inplace = True)
print('Total NA {}'.format(df_loja.loc[:,'CompetitionOpenSinceMonth'].isnull().sum()))

print('Total NA {}'.format(df_loja.loc[:,'CompetitionOpenSinceYear'].isnull().sum()))

print('Total NA {}'.format(df_loja.loc[:,'Promo2SinceWeek'].isnull().sum()))

print('Total NA {}'.format(df_loja.loc[:,'Promo2SinceYear'].isnull().sum()))
df_treino = df_treino[df_treino['Sales']!=0]

print(df_treino.shape)
print('{}:\n {}\n\n'.format('df_loja',df_loja[df_loja_nulos].isnull().sum()));

print('{}:\n {}\n\n'.format('df_treino',df_treino[df_treino_nulos].isnull().sum()));

print('{}:\n {}\n\n'.format('df_teste',df_teste[df_teste_nulos].isnull().sum()));
df_treino['Date_year'] = pd.DatetimeIndex(df_treino['Date']).year

df_treino['Date_month'] = pd.DatetimeIndex(df_treino['Date']).month

df_treino['Date_day'] = pd.DatetimeIndex(df_treino['Date']).day



df_teste['Date_year'] = pd.DatetimeIndex(df_teste['Date']).year

df_teste['Date_month'] = pd.DatetimeIndex(df_teste['Date']).month

df_teste['Date_day'] = pd.DatetimeIndex(df_teste['Date']).day
subset_month = pd.DataFrame()

subset_year = pd.DataFrame()
subset_df_treino = df_treino.head(0)
print(df_treino.groupby('Date_year').size())

print(df_treino.groupby('Date_month').size())

print(df_treino.groupby('Date_day').size())

print(df_treino.shape)
df_treino = pd.merge(df_treino, df_loja, on='Store')

df_teste = pd.merge(df_teste, df_loja, on='Store')
def encoder(df):

    for column in df.loc[:,df.dtypes == object].columns:

        df[column] = preprocessing.LabelEncoder().fit(df[column]).transform(df[column])
encoder(df_treino)

encoder(df_teste)
df_treino.head(3)
df_teste.head(3)
print('Teste: {} - Treino: {}'.format(df_teste.shape, df_treino.shape))
df_treino.describe()
df_teste_previsoes = pd.Series()
df_treino.head(7)
df_teste.head(7)
# Agrupando dados por loja para testar o comportamento do modelo treinado para cada loja

df_treino_agrupado = dict(list(df_treino.groupby('Store'))) # Transforma o Store (campo agrupado) em um Id na key

df_teste_agrupado = dict(list(df_teste.groupby('Store')))
stores_id = df_teste_agrupado[1].loc[:,'Id'].unique()
def get_treino(id):

    df_treino_features_label = df_treino_agrupado[id]

    df_treino_label = df_treino_features_label.loc[:, 'Sales'] 

    df_treino_features = df_treino_features_label.drop(['Customers','Sales'], axis = 1)    

    return df_treino_features, df_treino_label



def get_teste(id):

    df_teste_features_id = df_teste_agrupado[id]

    df_teste_id = df_teste_features_id.loc[:, 'Id'] 

    df_teste_features = df_teste_features_id.drop(['Id'], axis = 1)

    return df_teste_features[df_treino_features.columns], df_teste_id
len(df_teste_agrupado.keys())
print(len(df_treino_agrupado))

print(len(df_teste_agrupado))
def rmspe(real, previsto):

    return np.sqrt(np.mean(np.square((real-previsto)/real)))
from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor
df_treino.head(3)
def optmize(x_treino, y_treino):

    modelo = XGBRegressor()

    

    params = {'n_estimators':[300], 'max_depth': [4], 'subsample': [0.2, 0.5], 'colsample_bytree': [0.5, 0.9]}

    

    grid = model_selection.GridSearchCV(estimator=modelo, param_grid=params)

    grid.fit(x_treino, y_treino.ravel())

    

    return grid.best_estimator_
df_teste_previsoes = pd.Series()
def append_predictions(store, y_previsto):

    df_teste_previsoes.append(pd.Series(y_previsto, index=store))
%%time

from sklearn.preprocessing import StandardScaler



for store in df_teste_agrupado:

    

    modelo = GradientBoostingRegressor()

    modelo = GradientBoostingRegressor(n_estimators=241)

    

    # Obtendo as lojas com o store 'x'

    df_treino_features, df_treino_label = get_treino(store)

    df_teste_features, df_teste_id = get_teste(store)

    

    # Divisão dos dados em treino e teste

    x_treino, x_teste, y_treino, y_teste = model_selection.train_test_split(df_treino_features, df_treino_label, test_size =0.20) #1/5



    # Fit do modelo

    #fit = modelo.fit(x_treino, y_treino)  # Treinamos o mode com os dados de teste (x e y, respectivamente)

    #y_previsto = fit.predict(x_teste)

    

    # Avaliação

    #print_rmspe(y_teste, y_previsto)

    print("\n>>>>>> Training Store: {}".format(store))

    #cv = model_selection.cross_val_score(estimator=modelo, X=x_treino, y=y_treino, scoring='r2')

    # Otimizando o modelo

    modelo = optmize(x_treino, y_treino)

    fit = modelo.fit(x_treino, y_treino)

    y_previsto_op = fit.predict(x_teste)

    cv_op = model_selection.cross_val_score(estimator=modelo, X=x_treino, y=y_treino, scoring='r2')

    print("r2: {}% - rmspe: {}".format(round(cv_op.mean() * 100, 2), round(rmspe(y_teste, y_previsto_op), 6)))

    

    #print("r2: {}% - {}% \n".format(round(cv.mean() * 100, 2), round(cv_op.mean() * 100, 2)))

    #print("rmspe: {} - {} \n".format(round(rmspe(y_teste, y_previsto), 2),rmspe(y_teste, y_previsto_op)))

    

    #print(previsto)

    #print(df_teste_id)

    

    # Anexando resultados

    df_teste_previsoes = df_teste_previsoes.append(pd.Series(modelo.predict(df_teste_features), index=df_teste_id))   

    #print(modelo)

    #print()
%%time



df_treino = df_treino[df_treino_features.columns]

df_teste = df_teste[df_treino_features.columns]



parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'learning_rate': [.03, 0.05, .07], #so called `eta` value

              'max_depth': [5, 6, 7],

              'min_child_weight': [2,4],

              'subsample': [0.7],

              'colsample_bytree': [0.7],

              'n_estimators': [400]}



modelo = XGBRegressor()



x_treino, x_teste, y_treino, y_teste = model_selection.train_test_split(df_treino[df_treino_features.columns], df_treino.loc[:,'Sales'], test_size =0.20)



grid = model_selection.GridSearchCV(estimator=modelo, param_grid=parameters)

grid.fit(x_treino, y_treino.ravel())



previsoes = pd.Series(grid.best_estimator_.predict(df_teste))



print(grid.best_estimator_)

"""

XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=0.7, gamma=0, importance_type='gain',

       learning_rate=0.07, max_delta_step=0, max_depth=7,

       min_child_weight=4, missing=None, n_estimators=400, n_jobs=1,

       nthread=4, objective='reg:linear', random_state=0, reg_alpha=0,

       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,

       subsample=0.7)

"""



previsoes = pd.DataFrame({'real':y_teste, 'previsto':grid.best_estimator_.predict(x_teste)}) 



previsoes.head(4)



rmspe(previsoes.loc[:,'real'], previsoes.loc[:,'previsto']) # 0.19298122496957015
# Gravando as predições do modelo

df_teste_previsoes_EXPORT = pd.DataFrame({'Id':df_teste_previsoes.index,'Sales': list(df_teste_previsoes.values)})
df_teste_previsoes_EXPORT.head(5)
# Exportando o resultado 

df_teste_previsoes_EXPORT.to_csv('df_teste_previsoes_EXPORT.csv', index=False)