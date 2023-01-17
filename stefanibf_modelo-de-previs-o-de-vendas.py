# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importando os modulos

from sklearn.model_selection import train_test_split # Dividir os dados em treino e teste

from sklearn.metrics import r2_score # Avaliar o r2 do modelo

from sklearn.preprocessing import LabelEncoder # Converter os dados categoricos em numericos

from sklearn.model_selection import GridSearchCV # Testar os melhores parametros

from xgboost import XGBRegressor # Modelo para Regressao



# Modulos para analise expolratoria

from IPython.core.pylabtools import figsize

import matplotlib.pyplot as plt

%matplotlib inline



# Desativando warnings

import warnings

warnings.filterwarnings('ignore')



# Definindo o display para exibir todas as colunas

pd.set_option('display.max_columns', 30)
# Funcao para imprimir uma tabela com valores missing

def contador_na(dados):

    # Analise de valores missing

    valores_total = dados.isna().sum() # contagem de valroes missing

    valores_perc = round((dados.isna().sum() / len(dados)) * 100 ,2) # % de valroes missing

    valores_na = pd.concat([valores_total, valores_perc], axis = 1).sort_values(by = 0, ascending=False) # tabela com os valores e % ordenados

    valores_na.columns = ['Quantidade', 'Proporcao']



    # Visualizando o resutlado

    return(valores_na[valores_na.Quantidade > 0])



# funcao para converter os dados para numericos

def convert_num(dados):

    le = LabelEncoder()

    for column in dados.columns:

        if dados[column].dtype == "O":

            dados[column] = le.fit_transform(dados[column])

    return(dados)



# Funcao para remover outliers com base no intervalo interquartil

def remove_out(dados, item):

    q1 = dados[item].quantile(0.25)

    q3 = dados[item].quantile(0.75)

    

    iqr = q3 - q1

    

    menor = q1 - 3 * iqr

    maior = q3 + 3 * iqr

        

    # altera os outliers pelo valor da moda

    dados = dados.loc[(dados[item] < maior) & (dados[item] > menor) ,]

     

    return dados



# Funcao para avaliar o modelo com base nos dados de teste

def predicao(modelo, x_treino, y_treino, x_teste, y_teste):

    

    # Prevendo os dados de teste

    pred = modelo.predict(x_teste)

    pred = np.exp(pred)

    y_teste = np.exp(y_teste) 

    

    # avaliando o modelo 

    

    #RMSE

    rmsep = np.sqrt(np.mean((pred/y_teste - 1) ** 2))

    

    # R2

    r2 = r2_score(y_teste, pred)

    

    return rmsep, r2
# Lendo o conjunto de dados de treino

treino = pd.read_csv('../input/dataset_treino.csv')



# Inserindo o atributo ID com valor default para concatenar os dados

treino.insert(0, 'Id', 0)



# Lendo o conjunto de dados de teste

teste = pd.read_csv('../input/dataset_teste.csv')



# Inserindo o atributo Sales e Customers nos dados de teste para obter as mesmas colunas dos dados de treino

teste.insert(4, 'Sales', np.nan)



# Inserindo o atributo Customers

teste.insert(5, 'Customers', np.nan)



# Criando um novo dataset com os dados agrupados de treino e teste

df = pd.concat([treino, teste]).reset_index(drop=True)



# Visualizando os dados

df.head()
# Formatando o atributo date para o tipo data

df.Date = pd.to_datetime(df['Date'],format='%Y-%m-%d')



# Criando alguns atributos para trabalhar com series temporais



# Atributo mes

df.insert(3, 'Month', df.Date.dt.month)



# Tipo de dia da semana (util ou nao util)

day_type = ['weekend' if item in [6,7]  else 'weekday' for item in df.DayOfWeek]

df.insert(4, 'DayType', day_type)



# Dia do mes

df.insert(5, 'Day', df.Date.dt.day)



# Trimestre

df.insert(6, 'Quarter', df.Date.dt.quarter)



# Ano

df.insert(7, 'Year', df.Date.dt.year)



# Semana

df.insert(8, 'Week', df.Date.dt.week)



# Dia do ano

df.insert(9, 'Dayofyear', df.Date.dt.dayofyear)



# Visualizando os novos atributos

df.head()
# Estrutura dos dados

df.info()
# Valores unicos

df.nunique()
# O atributo StateHoliday possui alguns itens com valores 0 e outros strings

print(df.StateHoliday.unique())



# Convertendo os dados apenas para 0 e 1, onde 0 = dia normal e 1 = feriado

df.StateHoliday.replace([0, '0', 'a', 'b', 'c'], [0, 0, 1, 1, 1], inplace= True)
# Buscando por valores missing

contador_na(df)
# Apenas o atributo Open possui valores NA, ja que Customers e Sales foram incluidos manualmente

print(df.loc[df.Open.isnull(),].DayOfWeek.unique())



# Subistituindo valores NA por 1 pois se tratam de dias uteis

df.Open.fillna(1, inplace=True)
# Visivelmente existe uma correlacao positiva entre as vendas e a quantidade de clientes



#scatter plot de vendas x quantidade de clientes

plt.style.use('seaborn-white')

plt.scatter(x = df.loc[df.Sales > 0,'Customers'], y = df.loc[df.Sales > 0, 'Sales'].dropna(), alpha=0.5)

plt.title('Sales x Customers')

plt.xlabel('Customers')

plt.ylabel('Sales')

plt.show()
# Histograma do atributo Customers

plt.style.use('seaborn-white')

plt.hist(x = df.loc[df.Sales > 0,'Customers'], bins = 80)

plt.xlabel('Customers'); plt.ylabel('Frequencia')

plt.title('Distribuicao da quantidade de clientes')

plt.show()
# Resumo estatistico por ano

df.groupby('Year')['Customers'].describe()
# Removendo outliers para criar um atributo com a media de clientes por lojas



df_cust = df.loc[df.Customers > 0,:]

df_cust = remove_out(df_cust, 'Customers')
# O Desvio padrão diminuiu apos a remocao dos outliers

df_cust.groupby('Year')['Customers'].describe()
# Com a reducao dos outliers, 1 loja foi desconsiderada, mas sera substituida pela media

df_cust.Store.nunique() - df.Store.nunique()
# Criando faixas por media de clientes

customers_mean = pd.DataFrame(df_cust.groupby('Store').Customers.mean())



# Dataset com a media por lojas

customers_med = pd.DataFrame(customers_mean).reset_index(drop = False)

customers_med.rename(columns = {'Customers':'Customers_mean'}, inplace = True)



# Visualizando os dados

customers_med.head()
# Lendo o dataset com os descritivos das lojas

lojas = pd.read_csv('../input/lojas.csv')



# agrupando as medias de clientes com o dataset lojas

lojas = lojas.merge(customers_med, left_on='Store', right_on='Store', how = 'outer')



lojas.head()
# Buscando por valores missing

contador_na(lojas)
# Lojas que nao participam da promocao irao receber valores zero

lojas.loc[lojas.Promo2SinceWeek.isnull(), ['Promo2SinceWeek', 'Promo2SinceYear']] = 0



# apenas 3 itens nao tem a informacao da distancia, onde serao subsituidos pelas media

lojas.loc[lojas.CompetitionDistance.isnull(),'CompetitionDistance'] = lojas.CompetitionDistance.mean()



# demais valores NA serao subistituidos pela moda

lojas.loc[lojas.CompetitionOpenSinceMonth.isnull(),'CompetitionOpenSinceMonth'] = lojas.CompetitionOpenSinceMonth.mode()[0]

lojas.loc[lojas.CompetitionOpenSinceYear.isnull(),'CompetitionOpenSinceYear'] = lojas.CompetitionOpenSinceYear.mode()[0]



# Customers_mean com valor NA sera substituido pela media

lojas.Customers_mean.fillna(customers_med.Customers_mean.mean(), inplace = True)



# Dividindo as colunas atributo intervalo da promocao

lojas = pd.concat([lojas.drop('PromoInterval', axis = 1), lojas.PromoInterval.str.split(',',expand=True)], axis = 1)



# Convertendo os atributo nome mes mes para numero

for i in lojas.iloc[:,10:14].columns:

    lojas[i].replace(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'],

                     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace = True)



# Renomando as colunas criadas com o intervalo da promo2

lojas.rename(columns = {0:'m0', 1:'m1', 2:'m2', 3:'m3'}, inplace = True)



# Alterando valores NA dos atributos do intervalo por zero

lojas.fillna(0, inplace = True)



# Dados tratados

lojas.head()
# Mesclando os atributos das lojas aos dados de treinamento

df_geral = df.merge(lojas, left_on='Store', right_on='Store')
# Tratando os atributos de promocoes e concorrentes



# Diferenca entre a data da venda e o inicio da promocao

temp = (df_geral.Year - df_geral.Promo2SinceYear) * 12 + ((df_geral.Week - df_geral.Promo2SinceWeek) / 4)

temp = [1 if valor >= 0 else 0 for valor in temp]



# tempo a partir do inicio da promocao

df_geral['Promo2Inicio'] = temp



#Promo2 ativa (periodo da promocao ativa de acordo com os meses)

df_geral['Promo2_active'] = 0

df_geral.loc[((df_geral.Month == df_geral.m0) | (df_geral.Month == df_geral.m1) |

              (df_geral.Month == df_geral.m2) | (df_geral.Month == df_geral.m3)) & df_geral.Promo2Inicio == 1,'Promo2_active'] = 1



# Tempo em meses a partir do inicio da atividade da concorrencia

temp = (df_geral.Year - df_geral.CompetitionOpenSinceYear) * 12 + (df_geral.Month - df_geral.CompetitionOpenSinceMonth)

temp = [valor if valor >= 0 else 0 for valor in temp]





# Atributo tempo concorrencia

df_geral['Time_Competition'] = temp

df_geral.loc[df_geral.CompetitionOpenSinceYear == 0, 'Time_Competition'] = 0
# Histograma das vendas

plt.style.use('seaborn-white')



plt.hist(df_geral.loc[df_geral.Sales > 0, 'Sales'], bins = 80, edgecolor = 'k');

plt.xlabel('Sales'); plt.ylabel('Frequencia'); 

plt.title('Frequencia do atributo Sales');
# Excluindo atributos nao preditivos

df_geral.drop(['Date', 'Store', 'Customers', 'm0', 'm1', 'm2', 'm3'], axis = 1, inplace = True)



# Convertendo os dados do tipo strings para numericos

df_geral = convert_num(df_geral)
# Resumo estatistico dos dados

df_geral.describe().T
# Gerando os dados de teste tratados

teste = df_geral.loc[df_geral.Sales.isnull(),].sort_values(by = 'Id')



# Gerando os novos dados para treinamento

# Exclui vendas igual a zero

treino = df_geral.loc[df_geral.Sales > 0,].drop(['Id', 'Open'], axis = 1)



# removendo outliers daos dados de treino

treino = remove_out(treino, 'Sales')
# Dividindo os dados em X e y

X = treino.drop('Sales', axis = 1)

y = np.log(treino.Sales)



# Busca por atributos de maior importancia com base no modelo xgb

# tree_method='gpu_exact utilizando GPU

modelo = XGBRegressor(tree_method='gpu_exact').fit(X,y)



# Dataset com as importancia dos atributos

feat_imp = pd.DataFrame({'imp':modelo.feature_importances_, 'feat': X.columns}).sort_values(by = 'imp')



# objeto com os nomes dos atributos por ordem de importancia

importancias = feat_imp.sort_values(by = 'imp', ascending = False).feat.values



# Plotando os atributos por ordem de importancia

plt.style.use('seaborn-white')

figsize(8, 10)



# Criando um grafico de barras horizontais

feat_imp.plot(x = 'feat', y = 'imp', kind = 'barh',color = 'blue', edgecolor = 'black')



plt.ylabel(''); plt.xlabel(''); plt.yticks(size = 14); plt.xticks(size = 14)

plt.title('Importancia dos atributos', size = 20)

plt.show()
# Cheando qual seria o numero ideal de atributos para obter o melhor modelo



# Separando os dados para treino e teste

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 142)



cont = 0

rmse_total = []

r2_total = []



for i, item in enumerate(importancias):

    

    # a cada rodada sera criado um modelo com o incremento de um atributo

    cont = i + 1

    x_aux_train = x_train.loc[:,importancias[:cont]]

    x_aux_test = x_test.loc[:,importancias[:cont]]

    

    modelo = XGBRegressor(tree_method='gpu_exact').fit(x_aux_train,y_train)

    

    # avaliando o modelo

    rmse, r2  = predicao(modelo, x_aux_train, y_train, x_aux_test, y_test)

    

    rmse_total.append(rmse)

    r2_total.append(r2)
# Dataset com o resultado da busca pelo total de atributos com melhor resultado

resultado = pd.DataFrame({'R2': r2_total, 'RMSE': rmse_total})



# Visualizando os resultados

resultado.sort_values(by = 'RMSE')
# Selecionando os 15 primeiros atributos por importancia

imp = importancias[:15]



print(imp)
# Dividindo os dados em X e y de acordo com os features selecionados

X = treino[imp]

y = np.log(treino.Sales)



# Separando os dados para treino e teste

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 142)
# Criando e avaliando o modelo de regressao utilizando XGB

modelo_xgb = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

                      colsample_bytree=0.7, gamma=0, learning_rate=0.03, max_delta_step=0,

                      max_depth=12, min_child_weight=2, missing=None, n_estimators=300,

                      n_jobs=1, nthread= -1, objective='reg:linear',

                      reg_alpha=0, reg_lambda=1, scale_pos_weight=1, random_state = 142,

                      silent=True, subsample=0.9, tree_method='gpu_exact').fit(x_train, y_train)



# avaliando o modelo com base nos dados de teste

rmse, r2  = predicao(modelo_xgb, x_train, y_train, x_test, y_test)



# Visualizando os resultados

print('XGB: RMSE = {:.4} R2 = {:.2%}'.format(rmse, r2))
# Busca pelos melhores parametros



# Parametros

#param_grid = {'objective':['reg:linear'],

#              'learning_rate': [0.05, 0.04, 0.03, 0.02, 0.01],

#              'max_depth': [6, 8, 10, 12],

#              'subsample': [0.7, 0.8, 0.9, 1],

#              'colsample_bytree': [0.7, 0.8, 0.9, 1],

#              'n_estimators': [1000,2000,3000]}



# Criando o modelo

#modelo = XGBRegressor(tree_method = 'gpu_exact', silent = 1) 



# Buscando os parametros

#gs_cv = GridSearchCV(modelo, param_grid, scoring='neg_mean_squared_error',

#                     cv = 4,

#                     verbose=True)



# fit do modelo

#gs_cv.fit(x_train, y_train)



#Melhores parametros

#gs_cv.best_params_
# Modelo versão final

modelo_xgb = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

                      colsample_bytree=0.7, gamma=0, learning_rate=0.03, max_delta_step=0,

                      max_depth=10, min_child_weight=4, missing=None, n_estimators=3000,

                      n_jobs=1, nthread= -1, objective='reg:linear', num_boost_round = 100,

                      reg_alpha=0, reg_lambda=1, scale_pos_weight=1, random_state = 142,

                      silent=True, subsample=0.9, tree_method='gpu_exact').fit(x_train, y_train)



# avaliando o modelo com base nos dados de teste

rmse, r2  = predicao(modelo_xgb, x_train, y_train, x_test, y_test)



# Visualizando os resultados

print('XGB: RMSE = {:.4} R2 = {:.2%}'.format(rmse, r2))
# criando um subsete com os atributos selecionados

x_sub = teste[imp]



# prevendo os dados de validacao

pred_sub = modelo_xgb.predict(x_sub)



# retomando os dados para formato normal

pred_sub = np.exp(pred_sub)



# Carregando os valores previstos

teste.Sales = pred_sub



# Alterando os valores das vendas onde open for igual a zero

teste.loc[teste.Open == 0, 'Sales'] = 0



# Gerando dados para submissao

submissao = teste[['Id', 'Sales']]



# salvando o arquivo

submissao.to_csv('submissao.csv', index = False)