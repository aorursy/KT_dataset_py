import warnings
warnings.filterwarnings('ignore')

import sys

# Bibliteca para Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Biblioteca para Manipulação de Dados
import pandas as pd
import numpy as np

# Biblioteca para Análise e modelagem de séries temporais
from statsmodels.tsa.seasonal import seasonal_decompose

# Imports para formatação dos gráficos
import matplotlib.cbook
import matplotlib as m
m.rcParams['axes.labelsize'] = 14
m.rcParams['xtick.labelsize'] = 12
m.rcParams['ytick.labelsize'] = 12
m.rcParams['text.color'] = 'k'

# Imports para criação e validação dos modelos temporais
from statsmodels.tsa.arima_model import ARIMA
import sklearn
from sklearn.metrics import mean_squared_error 
import itertools

# Import para Padronização dos dados
from sklearn.preprocessing import StandardScaler

# Import utilizado para Feature Selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV

# Import utilizado para análise de MultiColinearidade dos Dados
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Import para Análise de Estacionaridade nos Modelos de Séries Temporais
from statsmodels.tsa.stattools import adfuller

# Import para obter os feriados usados no Modelo SARIMAX Exógeno
import holidays

# Import Modelo de Séries Temporais Multivariado
from statsmodels.tsa.vector_ar.var_model import VAR

# Imports para o modelo Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

# Import para aplicar Cross Validation
from sklearn.model_selection import cross_val_score

# Import para definir os KFold do CV
from sklearn.model_selection import RepeatedKFold

# Import para Otimização de HiperParametros
from sklearn.model_selection import GridSearchCV

# Import para o modelo de Regressão Linear
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.stats as sms

from matplotlib.pylab import rcParams 
rcParams['figure.figsize'] = 20,10
matplotlib.style.use('ggplot')

%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Import Dataset de Treino.
df_train = pd.read_csv('/kaggle/input/dataset_training.csv')
df_test = pd.read_csv('/kaggle/input/dataset_training.csv')
# Unificando o df_train e df_test pois lá na frente iremos utilizar séries temporais e os dados a serem previstos 
# devem ser os mais atuais possível.
df_full = pd.concat([df_train, df_test])
df_full = df_full.sort_values(['date'], ascending=True)
df_full.head()
df_train.shape
df_test.shape
df_full.shape
# Considerando a variável rv1 como a TARGET do dataset
column_target = 'rv1'
# Analisando as informações de cada dado
df_full.info()
# Resumo estatístico dos dados
df_full.describe()
def formata_dados(dataset):
    # Convertendo a coluna date para o formato DateTime
    dataset['date'] = pd.to_datetime(dataset['date'], format='%Y-%m-%d %H:%M:%S')
     
    # Aplicando o Split de alguns dados do campo Date  Obs.: O Campo Day_of_week vai ser transformado em inteiro.
    dataset['Month'] = dataset['date'].dt.month
    dataset['day'] = dataset['date'].dt.day
    dataset['hour'] = dataset['date'].dt.hour
    dataset['Day_of_week'] = dataset['date'].dt.dayofweek

    # Renomeando a variável WeekStatus para Weekend. Essa variável terá valores 0 e 1. {0 : Weekday , 1: Weekend}
    dataset.rename(columns={'WeekStatus':'Weekend'}, inplace=True)

    # Segunda Feira = 0 ... Sabado = 5, Domingo = 6
    dataset['Weekend'] = 0
    dataset.loc[(dataset.Day_of_week == 5) | (dataset.Day_of_week == 6), 'Weekend'] = 1    
    
    # Padronizando o nome das colunas para Lower
    dataset.columns = map(str.lower, dataset.columns)
    
    return dataset
df_train = formata_dados(df_train)
df_test = formata_dados(df_test)
df_full = formata_dados(df_full)
df_full.columns 
df_full.head(5)
df_full.hist(figsize=(10,10));
# Quantidade de dados coletados Weekend e WeekDay
fig, ax = plt.subplots(figsize = (10,6))

sns.countplot(df_full['weekend'])

ax.set_title('Dados Coletados Weekday vs Weekend')
ax.set_ylabel('Quantidade')
ax.set_xlabel('0: Weekday / 1: Weekend');
fig, ax = plt.subplots(figsize = (10,6))

df_full.groupby('weekend').mean()[column_target].plot(kind='bar')

ax.set_title('Gasto Médio Energia Weekday vs Weekend')
ax.set_ylabel('Volume Energia')
ax.set_xlabel('0: Weekday / 1: Weekend');
df_full.groupby('day_of_week').mean()[column_target]
fig, ax = plt.subplots(figsize = (10,6))

df_full.groupby('day_of_week').mean()[column_target].plot(kind='bar')

ax.set_title('Média Energia Gasta por Dia')
ax.set_ylabel('Volume Energia')
ax.set_xlabel('Dia Semana');
corrmat = df_full.corr()
top_corr_features = corrmat.index

plt.figure(figsize=(20,20))
plt.title('\nHeatmap Correlação de Variáveis\n', fontsize=18)

g=sns.heatmap(df_full[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# Variavel nsm e hour são altamente correlacionadas. Drop Coluna Hour.
df_train.drop('hour', axis=1, inplace=True)
df_test.drop('hour', axis=1, inplace=True)
df_full.drop('hour', axis=1, inplace=True)
# Variavel rv1 e rv2 são altamente correlacionadas e possuem a mesma informação. Vamos Dropar uma das colunas.
df_train.drop('rv2', axis=1, inplace=True)
df_test.drop('rv2', axis=1, inplace=True)
df_full.drop('rv2', axis=1, inplace=True)
# Definindo a variavel date como Index do Dataset. 
# Essa operação transforma os dados em séries possibilitando a análise como Series Temporais.
df_full.index = df_full['date']
df_full = df_full.drop('date', 1)
df_full.head(5)
df_train.index = df_train['date']
df_train = df_train.drop('date', 1)

df_test.index = df_test['date']
df_test = df_test.drop('date', 1)
# Utilizando Random Forest Regressor para identificar as melhores variáveis preditoras
X = df_full.loc[:, df_full.columns != column_target]
y = df_full.loc[:, df_full.columns == column_target]
model2 = RandomForestRegressor()
rfecv2 = RFECV(estimator=model2, cv=4)
m_rfecv = rfecv2.fit(X,y)
# Plotando o resultado do feature selection com RandomForestRegressor
plt.figure()
plt.title("\n Feature Selection com RandomForestRegressor\n")
plt.xlabel("\nNúmero de Features Consideradas")
plt.ylabel("\nCross validation score (# Classificações corretas)")
plt.plot(range(1, len(m_rfecv.grid_scores_) + 1), m_rfecv.grid_scores_)
plt.show()

# Print dos resultados
print("\nVariáveis Preditoras:", X.columns[:-1])
print("\nVariáveis Selecionadas: %s" % m_rfecv.support_)
print("\nRanking dos Atributos: %s" % m_rfecv.ranking_)
print("\nNúmero de Melhores Atributos: %d" % m_rfecv.n_features_)
# Analisando Multicolinearidade entre os dados
vif = pd.DataFrame()
vif['Feature']= df_full.loc[:, df_full.columns != column_target].columns
vif['VIF Factor'] = [variance_inflation_factor(df_full.loc[:, df_full.columns != column_target].values, i) for i in range(df_full.loc[:, df_full.columns != column_target].shape[1])]

vif.round(1).head(100)
# Pegando a data a cada X dias
def qtd_data(df_series, qtd):
    list_date = []
    control = 0
    
    for i in df_series.index.values:
        if control > qtd:
            control = 1        
    
        if control == 0 or control == qtd:
            list_date.append(i)
            control = control + 1
        else:
            control = control + 1        
    
    return list_date
df_full_series_Dia = df_full[column_target].resample('D').mean()
mean_ = [np.mean(df_full_series_Dia[:x]) for x in range(len(df_full_series_Dia))]
mean_series_Dia = pd.Series(mean_)
mean_series_Dia.index = df_full_series_Dia.index

fig, ax = plt.subplots(figsize = (12,4))
plt.plot(df_full_series_Dia, label = 'Gasto Diário')
plt.plot(mean_series_Dia, label = 'Média')
plt.legend()
plt.xticks(rotation = 90)
plt.xticks(qtd_data(df_full_series_Dia, 7))

ax.set_title('Gasto Médio Energia Diário');
matplotlib.style.use('ggplot')

fig, ax = plt.subplots(figsize = (12,12))

res = seasonal_decompose(df_full_series_Dia)

plt.subplot(411)
plt.plot(res.observed, label = 'Série Original')
plt.legend(loc = 'best')
plt.xticks(rotation = 90)
plt.xticks(qtd_data(df_full_series_Dia, 7))
plt.title('Análise Gasto Médio Energia Diário')

plt.subplot(412)
plt.plot(res.trend, label = 'Tendência')
plt.legend(loc = 'best')
plt.xticks(rotation = 90)
plt.xticks(qtd_data(df_full_series_Dia, 7))

plt.subplot(413)
plt.plot(res.seasonal, label = 'Sazonalidade')
plt.legend(loc = 'best')
plt.xticks(rotation = 90)
plt.xticks(qtd_data(df_full_series_Dia, 7))

plt.subplot(414)
plt.plot(res.resid, label = 'Resíduos')
plt.legend(loc = 'best')
plt.xticks(rotation = 90)
plt.xticks(qtd_data(df_full_series_Dia, 7))

plt.tight_layout();
# Função para testar a estacionaridade
def testa_estacionaridade(serie, tipo):
    if tipo is None:
        tipo = ''
    else:
        tipo = '('+tipo+')'
    
    
    # Calcula estatísticas móveis
    rolmean = serie.rolling(window = 12).mean()
    rolstd = serie.rolling(window = 12).std()

    # Plot das estatísticas móveis
    orig = plt.plot(serie, color = 'blue', label = 'Original')
    mean = plt.plot(rolmean, color = 'red', label = 'Média Móvel')
    std = plt.plot(rolstd, color = 'black', label = 'Desvio Padrão')
    plt.legend(loc = 'best')
    plt.title('Estatísticas Móveis - Média e Desvio Padrão ' + tipo)
    plt.xticks(rotation = 45)
    plt.show()
    
    # Teste Dickey-Fuller:
    # Print
    print('\nResultado do Teste Dickey-Fuller:\n')

    # Teste
    dfteste = adfuller(serie, autolag = 'AIC')

    # Formatando a saída
    dfsaida = pd.Series(dfteste[0:4], index = ['Estatística do Teste',
                                               'Valor-p',
                                               'Número de Lags Consideradas',
                                               'Número de Observações Usadas'])

    # Loop por cada item da saída do teste
    for key, value in dfteste[4].items():
        dfsaida['Valor Crítico (%s)'%key] = value

    # Print
    print (dfsaida)
    
    # Testa o valor-p
    print ('\nConclusão:')
    if dfsaida[1] > 0.05:
        print('\nO valor-p é maior que 0.05 e, portanto, não temos evidências para rejeitar a hipótese nula.')
        print('Essa série provavelmente não é estacionária.')
    else:
        print('\nO valor-p é menor que 0.05 e, portanto, temos evidências para rejeitar a hipótese nula.')
        print('Essa série provavelmente é estacionária.')
testa_estacionaridade(df_full_series_Dia, 'Diario')
df_train.shape[0]
df_test.shape
# Split do df_full em 2 novos dfs:
# Irei substituir as informações no df_train e df_test, porém deixando os dados com a msm qtd q antes.
df_train = df_full.iloc[0:df_train.shape[0]+1]
df_test = df_full.iloc[df_train.shape[0]: ]
df_train_series_Dia = df_train[column_target].resample('D').mean()
df_test_series_Dia = df_test[column_target].resample('D').mean()
# Função para medir o desempenho do modelo
def performance(y_true, y_pred): 
    mse = ((y_pred - y_true) ** 2).mean()
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return( print('MSE das previsões é {}'.format(round(mse, 2))+
                  '\nRMSE das previsões é {}'.format(round(np.sqrt(mse), 2))+
                  '\nMAPE das previsões é {}'.format(round(mape, 2))))
# Vamos definir p, d e q para que tenham valores entre 0 e 2 e testaremos as combinações.
p = d = q = range(0, 2)
# Lista de combinações de p, d, q
pdq = list(itertools.product(p, d, q))
pdq
# Lista de combinações dos hiperparâmetros sazonais P, D e Q
# Estamos usando List Comprehension
# 7 representa a sazonalidade
seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]
seasonal_pdq
print('\nExemplos de Combinações dos Hiperparâmetros Para o Modelo SARIMA:\n')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[3], seasonal_pdq[4]))
# Grid Search
#warnings.filterwarnings("ignore")

# Menor valor possível para a estatística AIC (nosso objetivo na otimização do modelo)
lowest_aic = sys.maxsize
lowest = ''

# Loop
for param in pdq:
    
    for param_seasonal in seasonal_pdq:
        try:
            # Cria o modelo com a combinação dos hiperparâmetros
            mod = sm.tsa.statespace.SARIMAX(df_train_series_Dia,
                                            order = param,
                                            seasonal_order = param_seasonal,
                                            enforce_stationarity = False,
                                            enforce_invertibility = False)
            
            # Treina o modelo
            results = mod.fit()
            
            # Print
            print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            
            # Coleta o menor valor de AIC
            if lowest_aic >  results.aic:
                lowest = 'SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic)
                lowest_aic = results.aic
        except:
            continue

print ("\nModelo com Menor Valor de AIC: " + lowest)
# Treina o modelo com a melhor combinação de hiperparâmetros
modelo_sarima = sm.tsa.statespace.SARIMAX(df_train_series_Dia,
                                             order = (0, 0, 1),
                                             seasonal_order = (0, 1, 1, 7),
                                             enforce_stationarity = False,
                                             enforce_invertibility = False)
# Treinamento (Fit) do modelo
modelo_sarima_fit = modelo_sarima.fit()
# Sumário do modelo
print(modelo_sarima_fit.summary())
df_test_series_Dia.index.values.min()
df_test_series_Dia.index.values.max()
# Vamos fazer previsões dos dados do TESTE
sarima_predict = modelo_sarima_fit.get_prediction(start = pd.to_datetime('2016-04-23'), 
                                                       end = pd.to_datetime('2016-05-27'),
                                                       dynamic = False)
# Intervalo de confiança
sarima_predict_conf = sarima_predict.conf_int()
sarima_predict_conf
rcParams['figure.figsize'] = 20,8

# Plot dos valores observados
ax = df_full_series_Dia.plot(label = 'Valores Observados', color = '#2574BF')

# Plot dos valores previstos
sarima_predict.predicted_mean.plot(ax = ax, 
                                     label = 'Previsões SARIMA(0, 0, 1)x(0, 1, 1, 7)', 
                                     alpha = 0.7, 
                                     color = 'red') 

# Plot do intervalo de confiança
ax.fill_between(sarima_predict_conf.index,
                # lower sales
                sarima_predict_conf.iloc[:, 0],
                # upper sales
                sarima_predict_conf.iloc[:, 1], color = 'k', alpha = 0.1)

# Títulos e Legendas
plt.title('Previsão de Consumo Médio Energia Por Dia com Modelo ARIMA Sazonal Dados Teste')
plt.xlabel('Data')
plt.ylabel('Média Consumo')
plt.legend()
plt.show()
# Calculando a performance
sarima_results_treino = performance(df_test_series_Dia, sarima_predict.predicted_mean)
sarima_results_treino
feriados = pd.Series()

# Como nosso dataset aparenta ser da Bélgica, baseado na informação do aeroporto, estamos procurando feriados no ano.
for i, feriado in holidays.Belgium(years = [2016]).items():
    feriados[i] = feriado
feriados_df = pd.DataFrame(feriados)
# Reset do index para ajustar os nomes das colunas
feriados_df.reset_index(level = 0, inplace = True)
feriados_df.columns = ['data_feriado', 'feriado']
# Visualiza
feriados_df.head()
feriados_df['data_feriado'] = pd.to_datetime(feriados_df['data_feriado'])
# Função
def adiciona_feriado(x):
    
    # Aplica a regra
    batch_df = feriados_df.apply(lambda y: 1 if (x['data'] == y['data_feriado']) else None, axis=1)
    
    # Limpa valores nulos
    batch_df = batch_df.dropna(axis = 0, how = 'all')  
    
    # Se estiver vazio, preenche com 0
    if batch_df.empty:
        batch_df = 0
    else: 
        batch_df = batch_df.to_string(index = False)
        
    return batch_df
# Cria um dataframe a partir da série
Frame_means = pd.DataFrame(df_train_series_Dia)

# Reset do índice para ajustar as colunas (podia ter feito tudo isso em um comando, ms didaticamente deixamos assim)
Frame_means.reset_index(level = 0, inplace = True)

# Ajusta o nome das colunas
Frame_means.columns = ['data', 'rv1']
Frame_means.head()

# Aplicamos a função e criamos a coluna feriado
Frame_means['feriado'] = Frame_means.apply(adiciona_feriado, axis = 1)

# Convertendo a coluna feriado para inteiro
Frame_means['feriado'] = pd.to_numeric(Frame_means['feriado'], downcast = 'integer')

# Vamos definir a order_date como índice
Frame_means.set_index("data", inplace = True)
df_train.columns
# E somente o feriado (mais uma constante requerida pelo statsmodels) na série de feriado
exog_var_treino = sm.add_constant(Frame_means['feriado'])
exog_var_treino
# Grid Search
warnings.filterwarnings("ignore")

# Menor valor possível para a estatística AIC (nosso objetivo na otimização do modelo)
lowest_aic = sys.maxsize
lowest = ''

# Loop
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            
            # Cria o modelo com a combinação dos hiperparâmetros
            mod = sm.tsa.statespace.SARIMAX(df_train_series_Dia,
                                            exog_var_treino,
                                            order = param,
                                            seasonal_order = param_seasonal,
                                            enforce_stationarity = False,
                                            enforce_invertibility = False)
            
            # Treina o modelo
            results = mod.fit()
            
            # Print
            print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            
            # Coleta o menor valor de AIC
            if lowest_aic >  results.aic:
                lowest = 'SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic)
                lowest_aic = results.aic
        except:
            continue

print ("\nModelo com Menor Valor de AIC: " + lowest)
# Treina o modelo com a melhor combinação de hiperparâmetros
modelo_sarima_v2 = sm.tsa.statespace.SARIMAX(df_train_series_Dia,
                                             exog_var_treino,
                                             order = (0, 0, 1),
                                             seasonal_order = (0, 1, 1, 7),
                                             enforce_stationarity = False,
                                             enforce_invertibility=False)
# Treinamento (Fit) do modelo
modelo_sarima_v2_fit = modelo_sarima_v2.fit()
# Sumário do modelo
print(modelo_sarima_v2_fit.summary())
df_train_series_Dia.index.values.min()
df_train_series_Dia.index.values.max()
# Vamos fazer previsões um passo a frente
sarima_predict_2_treino = modelo_sarima_v2_fit.get_prediction(start = pd.to_datetime('2016-01-20'), 
                                                       end = pd.to_datetime('2016-03-19'),
                                                       exog = exog_var_treino['20160120':'20160319'],
                                                       dynamic = True)
# Intervalo de confiança
sarima_predict_conf_2_treino = sarima_predict_2_treino.conf_int()
sarima_predict_conf_2_treino
rcParams['figure.figsize'] = 20,8

# Plot dos valores observados
ax = df_train_series_Dia.plot(label = 'Valores Observados', color = '#2574BF')

# Plot dos valores previstos
sarima_predict_2_treino.predicted_mean.plot(ax = ax, 
                                     label = 'Previsões SARIMA(0, 0, 1)x(0, 1, 1, 7) com variável Exógena', 
                                     alpha = 0.7, 
                                     color = 'red') 

# Plot do intervalo de confiança
ax.fill_between(sarima_predict_conf_2_treino.index,
                # lower sales
                sarima_predict_conf_2_treino.iloc[:, 0],
                # upper sales
                sarima_predict_conf_2_treino.iloc[:, 1], color = 'k', alpha = 0.1)

# Títulos e Legendas
plt.title('Previsão de Consumo Energia com Modelo ARIMA Sazonal Dados Treino')
plt.xlabel('Data')
plt.ylabel('Media Consumo Energia')
plt.legend()
plt.show()
# Calculando a performance Dados Treino
sarima_results_2_treino = performance(df_train_series_Dia, sarima_predict_2_treino.predicted_mean)
sarima_results_2_treino
# Cria um dataframe a partir da série
Frame_means_teste = pd.DataFrame(df_test_series_Dia)

# Reset do índice para ajustar as colunas (podia ter feito tudo isso em um comando, ms didaticamente deixamos assim)
Frame_means_teste.reset_index(level = 0, inplace = True)

# Ajusta o nome das colunas
Frame_means_teste.columns = ['data', 'rv1']
Frame_means_teste.head()

# Aplicamos a função e criamos a coluna feriado
Frame_means_teste['feriado'] = Frame_means_teste.apply(adiciona_feriado, axis = 1)

# Convertendo a coluna feriado para inteiro
Frame_means_teste['feriado'] = pd.to_numeric(Frame_means_teste['feriado'], downcast = 'integer')

# Vamos definir a order_date como índice
Frame_means_teste.set_index("data", inplace = True)
# E somente o feriado (mais uma constante requerida pelo statsmodels) na série de feriado
exog_var_teste = sm.add_constant(Frame_means_teste['feriado'])
exog_var_teste
exog_var_teste.shape
df_test_series_Dia.shape
df_test_series_Dia.index.values.min()
df_test_series_Dia.index.values.max()
df_test_series_Dia
exog_var_teste
# Vamos fazer previsões um passo a frente
sarima_predict_2_teste = modelo_sarima_v2_fit.get_prediction(start = pd.to_datetime('2016-03-19'), 
                                                       end = pd.to_datetime('2016-05-28'),
                                                       exog = exog_var_teste, #['20160320':'20160527'],
                                                       dynamic = True)
# Intervalo de confiança
sarima_predict_conf_2_teste = sarima_predict_2_teste.conf_int()
sarima_predict_conf_2_teste
rcParams['figure.figsize'] = 20,8

# Plot dos valores observados
ax = df_full_series_Dia.plot(label = 'Valores Observados', color = '#2574BF')

# Plot dos valores previstos
sarima_predict_2_teste.predicted_mean.plot(ax = ax, 
                                     label = 'Previsões SARIMA(0, 0, 1)x(0, 1, 1, 7) com variável Exógena', 
                                     alpha = 0.7, 
                                     color = 'red') 

# Plot do intervalo de confiança
ax.fill_between(sarima_predict_conf_2_teste.index,
                # lower sales
                sarima_predict_conf_2_teste.iloc[:, 0],
                # upper sales
                sarima_predict_conf_2_teste.iloc[:, 1], color = 'k', alpha = 0.1)

# Títulos e Legendas
plt.title('Previsão de Consumo Energia com Modelo ARIMA Sazonal Dados Teste')
plt.xlabel('Data')
plt.ylabel('Media Consumo Energia')
plt.legend()
plt.show()
# Calculando a performance Dados Teste
sarima_results_2_teste = performance(df_test_series_Dia, sarima_predict_2_teste.predicted_mean)
sarima_results_2_teste
df_train_Dia_mean = df_train.resample('D').mean()
df_test_Dia_mean = df_test.resample('D').mean()
df_train_Dia_mean.describe()
df_train_Dia_mean_Stand = df_train_Dia_mean.copy()

# Colunas que desejo aplicar a Padronização
cols = ['appliances', 'lights', 't1', 'rh_1', 't2', 'rh_2', 't3', 'rh_3', 't4',
       'rh_4', 't5', 'rh_5', 't6', 'rh_6', 't7', 'rh_7', 't8', 'rh_8', 't9',
       'rh_9', 't_out', 'press_mm_hg', 'rh_out', 'windspeed', 'visibility',
       'tdewpoint', 'nsm']

for i in cols:
    scale = StandardScaler().fit(df_train_Dia_mean[[i]])
    
    df_train_Dia_mean_Stand[i] = scale.transform(df_train_Dia_mean[[i]])

df_train_Dia_mean_Stand.head(2)
df_test_Dia_mean_Stand = df_test_Dia_mean.copy()

# Colunas que desejo aplicar a Padronização
cols = ['appliances', 'lights', 't1', 'rh_1', 't2', 'rh_2', 't3', 'rh_3', 't4',
       'rh_4', 't5', 'rh_5', 't6', 'rh_6', 't7', 'rh_7', 't8', 'rh_8', 't9',
       'rh_9', 't_out', 'press_mm_hg', 'rh_out', 'windspeed', 'visibility',
       'tdewpoint', 'nsm']

for i in cols:
    scale = StandardScaler().fit(df_test_Dia_mean[[i]])
    
    df_test_Dia_mean_Stand[i] = scale.transform(df_test_Dia_mean[[i]])

df_test_Dia_mean_Stand.head(2)
model = VAR(endog = df_train_Dia_mean_Stand,
            freq = df_train_Dia_mean_Stand.index.inferred_freq)

model_fit = model.fit()
# model_fit.plot_forecast(3);
df_test_Dia_mean_Stand.values.shape
pred_var = model_fit.forecast(y=df_test_Dia_mean_Stand.values, steps=70)
pred_var
model_fit.summary()
pred = pd.DataFrame(index=range(0,len(pred_var)),columns=[df_train_Dia_mean_Stand.columns])
pred

#len(pred_var)

for j in range(0,len(df_train_Dia_mean_Stand.columns)):
    for i in range(0, len(pred_var)):
        pred.iloc[i][j] = pred_var[i][j]
pred_var
df_forecast = pd.DataFrame(pred_var, index=df_test_Dia_mean_Stand.index, columns=df_test_Dia_mean_Stand.columns + '_2d')
fig, ax = plt.subplots(figsize = (10,10))
ax.plot(df_test_Dia_mean_Stand['rv1'])
ax.plot(df_forecast['rv1_2d']);
performance(df_forecast['rv1_2d'], df_test_Dia_mean_Stand['rv1'])
#X = df_full.loc[:, df_full.columns != column_target]
#y = df_full.loc[:, df_full.columns == column_target]

X_train = df_train_Dia_mean_Stand.loc[:, df_train_Dia_mean_Stand.columns != column_target]
Y_train = df_train_Dia_mean_Stand.loc[:, df_train_Dia_mean_Stand.columns == column_target]

X_test = df_test_Dia_mean_Stand.loc[:, df_test_Dia_mean_Stand.columns != column_target]
Y_test = df_test_Dia_mean_Stand.loc[:, df_test_Dia_mean_Stand.columns == column_target]
X_train.shape
#Y_train.shape
X_test.shape
#Y_test.shape
# GRID SEARCH para identificar os melhores parametros.

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# Grid de parâmetros
param_grid = {'learning_rate': [0.1, 0.01, 0.001],
              'max_depth': [4, 5, 6],
              'min_samples_leaf': [3, 4, 5],
              'subsample': [0.3, 0.5, 0.7],
              'n_estimators': [400, 700, 1000, 2000, 3000]
              }

# Regressor
est = GradientBoostingRegressor()

# Modelo criado com GridSearchCV
gs_cv = GridSearchCV(est, param_grid, scoring = 'neg_mean_squared_error', n_jobs = 4, return_train_score=True).fit(X_train, Y_train)

# Imprime os melhors parâmetros
print('Melhores Hiperparâmetros: %r' % gs_cv.best_params_)
gs_cv.best_params_
#est = GradientBoostingRegressor()

#params = {'min_samples_leaf': 3}
#est.set_params(**gs_cv.best_params_)
#est.fit(X_train, Y_train)
est = GradientBoostingRegressor(n_estimators = 6000, max_depth =8, learning_rate = 0.001, min_samples_leaf=4, subsample=0.3)
est.fit(X_train, Y_train)
yhat = est.predict(X_test)
yhat
predy = pd.DataFrame(index=range(0,len(yhat)), columns=['rv1_pred'])

j = 0
for i in range(0, len(yhat)):
    predy.iloc[i][j] = yhat[i]
predy
predy.index = Y_test.index
fig, ax = plt.subplots(figsize = (20,8))
ax.plot(df_full_series_Dia, color = "green", label = 'Valores Observados')
ax.plot(pd.Series(predy['rv1_pred']), color = 'red', label = 'Valores Previstos Gradient Boosting')
plt.legend();
performance(pd.Series(predy['rv1_pred']), pd.Series(Y_test['rv1']))
X_train = df_train_Dia_mean_Stand.loc[:, df_train_Dia_mean_Stand.columns != column_target]
Y_train = df_train_Dia_mean_Stand.loc[:, df_train_Dia_mean_Stand.columns == column_target]

X_test = df_test_Dia_mean_Stand.loc[:, df_test_Dia_mean_Stand.columns != column_target]
Y_test = df_test_Dia_mean_Stand.loc[:, df_test_Dia_mean_Stand.columns == column_target]
modelo = LinearRegression()
modelo_result = modelo.fit(X_train, Y_train)
yhat2 = modelo_result.predict(X_test)
# output list  
# function used for removing nested  
# lists in python.  
def reemovNestings(yhat2): 
    
    for i in yhat2: 
        if type(i) == list: 
            reemovNestings(i) 
        else: 
            output.append(i) 
    
    return output
output = []
yhat2 = reemovNestings(yhat2.tolist())
yhat2
predy2 = pd.DataFrame(index=range(0,len(yhat2)), columns=['rv1_pred'])

j = 0
for i in range(0, len(yhat2)):
    predy2.iloc[i][j] = yhat2[i]
predy2.index = Y_test.index
predy2
fig, ax = plt.subplots(figsize = (20,8))
ax.plot(df_full_series_Dia, color = "green", label = 'Valores Observados')
ax.plot(pd.Series(predy2['rv1_pred']), color = 'red', label = 'Valores Previstos Gradient Boosting')
plt.legend();
performance(pd.Series(predy2['rv1_pred']), pd.Series(Y_test['rv1']))