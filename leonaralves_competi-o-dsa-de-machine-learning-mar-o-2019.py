# Leitura de Pacotes

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from string import ascii_letters

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

from sklearn.preprocessing import Imputer, MinMaxScaler

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn import model_selection

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.svm import SVR

from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb

from scipy import stats

import statsmodels.api as sm 

from scipy.stats import uniform, randint

from pandas.plotting import scatter_matrix

from scipy.stats import norm, skew

from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
# Lendo as bases de dados

dataset_treino = pd.read_csv("../input/dataset_treino.csv")

dataset_teste = pd.read_csv("../input/dataset_teste.csv")

dataset_lojas = pd.read_csv("../input/lojas.csv")
# Visualização dos dados de treino

dataset_treino.head(5)
# Visualização dos dados das lojas

dataset_lojas.head(5)
# Juntando o conjunto de dados de treino e as lojas

dataset_treino = pd.merge(dataset_treino, dataset_lojas, on = "Store")



# Juntando o conjunto de dados de teste e as lojas

dataset_teste = pd.merge(dataset_teste, dataset_lojas, on = "Store")



dataset_treino.head(5)
# Shape do Dado

dataset_treino.shape
# Tipos de Dados

print(dataset_treino.dtypes)
# Mudando as variaveis que estão em categorias de letras para categorias numericas



# StoreType

label_encoder = LabelEncoder().fit(dataset_treino['StoreType'])

dataset_treino['StoreType'] = label_encoder.transform(dataset_treino['StoreType'])



dataset_teste['StoreType'] = label_encoder.transform(dataset_teste['StoreType'])



# Assortment

label_encoder = LabelEncoder().fit(dataset_treino['Assortment'])

dataset_treino['Assortment'] = label_encoder.transform(dataset_treino['Assortment'])



dataset_teste['Assortment'] = label_encoder.transform(dataset_teste['Assortment'])



# StateHoliday

dataset_treino['StateHoliday'] = np.array(dataset_treino['StateHoliday'], dtype = str)

label_encoder = LabelEncoder().fit(dataset_treino['StateHoliday'])

dataset_treino['StateHoliday'] = label_encoder.transform(dataset_treino['StateHoliday'])



dataset_teste['StateHoliday'] = label_encoder.transform(dataset_teste['StateHoliday'])
# Pegando informacoes da variavel 'Date'



# Pegando somente o dia da variavel 'Date'

dataset_treino["Dia"] = dataset_treino["Date"].str.split("-", n = 2, expand = True)[2]

dataset_treino['Dia'] = dataset_treino['Dia'].astype(int) 



dataset_teste["Dia"] = dataset_teste["Date"].str.split("-", n = 2, expand = True)[2]

dataset_teste['Dia'] = dataset_teste['Dia'].astype(int) 



# Pegando somente o mes da variavel 'Date'

dataset_treino["Mes"] = dataset_treino["Date"].str.split("-", n = 2, expand = True)[1]

dataset_treino['Mes'] = dataset_treino['Mes'].astype(int) 



dataset_teste["Mes"] = dataset_teste["Date"].str.split("-", n = 2, expand = True)[1]

dataset_teste['Mes'] = dataset_teste['Mes'].astype(int) 



# Pegando somente o ano da variavel 'Date'

dataset_treino["Ano"] = dataset_treino["Date"].str.split("-", n = 1, expand = True)[0]

dataset_treino['Ano'] = dataset_treino['Ano'].astype(int) 



dataset_teste["Ano"] = dataset_teste["Date"].str.split("-", n = 1, expand = True)[0]

dataset_teste['Ano'] = dataset_teste['Ano'].astype(int) 
# Separando os meses de Intervalo de Promocao

dataset_treino[['mes1', 'mes2', 'mes3', 'mes4']] = dataset_treino["PromoInterval"].str.split(",", n = 3, expand = True)



dataset_teste[['mes1', 'mes2', 'mes3', 'mes4']] = dataset_teste["PromoInterval"].str.split(",", n = 3, expand = True)
# Transformando os meses em valores numeros

transf_num = {np.nan: '0', 'Jan': 1, 'Feb':2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 

              'Jul': 7, 'Aug': 8, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

dataset_treino['mes1'] = dataset_treino['mes1'].map(transf_num)

dataset_treino['mes2'] = dataset_treino['mes2'].map(transf_num)

dataset_treino['mes3'] = dataset_treino['mes3'].map(transf_num)

dataset_treino['mes4'] = dataset_treino['mes4'].map(transf_num)



dataset_teste['mes1'] = dataset_teste['mes1'].map(transf_num)

dataset_teste['mes2'] = dataset_teste['mes2'].map(transf_num)

dataset_teste['mes3'] = dataset_teste['mes3'].map(transf_num)

dataset_teste['mes4'] = dataset_teste['mes4'].map(transf_num)
# Criando a variavel intervalo de promocao

dataset_treino['Intervalo_Promo'] = 0



dataset_teste['Intervalo_Promo'] = 0
# Verificando se a loja está em intervalo de promocao

dataset_treino.Intervalo_Promo[((dataset_treino['Mes'] == dataset_treino['mes1'])\

                                | (dataset_treino['Mes'] == dataset_treino['mes2'])\

                                | (dataset_treino['Mes'] == dataset_treino['mes3']) \

                                | (dataset_treino['Mes'] == dataset_treino['mes4']))] = 1



dataset_teste.Intervalo_Promo[((dataset_teste['Mes'] == dataset_teste['mes1'])\

                                | (dataset_teste['Mes'] == dataset_teste['mes2'])\

                                | (dataset_teste['Mes'] == dataset_teste['mes3']) \

                                | (dataset_teste['Mes'] == dataset_teste['mes4']))] = 1
# Pegando as semanas do ano

dataset_treino['Date'] = pd.to_datetime(dataset_treino['Date'], errors ='coerce')

dataset_treino['WeekOfYear'] = dataset_treino['Date'].dt.weekofyear



dataset_teste['Date'] = pd.to_datetime(dataset_teste['Date'], errors ='coerce')

dataset_teste['WeekOfYear'] = dataset_teste['Date'].dt.weekofyear
# Quanto tempo o concorrente abriu em anos

dataset_treino['Concorrente_Open'] = dataset_treino['Ano'] - dataset_treino['CompetitionOpenSinceYear']



dataset_teste['Concorrente_Open'] = dataset_teste['Ano'] - dataset_teste['CompetitionOpenSinceYear']
# Quanto tempo a promo2 abriu em anos

dataset_treino['Promo2_Open'] = dataset_treino['Ano'] - dataset_treino['Promo2SinceYear']



dataset_teste['Promo2_Open'] = dataset_teste['Ano'] - dataset_teste['Promo2SinceYear']
# Quanto tempo o concorrente abriu em meses

dataset_treino['Concorrente_Open_Mes'] = 12*(dataset_treino['Ano'] - dataset_treino['CompetitionOpenSinceYear'])- (dataset_treino['Mes'] - dataset_treino['CompetitionOpenSinceMonth'])/4.0



dataset_teste['Concorrente_Open_Mes'] = 12*(dataset_teste['Ano'] - dataset_teste['CompetitionOpenSinceYear']) - (dataset_teste['Mes'] - dataset_teste['CompetitionOpenSinceMonth'])/4.0
# Quanto tempo a promo2 abriu em meses

dataset_treino['Promo2_Open_Mes'] = 12*(dataset_treino['Ano'] - dataset_treino['Promo2SinceYear'])- (dataset_treino['Mes'] - dataset_treino['Promo2SinceWeek'])/4.0



dataset_teste['Promo2_Open_Mes'] = 12*(dataset_teste['Ano'] - dataset_teste['Promo2SinceYear'])- (dataset_teste['Mes'] - dataset_teste['Promo2SinceWeek'])/4.0
# Retirando as lojas que não tiveram vendas (não abriram)

dataset_treino = dataset_treino.drop(dataset_treino[dataset_treino.Sales == 0].index)
# Vendo se existem valores vazios na variavel Open nos dados de treino

dataset_treino['Open'].isnull().sum()
# Vendo se existem valores nulos na variavel Open nos dados de teste

dataset_teste['Open'].isnull().sum()
# Para os valores NA, colocando valores 1

dataset_teste.Open[dataset_teste['Open'].isnull()] = 1
# Separando os dados de teste em relação a variavel Open

sep_teste_notopen = dataset_teste[dataset_teste['Open'] == 0]

dataset_teste = dataset_teste[dataset_teste['Open'] == 1]
# Retirando a variável Open dos dados de treino e de teste

dataset_treino = dataset_treino.drop(columns = 'Open')



dataset_teste = dataset_teste.drop(columns = 'Open')
# Lojas que tiveram promoção

dataset_treino.groupby('Promo').size()
# Distribuição das lojas

dataset_treino.groupby('Store').size().head(10)
# Função para calcular valores faltantes

def missing_values_table(df):

        # Total de valores faltantes

        mis_val = df.isnull().sum()

        

        # Percentagem de valores faltantes

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Fazendo uma tabela com os resultados

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis = 1)

        

        # Renomeando as colunas

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Valores Missing', 1 : '% of Valores Totais'})

        

        # Ordenado a tabela pela percentagem de valores faltantes

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Valores Totais', ascending=False).round(1)

        

        # Print summary information

        print ("Este dataframe tem " + str(df.shape[1]) + " colunas.\n"      

            "Existem " + str(mis_val_table_ren_columns.shape[0]) +

              " colunas que tem valores faltantes.")

        

        # Returnar o dataframe com a informação dos valores missing

        return mis_val_table_ren_columns
# Calculando a % de valores faltantes

missing_values_table(dataset_treino)
# Histograma da Variavel 'Sales'

plt.hist(dataset_treino['Sales'], bins = 50, edgecolor = 'k');

plt.xlabel('Sales');  

plt.title('Distribuição das Vendas');
# Comparacao do Histograma da Variavel 'Sales'

fig = plt.figure(figsize = (10,5))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)

graf1 = sns.distplot(dataset_treino['Sales'], hist = True, label='skewness:{:.2f}'.format(dataset_treino['Sales'].skew()),ax = ax1)

graf1.legend()

graf1.set(xlabel = 'Sales', ylabel = 'Densidade', title = 'Distribuicao Sales')

graf2 = sns.distplot(np.log1p(dataset_treino['Sales']), hist = True, label = 'skewness:{:.2f}'.format(np.log1p(dataset_treino['Sales']).skew()),ax=ax2)

graf2.legend()

graf2.set(xlabel = 'log(Sales+1)',ylabel = 'Densidade', title = 'Distribuicao log(Sales+1)')

plt.show()
# Boxplot

variaveis_num = ['Sales', 'Customers', 'CompetitionDistance']

sns.boxplot(data = dataset_treino[variaveis_num], orient = "h")

plt.show()
# Gráfico de Correlação

sns.set(style = "white")



# Compute the correlation matrix

corr = dataset_treino.corr(method = 'pearson')



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize = (11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap = True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask = mask, cmap = cmap, vmax = .3, center = 0,

            square = True, linewidths = .5, cbar_kws={"shrink": .5})



plt.show()
# Grafico dos Customers x Sales

fig, ax = plt.subplots()

ax.scatter(dataset_treino['Customers'], dataset_treino['Sales'], edgecolors = (0, 0, 0))

ax.set_xlabel('Customers')

ax.set_ylabel('Sales')

plt.show()
# Vendo como estão as vendas da Store1

dataset_treino.loc[dataset_treino['Store'] == 1, ['Date','Sales']].plot(x = 'Date', y = 'Sales', title ='Store 1',

                                                                        figsize = (16,4), color = 'purple')

plt.show()
# Imputando os Dados Faltantes da variavel CompetitionDistance

mediana = dataset_treino['CompetitionDistance'].median()

d = {np.nan: mediana}

dataset_treino['CompetitionDistance'] = dataset_treino['CompetitionDistance'].replace(d).astype(int)



dataset_teste['CompetitionDistance'] = dataset_teste['CompetitionDistance'].replace(d).astype(int)
# Mudando os Na para 0

d = {np.nan: 0}

dataset_treino['CompetitionOpenSinceMonth'] = dataset_treino['CompetitionOpenSinceMonth'].replace(d).astype(int)

dataset_treino['CompetitionOpenSinceYear'] = dataset_treino['CompetitionOpenSinceYear'].replace(d).astype(int)



dataset_teste['CompetitionOpenSinceMonth'] = dataset_teste['CompetitionOpenSinceMonth'].replace(d).astype(int)

dataset_teste['CompetitionOpenSinceYear'] = dataset_teste['CompetitionOpenSinceYear'].replace(d).astype(int)



dataset_treino['Promo2SinceWeek'] = dataset_treino['Promo2SinceWeek'].replace(d).astype(int)

dataset_treino['Promo2SinceYear'] = dataset_treino['Promo2SinceYear'].replace(d).astype(int)



dataset_teste['Promo2SinceWeek'] = dataset_teste['Promo2SinceWeek'].replace(d).astype(int)

dataset_teste['Promo2SinceYear'] = dataset_teste['Promo2SinceYear'].replace(d).astype(int)
# Mudando os Na para 0

d = {np.nan: 0}

dataset_treino['Promo2_Open'] = dataset_treino['Promo2_Open'].replace(d).astype(int)

dataset_treino['Concorrente_Open'] = dataset_treino['Concorrente_Open'].replace(d).astype(int)



dataset_teste['Promo2_Open'] = dataset_teste['Promo2_Open'].replace(d).astype(int)

dataset_teste['Concorrente_Open'] = dataset_teste['Concorrente_Open'].replace(d).astype(int)



dataset_treino['Promo2_Open_Mes'] = dataset_treino['Promo2_Open_Mes'].replace(d).astype(int)

dataset_treino['Concorrente_Open_Mes'] = dataset_treino['Concorrente_Open_Mes'].replace(d).astype(int)



dataset_teste['Promo2_Open_Mes'] = dataset_teste['Promo2_Open_Mes'].replace(d).astype(int)

dataset_teste['Concorrente_Open_Mes'] = dataset_teste['Concorrente_Open_Mes'].replace(d).astype(int)
# Transformando os resultados em arrays

valores_storetype = np.array(dataset_treino['StoreType'])

valores_assortment = np.array(dataset_treino['Assortment'])

valores_state = np.array(dataset_treino['StateHoliday'])



valores_storetype_teste = np.array(dataset_teste['StoreType'])

valores_assortment_teste = np.array(dataset_teste['Assortment'])

valores_state_teste = np.array(dataset_teste['StateHoliday'])
# Reshape nos dados

inteiros_storetype = valores_storetype.reshape(len(valores_storetype),1)

inteiros_assortment = valores_assortment.reshape(len(valores_assortment),1)

inteiros_state = valores_state.reshape(len(valores_state),1)



inteiros_storetype_teste = valores_storetype_teste.reshape(len(valores_storetype_teste),1)

inteiros_assortment_teste = valores_assortment_teste.reshape(len(valores_assortment_teste),1)

inteiros_state_teste = valores_state_teste.reshape(len(valores_state_teste),1)
# Criando um objeto do tipo 'OneHotEnconder'

onehot_encoder1 = OneHotEncoder(sparse = False).fit(inteiros_storetype)

onehot_encoder2 = OneHotEncoder(sparse = False).fit(inteiros_assortment)

onehot_encoder3 = OneHotEncoder(sparse = False).fit(inteiros_state)



# Transformação nos dados de treino

vetores_binarios1 = onehot_encoder1.transform(inteiros_storetype)

vetores_binarios2 = onehot_encoder2.transform(inteiros_assortment)

vetores_binarios3 = onehot_encoder3.transform(inteiros_state)



# Transformação nos dados de teste

vetores_binarios1_teste = onehot_encoder1.transform(inteiros_storetype_teste)

vetores_binarios2_teste = onehot_encoder2.transform(inteiros_assortment_teste)

vetores_binarios3_teste = onehot_encoder3.transform(inteiros_state_teste)
# Concatenando os resultados

vetores_binarios = pd.concat([pd.DataFrame(vetores_binarios1, 

                                           columns = ['store0', 'store1', 'store2', 'store3']), 

                              pd.DataFrame(vetores_binarios2, 

                                           columns = ['assort0', 'assort1', 'assort2']), 

                              pd.DataFrame(vetores_binarios3, 

                                           columns = ['state0', 'state1', 'state2', 'state3'])], 

                             axis = 1)



vetores_binarios_teste = pd.concat([pd.DataFrame(vetores_binarios1_teste, 

                                                 columns = ['store0', 'store1', 'store2', 'store3']), 

                                  pd.DataFrame(vetores_binarios2_teste, 

                                               columns = ['assort0', 'assort1', 'assort2']), 

                                  pd.DataFrame(vetores_binarios3_teste, 

                                               columns = ['state0', 'state1', 'state2', 'state3'])], 

                                   axis = 1)



vetores_binarios.head(5)
# Alinhando os indices

vetores_binarios.index = dataset_treino.index



vetores_binarios_teste.index = dataset_teste.index
# Importância do Atributo com o Extra Trees Regressor



# Separando o array em componentes de input e output

colunas_choice =   ['Promo', 'Store', 'Mes', 'Ano', 'Dia', 'DayOfWeek', 

                    'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth',

                    'SchoolHoliday',  'CompetitionDistance',  'Promo2', 

                    'Promo2SinceWeek', 'Promo2SinceYear', 'Intervalo_Promo',

                    'WeekOfYear', 'Concorrente_Open', 'Promo2_Open', 

                    'Concorrente_Open_Mes', 'Promo2_Open_Mes', 

                    'Sales']



#StoreTypeAssortmentStateHoliday

df = dataset_treino[colunas_choice]

array = df.values



# Separando o array em componentes de input e output

X = array[:,0:(len(colunas_choice)-1)]

Y = array[:,(len(colunas_choice)-1)]



# Criação do Modelo - Feature Selection

modelo_feat = ExtraTreesRegressor()

modelo_feat.fit(X, Y)
# Plotando a importancia das variaveis

feat_importances = pd.Series(modelo_feat.feature_importances_, index = colunas_choice[:-1])

feat_importances.sort_values(ascending = True).plot(kind = 'barh')

plt.title('Importancia dos Atributos')

plt.show()
# Juntando os resultados de transformacao dos dados 

colunas_importance =  ['CompetitionDistance', 'Store',  'Promo', 

                       'CompetitionOpenSinceMonth',

                      'DayOfWeek','CompetitionOpenSinceYear', 

                      'Dia', 'Promo2SinceYear','WeekOfYear',

                      'Concorrente_Open_Mes','Promo2SinceWeek',

                      'Mes', 'Concorrente_Open', 'Promo2_Open_Mes','Ano']

data_modelo = dataset_treino[colunas_importance].copy()

data_modelo = data_modelo.join(vetores_binarios)



data_modelo_teste = dataset_teste[colunas_importance].copy()

data_modelo_teste = data_modelo_teste.join(vetores_binarios_teste)
# Separando em X e Y

X_treino = data_modelo.values

Y_treino = dataset_treino['Sales']
#def rmspe(predictions, targets):

#    return np.sqrt((((targets - predictions)/targets) ** 2).mean())
# Criando modelo de Machine Learning a partir de cada algoritmo



#modelos = []

#modelos.append(('LR', LinearRegression()))

#modelos.append(('LASSO', Lasso()))

#modelos.append(('EN', ElasticNet()))

#modelos.append(('Ridge', Ridge()))

#modelos.append(('KNN', KNeighborsRegressor()))

#modelos.append(('CART', DecisionTreeRegressor()))

#modelos.append(('SVR', SVR(gamma = 'auto')))

#modelos.append(('AB', AdaBoostRegressor(n_estimators = 100)))

#modelos.append(('GBM', GradientBoostingRegressor(n_estimators = 100)))

#modelos.append(('RF', RandomForestRegressor(n_estimators = 100)))

#modelos.append(('ET', ExtraTreesRegressor(n_estimators = 100)))

#modelos.append(('XG', XGBRegressor()))



#resultados = []

#nomes = []



# Percorrendo cada um dos modelos

#for nome, modelo in modelos:

#    kfold = model_selection.KFold(10, True, random_state = 42)

#    previsoes = cross_val_predict(modelo, X_treino, np.log1p(Y_treino), cv = kfold)

#    metrica = rmspe(np.expm1(previsoes), np.expm1(Y_treino))

#    resultados.append(previsoes)

#    nomes.append(nome)

#    texto = "%s: %f" % (nome, metrica)

#    print(texto)

# 0.18079538227473296

# 0.17492211856266057
# Graficos das previsoes

#fig, ax = plt.subplots()

#ax.scatter(np.expm1(Y_treino), np.expm1(previsoes), edgecolors = (0, 0, 0))

#ax.plot([Y_treino.min(), Y_treino.max()], [Y_treino.min(), Y_treino.max()], 'k--', lw = 4)

#ax.set_xlabel('Observado')

#ax.set_ylabel('Previsto')

#plt.show()
# Funcoes para calculo do RMPSE



def rmspe(y, yhat):

    return np.sqrt(np.mean((yhat/y-1) ** 2))



def rmspe_xgboost(yhat, y):

    y = np.expm1(y.get_label())

    yhat = np.expm1(yhat)

    return "rmspe", rmspe(y,yhat)
# Melhores parametros

params = {"objective": "reg:linear",

          "booster" : "gbtree",

          "eta": 0.1,

          "max_depth": 10,

          "subsample": 0.9,

          "colsample_bytree": 0.5,

          "silent": 1}



num_boost_round = 10000



# Dividindo em dados de treino e validação

X_train, X_valid, y_train, y_valid = train_test_split(X_treino, Y_treino, test_size = 0.04)



# Treinando um modelo XGBoost

dtrain = xgb.DMatrix(X_train, np.log1p(y_train)) # dados de treino

dvalid = xgb.DMatrix(X_valid, np.log1p(y_valid)) # dados de validacao



watchlist = [(dtrain, 'train'), (dvalid, 'eval')]



modelo = xgb.train(params, dtrain, num_boost_round, evals = watchlist, \

  early_stopping_rounds = 100, feval = rmspe_xgboost, verbose_eval = True)
# Resultado nos dados de validacao

y_pred = modelo.predict(xgb.DMatrix(X_valid), ntree_limit = modelo.best_ntree_limit)

erro = rmspe(y_valid.values, np.expm1(y_pred))

print('O RMSPE é : {:.5f}'.format(erro))
# Importancia das variaveis no modelo

fig, ax = plt.subplots(figsize = (10, 5))

xgb.plot_importance(modelo, ax = ax)

plt.show()
# Graficos das previsoes

fig, ax = plt.subplots()

ax.scatter(y_valid.values, np.expm1(y_pred), edgecolors = (0, 0, 0))

ax.plot([y_valid.values.min(), y_valid.values.max()], [y_valid.values.min(), y_valid.values.max()], 'k--', lw = 4)

ax.set_xlabel('Observado')

ax.set_ylabel('Previsto')

plt.show()
# Prevendo para os dados em que a loja foi aberta

dtest = xgb.DMatrix(data_modelo_teste.values)

previsoes_teste = np.expm1(modelo.predict(dtest, ntree_limit = modelo.best_ntree_limit))
# Juntando os resultados das previsoes

Submissao = pd.DataFrame(dataset_teste['Id'])

Submissao['Sales'] = previsoes_teste
# Adicionando a variavel Sales com valores zeros para lojas que não abriram

sep_teste_notopen['Sales'] = 0

Submissao2 = sep_teste_notopen[['Id','Sales']]
# Juntando os dataframes

Submissao_final = pd.concat([Submissao, Submissao2])

Submissao_final.head(10)
# Ordenando os dados de submissao

Submissao_final = Submissao_final.sort_values(by = ['Id'])
# Salvando os resultados

Submissao_final.to_csv('Submission.csv', header = True, index = False)