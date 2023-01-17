# Leitura de Pacotes

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from string import ascii_letters

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_squared_error

from sklearn import model_selection

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor

from xgboost import XGBRegressor

from scipy import stats

from pandas import read_csv

from scipy.stats import uniform, randint

from pandas.plotting import scatter_matrix

from scipy.stats import norm, skew

from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
# Lendo a base de dados

dataset_treino = pd.read_csv("../input/dataset_treino.csv", index_col = "Order")

dataset_teste = pd.read_csv("../input/dataset_teste.csv", index_col = "OrderId")
# Visualização dos dados

dataset_treino.head(5)
# Shape do Dado

dataset_treino.shape
# Tipos de Dados

print(dataset_treino.dtypes)
# Vendo as Colunas dos Dados

dataset_treino.columns
# Retirando colunas que não serão utilizadas

dataset_treino = dataset_treino.drop(["Property Id","Property Name","Parent Property Id",

                                      "Parent Property Name", "NYC Borough, Block and Lot (BBL) self-reported", 

                                      "NYC Building Identification Number (BIN)","Postal Code",

                                      "Street Number","Street Name",

                                      "Borough", "Address 1 (self-reported)", 'Address 2'],

                                      axis = 1)
# Retirando as Variaveis Categoricas com muitos NA

dataset_treino = dataset_treino.drop(["2nd Largest Property Use Type",

                                      "2nd Largest Property Use - Gross Floor Area (ft²)","3rd Largest Property Use Type",

                                      "3rd Largest Property Use Type - Gross Floor Area (ft²)", "Fuel Oil #1 Use (kBtu)",

                                      "Fuel Oil #2 Use (kBtu)", "Fuel Oil #4 Use (kBtu)", "Fuel Oil #5 & 6 Use (kBtu)",

                                      "Diesel #2 Use (kBtu)", "District Steam Use (kBtu)", "DOF Benchmarking Submission Status"], axis = 1)
# Retirando colunas duplicas

dataset_treino = dataset_treino.drop(["Largest Property Use Type","List of All Property Use Types at Property"], axis = 1)
# Criando uma nova coluna com os "Borough"

dataset_treino['BBL - 10 digits'] = dataset_treino['BBL - 10 digits'].astype('category')

dataset_treino['BBL - 10 digits'] = dataset_treino['BBL - 10 digits'].apply(lambda x: x[0]).astype('category')



dataset_teste['BBL - 10 digits'] = dataset_teste['BBL - 10 digits'].astype('category')

dataset_teste['BBL - 10 digits'] = dataset_teste['BBL - 10 digits'].apply(lambda x: x[0]).astype('category')
# Substituindo os Valores Faltantes por 0.001

# Dados de Treino

dataset_treino = dataset_treino.where((pd.notnull(dataset_treino)), 0.001)

d = {'Not Available': 0.001}

df = dataset_treino.replace(d)



# Dados de Teste

dataset_teste = dataset_teste.where((pd.notnull(dataset_teste)), 0.001)

df_teste = dataset_teste.replace(d)
# Colunas Numericas com Valores Faltantes

colunas = ["Year Built", "Site EUI (kBtu/ft²)","Weather Normalized Site EUI (kBtu/ft²)",

           "Weather Normalized Site Electricity Intensity (kWh/ft²)","Weather Normalized Site Natural Gas Intensity (therms/ft²)",

           "Natural Gas Use (kBtu)","Weather Normalized Site Electricity (kWh)","Total GHG Emissions (Metric Tons CO2e)",

           "Direct GHG Emissions (Metric Tons CO2e)","Indirect GHG Emissions (Metric Tons CO2e)",

           "Water Use (All Water Sources) (kgal)","Water Intensity (All Water Sources) (gal/ft²)","Source EUI (kBtu/ft²)",

          "Weather Normalized Site Natural Gas Use (therms)", "Electricity Use - Grid Purchase (kBtu)",

          "Natural Gas Use (kBtu)", "Weather Normalized Site Natural Gas Use (therms)","Weather Normalized Source EUI (kBtu/ft²)",

          "DOF Gross Floor Area","Number of Buildings - Self-reported", "Census Tract",

           "Largest Property Use Type - Gross Floor Area (ft²)","Property GFA - Self-Reported (ft²)","Occupancy",

           "Council District","Community Board"]

array = df[colunas].values

                                      

array_teste = df_teste[colunas].values
# Transformando o array em float

array = array.astype(float)



array_teste = array_teste.astype(float)
# Imputando os Dados Faltantes pela Media dos Dados

imp = SimpleImputer(missing_values = 0.001, strategy = 'median').fit(array)  

dataset_treino[colunas] = imp.transform(array)



dataset_teste[colunas] = imp.transform(array_teste)
# Visualizando Novamente os Dados

dataset_treino.head(5)
# Descrição dos dados

dataset_treino.describe()
# Correlação

dataset_treino.corr(method = 'pearson')
# Retirando as variaveis sem correlacao com a variavel resposta

dataset_treino = dataset_treino.drop(["DOF Gross Floor Area", "Largest Property Use Type - Gross Floor Area (ft²)",

                                      "Weather Normalized Site EUI (kBtu/ft²)","Number of Buildings - Self-reported",

                                     "Occupancy","Electricity Use - Grid Purchase (kBtu)",

                                     "Weather Normalized Site Electricity (kWh)","Indirect GHG Emissions (Metric Tons CO2e)",

                                     'Property GFA - Self-Reported (ft²)',"Water Use (All Water Sources) (kgal)",

                                     "Water Intensity (All Water Sources) (gal/ft²)","Latitude","Longitude",

                                     "Community Board", "Council District","Census Tract"],

                                      axis = 1)
# Verificando a Multicolinearidade

colunas_num = ["ENERGY STAR Score", "Site EUI (kBtu/ft²)", "Year Built",

               "Weather Normalized Site Electricity Intensity (kWh/ft²)", "Weather Normalized Site Natural Gas Intensity (therms/ft²)",

               "Natural Gas Use (kBtu)", "Weather Normalized Site Natural Gas Use (therms)", "Total GHG Emissions (Metric Tons CO2e)",

               "Direct GHG Emissions (Metric Tons CO2e)", "Source EUI (kBtu/ft²)", "Weather Normalized Source EUI (kBtu/ft²)"]





vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(dataset_treino[colunas_num].values, i) for i in range(dataset_treino[colunas_num].shape[1])]

vif["features"] = dataset_treino[colunas_num].columns

vif
# Retirando Variaveis Multicolineares (VIF>10)

dataset_treino = dataset_treino.drop(["Source EUI (kBtu/ft²)", "Weather Normalized Source EUI (kBtu/ft²)","Natural Gas Use (kBtu)",

                                     "Year Built","Direct GHG Emissions (Metric Tons CO2e)"],

                                      axis = 1)
# Conferindo o VIF das Variaveis Restantes

colunas_num = ["ENERGY STAR Score", "Site EUI (kBtu/ft²)", 

               "Weather Normalized Site Electricity Intensity (kWh/ft²)", "Weather Normalized Site Natural Gas Intensity (therms/ft²)",

               "Weather Normalized Site Natural Gas Use (therms)", "Total GHG Emissions (Metric Tons CO2e)"]



vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(dataset_treino[colunas_num].values, i) for i in range(dataset_treino[colunas_num].shape[1])]

vif["features"] = dataset_treino[colunas_num].columns

vif
# Refazendo a Correlação

dataset_treino.corr(method = 'pearson')
# Gráfico de Correlação

sns.set(style = "white")



# Compute the correlation matrix

corr = dataset_treino.corr(method = 'pearson')



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap = True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
# Boxplot

sns.boxplot(dataset_treino["ENERGY STAR Score"])

plt.show()
# Distplot

sns.distplot(tuple(dataset_treino["ENERGY STAR Score"]), fit = stats.norm)

plt.show()
# Boxplot das Variaveis Restantes (ha outliers)

sns.boxplot(data = dataset_treino[["ENERGY STAR Score", "Site EUI (kBtu/ft²)", 

               "Weather Normalized Site Electricity Intensity (kWh/ft²)", "Weather Normalized Site Natural Gas Intensity (therms/ft²)",

               "Weather Normalized Site Natural Gas Use (therms)", "Total GHG Emissions (Metric Tons CO2e)"]],

            orient = "h")

plt.show()
# Retirando os outliers

coluna_out = "Weather Normalized Site Natural Gas Use (therms)"

mean = np.mean(dataset_treino[coluna_out])

sd = np.std(dataset_treino[coluna_out])

out = mean + 2*sd

dataset_treino = dataset_treino.drop(dataset_treino[dataset_treino[coluna_out] > out].index)



coluna_out = "Weather Normalized Site Natural Gas Intensity (therms/ft²)"

mean = np.mean(dataset_treino[coluna_out])

sd = np.std(dataset_treino[coluna_out])

out = mean + 2*sd

dataset_treino = dataset_treino.drop(dataset_treino[dataset_treino[coluna_out] > out].index)



coluna_out = "Weather Normalized Site Electricity Intensity (kWh/ft²)"

mean = np.mean(dataset_treino[coluna_out])

sd = np.std(dataset_treino[coluna_out])

out = mean + 2*sd

dataset_treino = dataset_treino.drop(dataset_treino[dataset_treino[coluna_out] > out].index)



coluna_out = "Site EUI (kBtu/ft²)"

mean = np.mean(dataset_treino[coluna_out])

sd = np.std(dataset_treino[coluna_out])

out = mean + 2*sd

dataset_treino = dataset_treino.drop(dataset_treino[dataset_treino[coluna_out] > out].index)



coluna_out = "Total GHG Emissions (Metric Tons CO2e)"

mean = np.mean(dataset_treino[coluna_out])

sd = np.std(dataset_treino[coluna_out])

out = mean + 2*sd

dataset_treino = dataset_treino.drop(dataset_treino[dataset_treino[coluna_out] > out].index)
# Transformando as variaveis Categoricas restantes em Dummies

coluna_cat = ["Primary Property Type - Self Selected",  "Metered Areas (Energy)", 'BBL - 10 digits','Water Required?']

dummies = pd.get_dummies(dataset_treino[coluna_cat])

dummies_teste = pd.get_dummies(dataset_teste[coluna_cat])



final_train, final_test = dummies.align(dummies_teste,join='left',axis=1)
# Separando as Colunas Categoricas das Numericas

coluna_num = ["Site EUI (kBtu/ft²)", 

               "Weather Normalized Site Electricity Intensity (kWh/ft²)", "Weather Normalized Site Natural Gas Intensity (therms/ft²)",

               "Weather Normalized Site Natural Gas Use (therms)", "Total GHG Emissions (Metric Tons CO2e)"]



target = "ENERGY STAR Score"
# Transformando os dados para a mesma escala (entre 0 e 1)



X = dataset_treino[coluna_num].values

Y_target = dataset_treino[target].values



X_teste = dataset_teste[coluna_num].values



# Gerando a nova escala

scaler = MinMaxScaler(feature_range = (-1, 1)).fit(X)

rescaledX = scaler.transform(X)



rescaledX_teste = scaler.transform(X_teste)
# Juntando os resultados de transformacao dos dados 

data_modelo = dataset_treino[coluna_num].copy()

data_modelo[coluna_num] = rescaledX

data_modelo = data_modelo.join(final_train)



data_modelo_teste = dataset_teste[coluna_num].copy()

data_modelo_teste[coluna_num] = rescaledX_teste

data_modelo_teste = data_modelo_teste.join(final_test)
data_modelo_teste[np.isnan(data_modelo_teste)] = 0
X_treino = data_modelo.values

Y_treino = Y_target
# Criando modelo de Machine Learning a partir de cada algoritmo, ja ajustado os parametros por GridSearchCV

# Vamos utilizar como métrica o MAE (Mean Absolute Error). Valor igual a zero indica excelente nível de precisão.

modelos = []



modelos.append(('GBM', GradientBoostingRegressor(n_estimators = 270, loss = 'huber', max_depth = 4,

                                                 learning_rate = 0.05, subsample = 0.7)))

modelos.append(('XG', XGBRegressor(n_estimators = 300, max_depth = 4, learning_rate = 0.05,

                                   gamma = 0.3, subsample= 0.75, min_child_weight = 0.5)))

resultados = []

nomes = []



# Percorrendo cada um dos modelos

for nome, modelo in modelos:

    kfold = model_selection.KFold(50, True, random_state = 7)

    cross_val_result = model_selection.cross_val_score(modelo, 

                                                        X_treino, 

                                                        Y_treino, 

                                                        cv = kfold, scoring = 'neg_mean_absolute_error')

    resultados.append(cross_val_result)

    nomes.append(nome)

    texto = "%s: %f (%f)" % (nome, cross_val_result.mean(), cross_val_result.std())

    print(texto)
# Comparando os algoritmos

fig = plt.figure()

fig.suptitle('Comparando os Algoritmos')

ax = fig.add_subplot(111)

plt.boxplot(resultados)

ax.set_xticklabels(nomes)

plt.show()
# Modelo GBM

model_gbm = GradientBoostingRegressor(n_estimators = 270, loss = 'huber', max_depth = 4,

                                      learning_rate = 0.05, subsample = 0.7)

model_gbm.fit(X_treino,Y_treino)

previsoes_final_gbm = model_gbm.predict(data_modelo_teste.values)
# Ajustando os resultados para ficarem entre 1-100

for i in range(0,len(previsoes_final_gbm)):

        if previsoes_final_gbm[i] > 100:

            previsoes_final_gbm[i] = 100          
# Ajustando os resultados para ficarem entre 1-100

for i in range(0,len(previsoes_final_gbm)):

        if previsoes_final_gbm[i] < 1:

            previsoes_final_gbm[i] = 1
# Transformando as previsoes em numeros inteiros

previsoes_final = previsoes_final_gbm.round(0).astype('int')
dataset_test_submission = pd.read_csv("../input/dataset_teste.csv", index_col = 'Property Id')

dataset_test_submission['score'] = previsoes_final

Submission = dataset_test_submission['score']

Submission = pd.DataFrame(Submission);Submission.head(10)
Submission.to_csv('Submission.csv', header = True)