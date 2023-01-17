import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve, accuracy_score 

from sklearn import linear_model
import warnings 

import math
data_cidades = pd.read_csv('../input/cities.csv')

data_q_vida = pd.read_csv('../input/movehubqualityoflife.csv')

data_c_vida = pd.read_csv('../input/movehubcostofliving.csv')
data_cidades.head() #imprimindo os dados contidos nas primeiras linhas do dataset
data_cidades.tail() #imprimindo os dados contidos nas ultimas linhas do dataset
data_cidades.shape #imprimindo numero de linhas e coulnas no dataset
data_q_vida.head() #imprimindo os dados contidos nas primeiras linhas do dataset
data_q_vida.tail() #imprimindo os dados contidos nas ultimas linhas do dataset
data_q_vida.shape #imprimindo numero de linhas e coulnas no dataset
data_c_vida.head() #imprimindo os dados contidos nas primeiras linhas do dataset
data_c_vida.tail() #imprimindo os dados contidos nas ultimas linhas do dataset
data_c_vida.shape #imprimindo numero de linhas e coulnas no dataset
data_cidades.isna().sum() #verificando valores nulos 
data_q_vida.isna().sum() #verificando valores nulos
data_c_vida.isna().sum() #verificando valores nulos
data_cidades[data_cidades['Country'].isnull()] #retornando valores nulos do dataset em uma coluna especifica
data_cidades.iloc[654, 1]= 'Ukraine' #setando pais na coluna country após pesquisa
data_cidades.iloc[724, 1]= 'Russia' #setando pais na coluna country após pesquisa
data_cidades.iloc[1529, 1]= 'Kosovo' #setando pais na coluna country após pesquisa
data_cidades[data_cidades['Country'].isnull()] #validando a atualização
data_q_c_vida = pd.merge(data_c_vida, data_q_vida) #realizando a junção dos datasets de custo de vida e qualidade de vida.
data_q_c_vida.head() #imprimindo os dados contidos nas primeiras linhas do dataset
data_q_c_vida.tail() #imprimindo os dados contidos nas ultimas linhas do dataset
data_q_c_vida.shape #imprimindo o numero de linhas e colunas do dataset
data_q_c_vida.dtypes #imprimindo tipos contidos nas variaveis
data_q_c_vida = data_q_c_vida.sort_values(by='City') #ordenação dos valores do novo dataset ao longo do eixo City
data_q_c_vida.reset_index(drop=True) #redefinindo indices e removendo niveis multiIndex
data_q_c_vida_2 = pd.merge(data_q_c_vida, data_cidades,how='left', on='City') #junção das colunas, cidades do lado esquerdo
data_q_c_vida_2.head() #imprimindo colunas para verificação do MERGE
data_q_c_vida_2[data_q_c_vida_2['Country'].isnull()] #verificando valores nulos
data_q_c_vida_2.iloc[227,0]='Zürich' #preenchendo valores nulos com valores pesquisados para novo dataset
data_q_c_vida_2.iloc[224,0]='Washington, D.C.'
data_q_c_vida_2.iloc[201,0]='Tampa, Florida'
data_q_c_vida_2.iloc[188,0]='São Paulo'
data_q_c_vida_2.iloc[185,0]='San Francisco, California'
data_q_c_vida_2.iloc[184,0]='San Diego, California'
data_q_c_vida_2.iloc[193,13]='Malta'
data_q_c_vida_2.iloc[10,13]='United States'
data_q_c_vida_2.iloc[51,13]='Philippines'
data_q_c_vida_2.iloc[61,13]='Argentina' 
data_q_c_vida_2.iloc[66,0]='Davao City'
data_q_c_vida_2.iloc[74,0]='Düsseldorf'
data_q_c_vida_2.iloc[79,0]='Frankfurt am Main'
data_q_c_vida_2.iloc[81,13]='Ireland' 
data_q_c_vida_2.iloc[100,0]='İstanbul'
data_q_c_vida_2.iloc[101,0]='İzmir'
data_q_c_vida_2.iloc[122,13]='Poland'
data_q_c_vida_2.iloc[129,0]='Málaga'
data_q_c_vida_2.iloc[130,0]='Malmö'
data_q_c_vida_2.iloc[134,13]='Spain'
data_q_c_vida_2.iloc[136,0]='Medellín'
data_q_c_vida_2.iloc[139,0]='Miami, Florida'
data_q_c_vida_2.iloc[141,0]='Minneapolis, Minnesota'
data_q_c_vida_2.iloc[164,13]='Thailand'
data_q_c_vida_2.iloc[166,0]='Philadelphia, Pennsylvania'
data_q_c_vida_2.iloc[167,0]='Phoenix, Arizona'
data_q_c_vida_2.iloc[168,0]='Portland, Oregon'
data_q_c_vida_2.iloc[176,0]='Rio de Janeiro'
data_q_c_vida_2.iloc[178,13]='United States'
data_q_c_vida_2.iloc[183,0]='San Antonio, Texas'
data= pd.merge(data_q_c_vida_2, data_cidades, how='inner', on='City') #criando merge para novo dataset
data.head() #imprimindo colunas para verificação do MERGE - criado Country_x e Country_y para comparação
data.tail() #podemos verificar que Country_x ainda esta com valores nulos comparado a nova coluna Cuntry_y
data= data.drop('Country_x', axis=1) #dropando cloluna com valores nulos

data=data.rename(columns={'Country_y': 'Country'}) #renomeando coluna que vamos trabalhar



data[data['Country'].isnull()] #verificando a existencia de valores nulos
data.isna().sum()
#data['City'].duplicated() # verificar dados duplicados
cidades_duplicadas = data.City.value_counts() # passando para variavel os valor e seus contadores para tratar duplicatas na coluna City
cidades_duplicadas.head(11) # lendo os valores duplicados existentes na coluna City
data[(data['City'] == 'Valencia')] # lendo os valores duplicados existentes na coluna City de um dado especifico
data[(data['City'] == 'Cambridge')]
data[(data['City'] == 'London')]
data.set_index('City').index.get_duplicates() # retornando uma lista classificada de elementos de índice que aparecem mais de uma vez no índice.
data = data.drop_duplicates(subset=['City', 'Country']) # dropando dados duplicados  considerando as colunas City e Country

# desta forma o subset ajuda a não excluir cidades que estajam duplicadas, pois a avaliação será feita nas duas colunas e 

# somente quando existir City e Country na mesma linha com duplicados = True o mesmo irá dropar.
cidades_duplicadas = data.City.value_counts() # Observando os dados que foram tratados na função drop_duplicates + Subset
cidades_duplicadas.head(11) 
data[(data['City'] == 'Valencia')] # Observando os dados que foram tratados na função drop_duplicates + Subset
data[(data['City'] == 'Cambridge')] 
data[(data['City'] == 'London')] 
nomes_cidades = data.City.value_counts() # Observando os dados que foram tratados na função drop_duplicates + Subset
nomes_cidades[nomes_cidades> 1] 
data.columns = data.columns.str.replace(' ','_') # padronização das colunas
data.head() # lendo o dataset
data.describe() # verificando contador de linhas, media, desvio padrão, minimo, 1° quartil, 2° quartil, 3° quartil e o maximo
col = data.drop(['City', 'Country'], axis = 1) # passando para uma variavel as colunas numericas para analise de OUTLIERS
col[(np.abs(col)> 3000).any(1)] # selecionar todas as linhas que excedam o valor 3300 utilizando any para booleano
col[(np.abs(col)< 0).any(1)] # selecionar todas as linhas que tenha valores negativos utilizando any para booleano
superior = 3000

avg = 1427



data.loc[(data['Avg_Disposable_Income']>superior), 'Avg_Disposable_Income'] = avg

data.loc[(data['Avg_Rent']>superior), 'Avg_Rent'] = avg
# passando preditoras para X

# passando target para Y



X = data.drop(['Movehub_Rating','Country','City'], axis = 1) 

Y = data['Movehub_Rating']

test_size=0.30

seed=42
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X=StandardScaler().fit_transform(X)

scaler
# importação train_test_split

from sklearn.model_selection import train_test_split



# utilização train_test_split, 30% teste e 70% treino



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# modelo regressão linear



reg_log = linear_model.LinearRegression()

reg_log.fit (X_train,Y_train)



Y_pred = reg_log.predict(X_test)
from sklearn.metrics import explained_variance_score

explained_variance_score=explained_variance_score(Y_test, Y_pred) 
from sklearn.metrics import mean_squared_error

mean_squared_error=mean_squared_error(Y_test, Y_pred)
from sklearn.metrics import mean_absolute_error

erro_medio_absoluto=mean_absolute_error(Y_test, Y_pred)
print ("Determination coefficient r²: {:.3f}".format(reg_log.score(X_train,Y_train)))

print ("Determination coefficient r² of the testing set: {:.3f} ".format(reg_log.score(X_test,Y_test)))

print("mean_squared_error: {:.2f} ".format(mean_squared_error))

print("mean_absolute_error: {:.2f} ".format(erro_medio_absoluto))

print("Explained_variance_score: {:.3f} ".format(explained_variance_score))

print ('Linear Coefficient {:.2f}'.format(reg_log.intercept_))

print ('Angular Coefficient{:}'.format(reg_log.coef_))
pd.DataFrame(list(zip(Y_test[10:20],Y_pred[10:20])),columns=['Avaliação Movehub','Predição']) # resultado