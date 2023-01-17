import pandas as pd
import numpy as np
import missingno as msno
# Matplotlib for additional customization
from matplotlib import pyplot as plt
%matplotlib inline

# Seaborn for plotting and styling
import seaborn as sns
# Para reducir memoria de los datasets:
def reduce_memory(ds_tmp):
    int_columns = ds_tmp.select_dtypes(include=[np.int16,np.int32,np.int64]).columns.tolist()
    for col in int_columns:
        ds_tmp[col] = pd.to_numeric(arg=ds_tmp[col], downcast='integer')

    float_columns = ds_tmp.select_dtypes(include=[np.float64]).columns.tolist()
    for col in float_columns:
        ds_tmp[col] = pd.to_numeric(arg=ds_tmp[col], downcast='float')
import os
print(os.listdir("../input/sistema-recomendador-bbva-kfc/"))
data_base_trx = pd.read_csv("../input/sistema-recomendador-bbva-kfc/01dataBaseTrainTrxRec.csv")
data_base_perfil = pd.read_csv("../input/sistema-recomendador-bbva-kfc/02dataBasePerfilRec.csv")
print("data_base_trx:",data_base_trx.shape)
print("data_base_perfil:",data_base_perfil.shape)
data_base_trx.head()
data_base_perfil.head()
data_base_trx.info()
data_base_perfil.info()
# Agruparemos todos los consumos del año de los clientes por establecimiento
cliEstab = data_base_trx.groupby(['codCliente','codEstab'], as_index=False ).agg({'ratingMonto':'sum'})
df_ratingMonto = cliEstab.copy()
df_ratingMonto.head(25)
df_ratingMonto.info()
reduce_memory(df_ratingMonto)
df_ratingMonto.info()
n_clie_ori = df_ratingMonto.codCliente.nunique()
n_estab_ori = df_ratingMonto.codEstab.nunique()

print('Num de Clientes: '+ str(n_clie_ori))
print('Num de Establecimientos: '+str(n_estab_ori))
list_codEstab = df_ratingMonto.codEstab.value_counts().index.values
list_codEstab.sort()
list_codEstab = ['estab' + str(i) for i in list_codEstab ] # agregando el prefijo "estab" a cada codEstab
list_codEstab[0:10]
min_nClie_x_estab = 50 # valor elegido ad-hoc
# Contamos el nro de clientes por Establecimiento
contadorRM_Est = df_ratingMonto.groupby("codEstab")['codCliente'].count()
# Identificamos los establecimientos que cumplen el nro mínimo de clientes
Estab_selected = contadorRM_Est[contadorRM_Est >= min_nClie_x_estab].index.tolist()
Estab_selected.sort()
Estab_selected[:5]
# Seleccionamos los registros con los establecimientos identificados: fx: ".isin()"
df_ratingMonto = df_ratingMonto.loc[df_ratingMonto['codEstab'].isin(Estab_selected)]
df_ratingMonto.shape
min_nEstab_x_clie = 20 # valor elegido ad-hoc
# Contamos el nro de establecimientos por Cliente
contadorRM_Clie = df_ratingMonto.groupby("codCliente")['ratingMonto'].count()
# Identificamos los clientes que cumplen el nro mínimo de establecimientos
Clie_selected = contadorRM_Clie[contadorRM_Clie >= min_nEstab_x_clie].index.tolist()
Clie_selected.sort()
Clie_selected[:5]
# Seleccionamos los registros con los clientes identificados: fx: ".isin()"
df_ratingMonto = df_ratingMonto.loc[df_ratingMonto['codCliente'].isin(Clie_selected)]
df_ratingMonto.shape
# **Tamaño de la base final:**
df_ratingMonto.shape
print("Las dimensiones de Clientes y Establecimientos de la base final a trabajar el Sistemas de Recomendación son:")
n_clie = df_ratingMonto.codCliente.nunique()
n_estab = df_ratingMonto.codEstab.nunique()

print('Num de Clientes: '+ str(n_clie))
print('Num de Establecimientos: '+str(n_estab))
df_ratingMonto.head()
list_codEstab = df_ratingMonto.codEstab.value_counts().index.values
list_codEstab.sort()
print(list_codEstab[0:10])
list_codEstab = ['estab' + str(i) for i in list_codEstab ] # agregando el prefijo "estab" a cada codEstab
print(list_codEstab[0:10])
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df_ratingMonto, test_size=0.20, random_state = 99 ) # 20% Test

train_data.reset_index(drop = True,inplace=True)
test_data.reset_index(drop = True,inplace=True)
#Creando la matrices cliente-establecimiento para el training y calcular las predicciones para el test
train_data_matrix = np.zeros((n_clie_ori + 1, n_estab), dtype='float32')
train_data_matrix.shape
# Seleccionamos 
train_data_matrix = pd.DataFrame(train_data_matrix, columns = list_codEstab).iloc[df_ratingMonto.codCliente.unique(),:]
train_data_matrix.shape
from time import time
time_star = time()

for line in train_data.itertuples():
    train_data_matrix.loc[line[1], "estab"+str(line[2])] = line[3]  
    
time_end = time()
print ("Time: ", np.round((time_end-time_star)/60,1), " minutes")
train_data_matrix.head(10)
train_data_matrix.tail(10)
train_data_matrix.info()
#Create two user-item matrices, testing
test_data_matrix = np.zeros((n_clie_ori + 1, n_estab), dtype='float32') 
test_data_matrix.shape
test_data_matrix = pd.DataFrame(test_data_matrix, columns = list_codEstab).iloc[df_ratingMonto.codCliente.unique(),:]
test_data_matrix.shape
from time import time
time_star = time()

for line in test_data.itertuples():
    test_data_matrix.loc[line[1], "estab"+str(line[2])] = line[3]  
    
time_end = time()
print ("Time: ", np.round((time_end-time_star)/60,1), " minutes")
test_data_matrix.head(10)
test_data_matrix.tail(10)
test_data_matrix.info()
# Funcion para extraer los establecimientos top por cliente
def fx_estab_tops_xCliente(base_ratings, cod_cliente, n_tops = None):
    "Funcion para extraer los establecimientos top(de mayor a menor rating) de un cliente determinado."
    if cod_cliente not in base_ratings.index.values:
            print("Cliente No Encontrado")
            return(None)
    else:
        row_cliente = base_ratings.loc[cod_cliente,:]
        row_cliente.fillna(0, inplace = True)
        estab_of_cli = row_cliente.index[row_cliente.nonzero()].values
        lista_estab_final = row_cliente[estab_of_cli].sort_values(ascending = False)
        lista_estab_final = pd.DataFrame({'Estab': lista_estab_final.index.values, 
                                          'montoRating': lista_estab_final.values})
        return(lista_estab_final.head(n_tops))
fx_estab_tops_xCliente(base_ratings = test_data_matrix, cod_cliente = 99, n_tops = None)
from sklearn.metrics.pairwise import cosine_similarity

from time import time
time_star = time()

print("Iniciando clie_similaridad ")
clie_similaridad = cosine_similarity(X = train_data_matrix) # metric='cosine', 'euclidean'
print("Finalizó clie_similaridad ")

time_end = time()
print ("Time: ", np.round((time_end-time_star)/60,4), " minutes")
print("Dimensiones:",clie_similaridad.shape)
pd.DataFrame(clie_similaridad).head()
from time import time
time_star = time()

print("Iniciando estab_similaridad ")
estab_similaridad = cosine_similarity(X = train_data_matrix.T)
print("Finalizó estab_similaridad")

time_end = time()
print ("Time: ", np.round((time_end-time_star)/60,4), " minutes")
print("Dimensiones:",estab_similaridad.shape)
pd.DataFrame(estab_similaridad).head()

def predict(ratings, similaridad, type='clie'):
    if type == 'clie':
        mean_clie_rating = np.nanmean(ratings,axis=1)
        #np.newaxis: crea una nueva dimensión al array (o matrix)
        ratings = np.nan_to_num(ratings) # reemplazar nulos con ceros
        ratings_diff = (ratings - mean_clie_rating[:, np.newaxis]) 
        pred = mean_clie_rating[:, np.newaxis] + similaridad.dot(ratings_diff)/np.array([np.abs(similaridad).sum(axis=1)]).T
    elif type == 'estab':
        pred = ratings.dot(similaridad)/np.array([np.abs(similaridad).sum(axis=1)])     
    return pred
print("clie_based_prediccion...")
clie_prediccion = predict(train_data_matrix.values, clie_similaridad, type='clie')
clie_prediccion = pd.DataFrame(clie_prediccion)
clie_prediccion.index, clie_prediccion.columns = train_data_matrix.index, train_data_matrix.columns
print("fin clie_based_prediccion...")
print("estab_based_prediccion...")
estab_prediccion = predict(train_data_matrix.values, estab_similaridad, type='estab')
estab_prediccion = pd.DataFrame(estab_prediccion)
estab_prediccion.index, estab_prediccion.columns = train_data_matrix.index, train_data_matrix.columns
print("fin estab_based_prediccion...")
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediccion, ratings_true):
    indexes_nonzeros = ratings_true.nonzero() # obtener los indices de los ratings reales (<> 0)
    prediccion = prediccion[indexes_nonzeros].flatten() # flatten(): convierte la matriz en una sola dimension
    ratings_true = ratings_true[indexes_nonzeros].flatten() # flatten(): convierte la matriz en una sola dimension
    return sqrt(mean_squared_error(prediccion, ratings_true))
# Comparando ambas predicciones
print('clie-based CF RMSE: ' + str(rmse(clie_prediccion.values, test_data_matrix.values)))
print('estab-based CF RMSE: ' + str(rmse(estab_prediccion.values, test_data_matrix.values)))
