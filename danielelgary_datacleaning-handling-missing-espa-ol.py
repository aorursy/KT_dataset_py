
# Librerias a utilizar
import pandas as pd
import numpy as np

# Lectura de los DataSets
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# Usar una semilla para reproducir los mismos conjuntos de datos.
np.random.seed(0) 
# Miremos algunas filas de nfl_data. Podemos ver que faltan datos!
nfl_data.sample(5)
# your turn! Look at a couple of rows from the sf_permits dataset. Do you notice any missing data?

sf_permits.sample(10)

# your code goes here :)
# get the number of missing data points per column
# Obtenemos el numero de los datos perdidos por cada columna (sum funciona para cada columna del DataSet)
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
# Miremos la cuenta de las primeras 10 columnas.
missing_values_count[0:10]
# how many total missing values do we have?
# Cuantos valores perdidos tenemos en total?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
# Porcentaje de datos perdidos en el DatSet
(total_missing/total_cells) * 100
# your turn! Find out what percent of the sf_permits dataset is missing

conteo_nulos = sf_permits.isnull().sum()
conteo_nulos[:15]

#Multiplica cantidad de filas * cantidad de columnas del DataSet para saber cantidad de datos totales incluidos NaN
total_datos = np.product(sf_permits.shape)
#Suma el total de datos perdidos
total_perdidos = conteo_nulos.sum()
#Porcentaje de datos perdidos del DataSet
(total_perdidos/total_datos)*100
# look at the # of missing points in the first ten columns
missing_values_count[0:15]
print (conteo_nulos["Street Number Suffix"])
print (conteo_nulos["Zipcode"])

sf_permits.sample(20)
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_permits.dropna()
# Now try removing all the columns with empty values. Now how much of your data is left?
columnas_sin_nan = sf_permits.dropna(axis=1)
columnas_sin_nan.head()

print ("Columnas en el DataSet original = %d \n" % sf_permits.shape[1])
print ("Columnas en el DataSet sin NaN = %d \n" % columnas_sin_nan.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then replacing any remaining NaN's with 0

sf_permits.fillna(method='pad', axis=0).fillna(0).head()
