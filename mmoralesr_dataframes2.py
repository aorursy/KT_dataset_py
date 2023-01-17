import pandas as pd

# crea un data frame con dos columnas y cuatro filas

df = pd.DataFrame({"col1": [1,3,11,2,4,7,9,5], "col2": [9,23,0,2,1,7,3,4]})

df 
# tres primeras filas 

df.head(4) 
# tres últimas filas 

df.tail(3) 
df.info()
# numero de filas y columnas del data 

df.shape 
df['col1']
df.col2 
import numpy as np # para tener disponible la función sqrt

### creación de una nueva columna 

#df['rcol2']=np.sqrt(df["col2"])

#df

df['rcol2']=df.eval('sqrt(col2)') 

df
df['sum_col']=df.eval('col1+col2')

#df

#otra forma es mediante

#df['sum_col']=df['col1']+df['col2']

# o mediante 

#df['sum_col']=df.col1 + df.col2

df
df[['col1','rcol2']]
# selecciona la primera fila del data

#print(df.iloc[0]) 

# equivalente a df.iloc[0,]

df.iloc[0,]

# selecciona la segunda fila del data

#print(df.iloc[1])

# equivalente a df.iloc[1,]

# selecciona la última fila del data

#print(df.iloc[-1])

# las primeras 5 filas 

#df.iloc[0:3]
# selecciona la primera columna del data con todas las filas 

#print(df.iloc[:,0])

# equivalente a df.iloc[0,]

#df.iloc[0,]

# selecciona la segunda columna del data con todas las filas 

print(df.iloc[:,2])

# equivalente a df.iloc[1,]

# selecciona la última columna del data con todas las filas 

#df.iloc[:,-1]
# Las dos primeras columnas con todas las filas 

df.iloc[:,0:2]
# filas y columnas específicas 

df.iloc[[0,3,6], [0,2]]
# selecciona las columnas que les corresponda True

df.iloc[:,[False,True,False,True]]
# selecciona las filas con ìndice par 

#print(df.index)

#print(list(df.index))

#lambda x: x.index % 2 == 0]

df.iloc[lambda x: x.index % 2 == 0] 
#df.col1>4

df.loc[df.col1>4]
df.loc[lambda df: df['col1']>4]