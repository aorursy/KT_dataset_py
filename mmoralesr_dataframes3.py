import pandas as pd

# crea un data frame con dos columnas y cuatro filas

df = pd.DataFrame({"col1": [1,3,11,2,4,7,9,5], "col2": [9,23,0,2,1,7,3,4]})

df 
import numpy as np # para tener disponible la función sqrt

### creación de una nueva columna 

df['rcol2']=np.sqrt(df["col2"])

df
df['sum_col']=df.eval('col1+col2')

#df

#otra forma es mediante

#df['sum_col']=df['col1']+df['col2']

# o mediante 

#df['sum_col']=df.col1 + df.col2

df
df1=df[df.col1>4] 

df1
df2=df[df.col1<=4]

df2
df12=df1.append(df2)

df12
tablas=[df1,df2] # un arreglo de tablas 

tablas
dfC=pd.concat(tablas, ignore_index=True)

#dfC=pd.concat(tablas, ignore_index=False)

dfC
dfC=pd.concat(tablas,keys=['Hombres', 'Mujeres'])

dfC
dfC.loc['Mujeres']
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],

                       'B': ['B0', 'B1', 'B2', 'B3'],

                        'C': ['C0', 'C1', 'C2', 'C3'],

                        'D': ['D0', 'D1', 'D2', 'D3']},

                       index=[0, 1, 2, 3])

print(df1)

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],

                        'B': ['B4', 'B5', 'B6', 'B7'],

                        'C': ['C4', 'C5', 'C6', 'C7'],

                        'D': ['D4', 'D5', 'D6', 'D7']},

                       index=[4, 5, 6, 7])

print(df2)

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],

                        'B': ['B8', 'B9', 'B10', 'B11'],

                        'C': ['C8', 'C9', 'C10', 'C11'],

                        'D': ['D8', 'D9', 'D10', 'D11']},

                       index=[8, 9, 10, 11])

print(df3)
frames=[df1,df2,df3]

result=pd.concat(frames)

result
result = pd.concat(frames, keys=['x', 'y', 'z'])

result
df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],

                    'F': ['F2', 'F3', 'F6', 'F7'],

                     'D': ['D2', 'D3', 'D6', 'D7']},

                         index=[2, 3, 7, 6])

df4
print(df1)

print(df4)

result = pd.concat([df1, df4],axis=0)

result
print(df1)

print(df4)

#result = pd.concat([df1, df4], axis=1, sort=False)

result = pd.concat([df1, df4], axis=1, sort=True)

result
result.shape

#result["B"]

#result.loc[:,"B"]

#result.columns

#list(result.columns)
result = pd.concat([df1, df4], axis=1, join='inner')

result
result = pd.concat([df1, df4], axis=1, join='outer')

result
result=df1.append(df2)

result
result=df1.append(df4)

result
a = np.array([1,2,3,4]) 

#print( np.tile(a,2) ) 

df["col3"]=np.tile(a, 2)

df
df.sort_values(by=['col1'])
df.sort_values(by=['col3'])
df.sort_values(by=['col3','col1'])
df.info()
from pandas.api.types import CategoricalDtype

# crea una nueva columna como factor a partir de col3 

#df["factor"]=df.col3.astype('category')

# cambia el tipo de dato a factor 

#df["col3"]=df.col3.astype('category')

#df.info()

#cat_dtypes=CategoricalDtype(categories=["C1","C2","C3","C4"], ordered=False)

#df_cat=df["col3"].astype(cat_dtypes)
df_copia=df.copy()

rep={1:"c1",2:"c2",3:"c3",4:"c4"}

df_copia["col3"]=df_copia["col3"].replace(rep) 

#df_copia=df_copia.replace(rep)
df_copia.head()

#df.col3.rename_categories(["C1","C2","C3","C4"])
df_copia.info()
df_copia["col3"]=df_copia.col3.astype('category') 
df_copia.info()
df_copia.col3.value_counts() 
import random as rnd 

genero=rnd.choices(['f', 'm'], [100,100], k=169) 

datos=pd.DataFrame( {'Edad':[27,48,27,63,53,43,24,26,56,39,24,26,40

,27,32,63,30,33,34,45,27,31,32,51,30,28

,42,35,44,21,24,28,26,30,26,25,30,27,29

,23,22,22,45,25,23,29,32,23,32,41,26,29

,37,23,44,28,43,61,48,43,18,23,47,36,24

,47,37,45,42,39,24,34,29,38,47,30,24,28

,30,33,40,40,40,29,41,24,53,34,42,50,22

,27,26,48,26,22,32,53,50,40,26,33,31,50

,47,29,36,29,25,38,30,30,23,46,31,42,30

,41,26,51,48,21,62,27,31,24,21,29,34,38

,19,52,31,53,26,25,22,30,18,19,37,27,28

,52,20,28,27,22,34,27,24,49,37,40,28,23

,48,37,44,38,48,46,38,26,49,36,31,31,39], 'Genero':genero} ) 

datos.head()
print("minimo", datos.Edad.min())

print("maximo", datos.Edad.max())



datos['GrEdad'] = pd.cut(datos['Edad'], bins=[17, 26, 59, float('Inf')], labels=['Joven', 'Adulto', 'Viejo'])

datos[datos.Edad==18]
datos.head()
datos.GrEdad.value_counts() 
datosgrp=datos.groupby(['Genero','GrEdad']) 

datosgrp.count()