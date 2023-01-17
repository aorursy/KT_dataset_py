# Importar la librería Pandas
import pandas as pd

# Importar la librería NumPy
import numpy as np
# Crear una serie simple
serie_simple = pd.Series(index=[0,1,2,3], name='Volume', data=[1000,2600,1524,98000])
serie_simple
# Crear un DataFrame simple
df_simple = pd.DataFrame(index=[0,1,2,3], columns=['Volume'], data=[1000,2600,1524,98000])
df_simple
# Crear otro DataFrame
otro_df = pd.DataFrame(index=[0,1,2,3], columns=['Date','Volume'], data=[[20190101,1000],[20190102,2600],[20190103,1524],[20190104,98000]])
otro_df
# Cargar un archivo como DataFrame y asignarlo a df
df = pd.read_csv('D.csv')
# Ver el encabezado (head) del DataFrame (las primeras filas)
df.head()
# Mirar la "cola" (las últimas filas) del DataFrame
df.tail()
# Representa las dimensiones de la estructura de la forma (num_filas, num_columnas)
df.shape
# Lista de los nombres de las columnas del DataFrame
list(df.columns)
# Lista de los índices del DataFrame
list(df.index)[0:20] # Sólo se muestran los primeros 20 valores para reducir la información que se imprime
# Añadir una columna llamada "Symbol" (símbolo)
df['Symbol'] = 'D'
df.head()
# Podemos acceder a una columna escribiéndola entre paréntesis cuadrados []
df['Volume'].head() # Se añade .head() para reducir los datos a imprimir
# Se añade una nueva columna llamada "Volume_Millions" (Volumen en millones), calculada desde "Volume"
df['Volume_Millions'] = df['Volume'] / 1000000.0 # divide cada dila de df['Volume'] por 1 millón y lo guarda en la nueva columna
df.head()
# Demos un vistazo a la nueva forma del DataFrame. Se han añadido dos columnas más.
df.shape
df['VolStat'] = (df['High'] - df['Low']) / df['Open']
df['Return'] = (df['Close'] / df['Open']) - 1.0
df.head()
# Calcular el valor mínimo de la columna Volume_Millions
df['Volume_Millions'].min()
# Calcular la mediana de la columna Volume_Millions
df['Volume_Millions'].median()
# Calcular el valor promedio de la columna Volume_Millions
df['Volume_Millions'].mean()
# Calcular el valor máximo de la columna Volume_Millions
df['Volume_Millions'].max()
# Calcular el percentil 25 (1er cuartil)
df['Volume_Millions'].quantile(0.25)
# Calcular el percentil 75 (3er cuartil)
df['Volume_Millions'].quantile(0.75)
df['Volume_Millions'].describe()
# Cargar 5 archivos CSV en un solo DataFrame
print("Definir Símbolos de acciones a leer")
simbolos = ['D','EXC','NEE','SO','DUK']
lista_de_df = []

# Ciclo sobre los símbolos
print(" --- Inicio del ciclo sobre los símbolos --- ")
for i in simbolos:
    print("Procesando símbolo: " + i)
    temp_df = pd.read_csv(i + '.csv')
    temp_df['Volume_Millions'] = temp_df['Volume'] / 1000000.0
    temp_df['Symbol'] = i # Se añade nueva columna con el símbolo para diferenciar los datos de cada uno
    lista_de_df.append(temp_df)

print(" --- Fin del ciclo sobre los símbolos --- ")
    
# Combinar en un solo DataFrame usando concat
print("Consolidando datos...")
df_uni = pd.concat(lista_de_df, axis=0)

# Se añaden estadísticas destacadas y se prepara análisis de volatilidad
print('Calculando estadísticas destacadas...')
df_uni['VolStat'] = (df_uni['High'] - df_uni['Low']) / df_uni['Open'] # Volatilidad diaria
df_uni['Return'] = (df_uni['Close'] / df_uni['Open']) - 1.0 # Retorno diario

print("Dimensiones del DataFrame df_uni (filas, columnas): ")
print(df_uni.shape)

print("Encabezado de df_uni: ")
df_uni.head()
symbol_DUK_df = df_uni[df_uni['Symbol'] == 'DUK']
symbol_DUK_df.head()
# Observe que al usar el método groupby() se obtiene un objeto de tipo DataFrameGroupBy.
df_uni.groupby('Symbol')
grp_obj = df_uni.groupby('Symbol') # Datos agrupados por símbolo en df_uni

# Ciclo sobre cada grupo
for item in grp_obj:
    print(" ------ Inicio del Ciclo ------ ")
    print(type(item))     # Tipo de objeto de cada item en grp_obj
    print(item[0])        # Símbolo (Symbol)
    print(item[1].head()) # DataFrame de datos del símbolo (Symbol)
    print(" ------ Fin del Ciclo ------ ")
grp_obj = df_uni.groupby('Symbol') # Datos agrupados por símbolo en df_uni

# Ciclo sobre cada grupo
for item in grp_obj:
    print('------Símbolo: ', item[0])
    grp_df = item[1]
    relevant_df = grp_df[['VolStat']]
    print(relevant_df.describe())
# Volatilidad: VolStat
df_uni[['Symbol','VolStat']].groupby('Symbol').describe()
# Determinar umbrales inferiores de volatilidad para cada acción
umbrales_volstat = df_uni.groupby('Symbol')['VolStat'].quantile(0.5) # Percentil 50 (mediana)
print(umbrales_volstat)
# Definición de los símbolos a usar
print("Definiendo símbolos de las acciones")
simbolos = ['D','EXC','NEE','SO','DUK']
lista_de_df = []

# Ciclo sobre los símbolos a usar
print(" --- Inicio del ciclo sobre los símbolos --- ")
for i in simbolos:
    print("Etiquetando el comportamiento de la volatilidad para la acción: " + i)
    temp_df = df_uni[df_uni['Symbol'] == i].copy() # Crea una copia del dataframe para no modificar df_uni
    volstat_t = umbrales_volstat.loc[i]

    temp_df['NivelVol'] = np.where(temp_df['VolStat'] < volstat_t, 'BAJA', 'ALTA') # Etiquetado de la volatilidad
    lista_de_df.append(temp_df)
    
print(" --- Fin del ciclo sobre los símbolos --- ")

print("Consolidando los datos...")
df_etiquetado = pd.concat(lista_de_df)
df_etiquetado.head()
df_etiquetado.groupby(['Symbol','NivelVol'])[['Volume_Millions']].mean()
# Una solución posible
# Definir umbrales inferiores de volatilidad para cada acción
umbrales_volstat_75 = df_uni.groupby('Symbol')['VolStat'].quantile(0.75) # Percentil 75
umbrales_volstat_25 = df_uni.groupby('Symbol')['VolStat'].quantile(0.25) # Percentil 25

# Definición de los símbolos a usar
print("Definiendo símbolos de las acciones...")
simbolos = ['D','EXC','NEE','SO','DUK']
lista_de_df = []

# Loop over all symbols
print(" --- Ciclo sobre los símbolos --- ")
for i in simbolos:
    print("Etiquetando el comportamiento de la volatilidad para la acción: " + i)
    temp_df = df_uni[df_uni['Symbol'] == i].copy() #  Crea una copia del dataframe para no modificar df_uni
    volstat_t75 = umbrales_volstat_75.loc[i]
    volstat_t25 = umbrales_volstat_25.loc[i]
    
    temp_df['NivelVol'] = np.where(temp_df['VolStat'] > volstat_t75, 'ALTA',
                                  np.where(temp_df['VolStat'] > volstat_t25, 'MEDIA','BAJA')) # Etiqueta de volatilidad
    lista_de_df.append(temp_df)
    
print(" --- Ciclo finalizado --- ")

print("Consolidando datos...")
df_final = pd.concat(lista_de_df)
print(df_final.groupby(['Symbol','NivelVol'])[['Volume_Millions']].mean())
# Librería esencial para gráficas en Python
import matplotlib.pyplot as plt

# Instrucción para graficar directamente en el notebook
%matplotlib inline
# Convertir cadena a formato de fecha/hora
df_uni['DateTime'] = pd.to_datetime(df_uni['Date'], format='%Y-%m-%d')

# definir la fecha y hora como índice para graficar
df_uni = df_uni.set_index(['DateTime'])
df_uni.head()
# Vistazo al comportamiento de la volatilidad
fig, ax = plt.subplots(figsize=(15,6))
df_uni.groupby('Symbol')['VolStat'].plot(ax=ax, legend=True, title='Tendencia del Sector Energético - VolStat');
# Una posible solución

# Se agrega una columna para el año
lista_año_mes = []
for i in df_uni['Date']:
    lista_año_mes.append(i[:4]+i[5:7])
    
df_uni['YYYYMM'] = lista_año_mes

# Agrupar por símbolo y luego hacer un ciclo sobre el grupo y agrupar de nuevo por año y mes,
# luego calcular el volumen medio transado para cada mes
grp = df_uni.groupby('Symbol')
for item in grp:
    print('------Símbolo: ', item[0])
    grp_df = item[1]
    grp_df.head()
    df_mes_vol = grp_df[['YYYYMM','Volume_Millions']]
    df_año_mes = df_mes_vol.groupby('YYYYMM').mean()
    
    max_volumen = float(df_año_mes.max())
    print(df_año_mes[df_año_mes['Volume_Millions'] == max_volumen])
