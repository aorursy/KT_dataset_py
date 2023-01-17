#Importamos las librerias basicas y necesarias

import pandas as pd #Libreria de manipulacion de datos

import numpy as np #Libreria numerica muy potente

import matplotlib.pyplot as plt #Libreria para graficos

import seaborn as sns #Libreria para graficos basada en Matplotlib (es mas simple)
df_airbnb = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv") 

#La funcion read_ tiene muchas opciones dependiendo el tipo de dato que se vaya a cargar
df_airbnb.head(6) #Vemos los primeros 6 datos para ver que se cargaron correctamente
df_airbnb.shape #Esto nos brinda las dimensiones de los datos ("filas","columnas")
df_airbnb[48889:-1] #Esta es otra forma de filtrar una porcion del dataset

#esto se interpreta [desde:hasta], el -1 se utiliza para indicar el ultimo valor
df_airbnb.columns #Vemos los features (asi vamos a llamar a las columnas) que tiene nuestro dataset
print("cantidad de barrios:", len(df_airbnb["neighbourhood_group"].unique())) 

#len brinda la longitud de un array

#unique() brinda los valores sin repetir 

#df["host_id"] metodo de filtrado por features
df_airbnb[["minimum_nights","price","availability_365"]].describe()

#Brinda los estadisticos basicos de las columnas seleccionadas
df_filter = df_airbnb[df_airbnb["price"]<df_airbnb["price"].mean()]

#Esta es otra forma de filtrar datos por condiciones

#En este caso filtramos por precio, los alquileres que esten por debajo de la media
df_filter["price"].max() #Vemos si se aplico el filtro, visualizamos el valor maximo del precio
df_airbnb["price"].mean() #Validamos y vemos que filtro correctamente los datos por la media
df_airbnb.isna().sum() #De esta forma vemos el total de valores NaN en nuestro dataset

#Si no se utiliza el sum() va a devolver la condicion True o False de los campos
df_airbnb.isna().sum()/df_airbnb.shape[0]*100
#Aprovechamos a eliminar otras columnas que no seran necesarias

df_airbnb.drop(['name','id','host_name','last_review','reviews_per_month','calculated_host_listings_count'], axis=1, inplace=True)

#cuando vean en la mayoria de las funciones AXIS es = 1 (columna), 0 (fila)
#Agrupamos por "neighbourhood_group" y mostramos el precio mas alto por barrio

df_airbnb.groupby(['neighbourhood_group'])[["price"]].max()
df_airbnb.columns
plt.figure(1, figsize=(10,6)) #Tamaño del grafico

plt.title("Distribucion del precio") #Titulo

sns.boxplot(df_airbnb["price"]) #Grafico, en este caso estamos usando la libreria seaborn



plt.figure(2, figsize=(10,6))

plt.title("Distribucion de las noches minimas de estadia")

sns.distplot(df_airbnb["minimum_nights"])



plt.figure(3, figsize=(10,6))

plt.title("Distribucion de la cantidad de reviews")

sns.distplot(df_airbnb["number_of_reviews"])
#Quitando valores extremos en el precio

p10 = np.percentile(df_airbnb["price"], 10)

p90 = np.percentile(df_airbnb["price"], 90)

df_airbnb = df_airbnb[(df_airbnb["price"] >= p10) & (df_airbnb["price"] <= p90)]
#Quitando valores extremos en la cantidad de noches minimas

p10 = np.percentile(df_airbnb["minimum_nights"], 10)

p90 = np.percentile(df_airbnb["minimum_nights"], 90)

df_airbnb = df_airbnb[(df_airbnb["minimum_nights"] >= p10) & (df_airbnb["minimum_nights"] <= p90)]
plt.figure(1, figsize=(10,6)) #Tamaño del grafico

plt.title("Distribucion del precio") #Titulo

sns.boxplot(df_airbnb["price"]) #Grafico, en este caso estamos usando la libreria seaborn



plt.figure(2, figsize=(10,6))

plt.title("Distribucion de las noches minimas de estadia")

sns.distplot(df_airbnb["minimum_nights"])
plt.figure(figsize=(8,6))

plt.title("Tipos de habitacion por precio")

sns.barplot(y='price',x='room_type',data=df_airbnb, palette="Set1")

plt.show()
plt.figure(figsize=(8,6))

plt.title("Cantidad de alojamientos por barrio")

sns.countplot(df_airbnb['neighbourhood_group'])
plt.figure(figsize=(10,7))

sns.barplot(x = "neighbourhood_group", y="price", hue = "room_type",data = df_airbnb, palette="Set1")

plt.title("Precio agrupado por barrios y tipo de habitacion")

plt.show()
f,ax = plt.subplots(figsize=(16,8))

ax = sns.scatterplot(y=df_airbnb["latitude"],x=df_airbnb["longitude"],hue=df_airbnb["neighbourhood_group"],palette="coolwarm")

plt.show()
#Vamos a trabajar solo con las columnas categoricas

df_airbnb_cat = df_airbnb[["neighbourhood_group","neighbourhood","room_type"]]

df_airbnb_cat.head()
dic_room_type = {

    "Private room":0,

    "Entire home/apt":1,

    "Shared room":2

}
#Con la funcion map unimos el diccionario a nuestro dataset agregandolo a una nueva columna "dic_room_type"

df_airbnb_cat['dic_room_type'] = df_airbnb_cat['room_type'].map(dic_room_type) 
df_airbnb_cat.head()
neighbourhood_dummy = pd.get_dummies(df_airbnb_cat["neighbourhood_group"])

neighbourhood_dummy.head() #Vemos como quedaron transformados los barrios
df_airbnb_cat = pd.concat([df_airbnb_cat, neighbourhood_dummy], axis=1) #Unimos nuestro "neighbourhood_dummy" al "df_airbnb_cat"

df_airbnb_cat.head()

df_airbnb_cat.drop(['neighbourhood_group',"neighbourhood","room_type","dic_room_type"], axis=1, inplace=True)
room_type_d = pd.get_dummies(df_airbnb["room_type"]) #Corregimos el room_type"
df_airbnb = pd.concat([df_airbnb, neighbourhood_dummy,room_type_d], axis=1) #Unimos todos los dataframe
df_airbnb.drop(['neighbourhood_group',"neighbourhood","room_type"], axis=1, inplace=True) #Eliminamos las columnas que ya no necesitamos
df_airbnb.head()