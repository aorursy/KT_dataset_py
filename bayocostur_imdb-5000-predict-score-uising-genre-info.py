# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
x = pd.read_csv('../input/movie_metadata.csv', delimiter=',')

x=pd.DataFrame(x)#Cargamos los datos
#Hacemos un chqeueo general para conocer qué tipo de datos tenemos.

x.info()
#x = x.dropna(how='any')#Eliminamos la filas con NAN'

print(x.shape[0])
#Ahora queremos extraer información útil dela variable genres.

#Para ello necesitamos depurar la información que aparece.

#En primer lugar nos centramos en obtener el total de géneros que existen.

total_gen=list(set(x["genres"]))#Reducimos duplicados

total_gen='|'.join(total_gen)#Construimos un string a partir de la lista anterior separando por'|'.

total_gen= set(total_gen.split("|"))#Construimos una lista, entendiendo '|' como separadores.

#y aplicamos el conjunto para cargarnos duplicados.

total_gen=list(total_gen)#Ya tenemos los diferentes generos, ahora de nuevo pasamosa lista.

#Ahora buscamos defnir para cada film una variable para cada genero, que tome valor 1 o 0 en

#función de si el film pertenece al genero. En primer lugar necesitamos la siguiente función.

def list_str_cont(x,g): #Funciona con strings, y verifica si x está contenido en g

    return 1*(g in x)

for gen in total_gen: #Ahora para cada genero creamos una nueva variable como queríamos.

    x[gen]=x["genres"].apply(list_str_cont,g=gen)#Aplicamos la funcion anerior a la lista de genres

x.drop(["genres"],axis=1,inplace=True)#Borramos la variable antigua
x.ix[:,-len(list(total_gen)):].head(2)#Vemos las nuveas columnas añadidas.
gen_corr_mat=x.corr(method="pearson")#matriz de correlación

gen_corr=gen_corr_mat["imdb_score"][-len(list(total_gen)):]#restrinjo a las ultimas variables y me quedo con imdbscore

gen_corr_index=gen_corr.index[abs(gen_corr)>0.07]#me quedo con las de mayor correlación

not_gen_corr_index=[i for i in x.columns[-len(list(total_gen)):] if(i not in gen_corr_index)]

not_gen_corr_index#almaceno las de menor correlación para borrarlas (las borro justo antes de la fase de análisis)
x=x.select_dtypes([np.number])#Selecionamos solo las variables que son numéricas.
x.head()
x.ix[:,:-len(gen_corr_index)].columns
corr_mat=x.ix[:,:-len(total_gen)].corr(method='pearson')#Hacemos la correlación con las variables iniciales.

corr_mat=abs(corr_mat["imdb_score"]).sort_values()

noncorrelated_index=corr_mat[corr_mat<0.07].index#Extraemos aquellas que tienen baja correlación.

print(noncorrelated_index)

x.drop(noncorrelated_index,axis=1,inplace=True)
np.sum(x.isnull())#número de nans para cada variable
x.corr(method='pearson')["imdb_score"].head()
x_nnan=x.dropna(how='any')#Creamos una variable auxiliar para borrar nan's y poder montar un modelo para

# extapolar gross
from sklearn.ensemble import RandomForestRegressor #libreria del modelo que diseñamos.

dt = RandomForestRegressor()



xtrain=x_nnan.drop(["gross"],axis=1) #variables

ytrain=x_nnan["gross"] #variables respuesta

dt.fit(xtrain,ytrain) #modelo
x.columns.drop("gross")
#Ahora hacemos la perdicción para los NAN's de gross.

x_gross=x[sum([ 1*x[i].isnull() for i in x.columns.drop("gross")])==0]

x_gross["gross"][x_gross["gross"].isnull()]=dt.predict(x_gross.drop(["gross"],axis=1)[x_gross["gross"].isnull()])
x_gross.info() #vemos que ya tenemos una muestra sin NAN'S y mucho mayor

#que la que hubiera resultado de eliminar todos los NAN'S de gross.
xx=x_gross.drop(not_gen_corr_index,axis=1) #Guardamos en xx la base de datos definitiva que vamos a usar.

#y borramos las variables de genero que no tienen gran correlación.
np.random.seed(111)

x_shouffle=xx.sample(frac=1)

x_train=x_shouffle[:-300] #seleccionamos todas menos las 300 finales para train

x_test=x_shouffle[-300:] # seleccionamos las 300 finales para test

y_test=x_test["imdb_score"] # extraemos la variable respuesta

x_test=x_test.drop(["imdb_score"],axis=1)# quitamos la variable respuesta

y_train=x_train["imdb_score"]

x_train=x_train.drop(["imdb_score"],axis=1)
print(x_train.shape[1],x_test.shape[1])
print(x_train.shape[0],x_test.shape[0])
#Ajustamos el predictor a los datos. Utilizamos el modelo RandomForestRegressor.

from sklearn.ensemble import RandomForestRegressor

dt = RandomForestRegressor()

dt.fit(x_train, y_train)

#Calculamos scores.

dt_score_train = dt.score(x_train, y_train)

dt_score_train = dt.score(x_train, y_train)

print("Training score: ",dt_score_train)

dt_score_test = dt.score(x_test, y_test)

print("Testing score: ",dt_score_test)
sum(abs(dt.predict(x_test)-y_test))/len(y_test)#media de los errores.