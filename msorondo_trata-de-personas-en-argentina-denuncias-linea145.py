import pandas as pd

import numpy as np

from IPython.display import display

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



datos17 = pd.read_csv("../input/trata-arg/lucha-contra-la-trata-de-personas-denuncias-linea-145-201712.csv")

 

docs18 = []

docs19 = []

for k in range(12):

    if (k<9):

        docs18.append(pd.read_csv("../input/trata-arg/d20180" + str(k+1) + ".csv"))

        docs19.append(pd.read_csv("../input/trata-arg/lucha-contra-la-trata-de-personas-denuncias-linea-145-20190"+ str(k+1) + ".csv") )

    else:

        docs18.append(pd.read_csv("../input/trata-arg/d2018" + str(k+1) + ".csv"))

        docs19.append(pd.read_csv("../input/trata-arg/lucha-contra-la-trata-de-personas-denuncias-linea-145-2019"+ str(k+1) + ".csv") )



datos18 = pd.concat(docs18,sort=False)

datos19 = pd.concat(docs19,sort=False)

datos1819 = pd.concat([datos18,datos19],sort=False)#dadas ciertas similitudes entre éstos dos años defino un dataset con las denuncias del 2018-2019

datos_todos = pd.concat([datos17,datos18,datos19], sort=False)
denuncias_año = pd.DataFrame({"denuncias":[len(datos17.index),len(datos18.index),len(datos19.index)]}, index=[2017,2018,2019])

denuncias_año.rename_axis("año", axis=0)

datos_todos["denuncia_fecha"].value_counts().sort_index().head(32)

datos_todos = datos_todos.sort_values("denuncia_fecha")

datos_todos.iloc[162:,]

denuncias_mes_17 = list(range(0,12,1))

datos17["denuncia_fecha"] = pd.to_datetime(datos17["denuncia_fecha"].sort_values().values)



for k in datos17["denuncia_fecha"]:

    denuncias_mes_17[k.month-1] = denuncias_mes_17[k.month-1]+1

denuncias_mes_17_df = pd.DataFrame(denuncias_mes_17, index = range(1,13,1)).rename_axis("mes, 2017",axis="rows").rename_axis(2017,axis="columns")

denuncias_mes_18 = []

for k in range(0,12,1):

    denuncias_mes_18.append(docs18[k]["denuncia_fecha"].value_counts().sum())

denuncias_mes_18

denuncias_mes_18_df = pd.DataFrame(denuncias_mes_18, index = range(1,13,1)).rename_axis("mes",axis="rows").rename_axis(2018,axis="columns")

denuncias_mes_19 = []

for k in range(0,12,1):

    denuncias_mes_19.append(docs19[k]["denuncia_fecha"].value_counts().sum())

denuncias_mes_19

denuncias_mes_19_df = pd.DataFrame(denuncias_mes_19, index = range(1,13,1)).rename_axis("mes",axis="rows").rename_axis(2019,axis="columns")

plt.figure(figsize=(15,10))

p121 = sns.lineplot(data=denuncias_mes_17_df)

plt.title("Denuncias por mes, año 2017")

plt.ylabel("Número de denuncias")



plt.figure(figsize=(15,10))

p122 = sns.lineplot(data=denuncias_mes_18_df)


plt.figure(figsize=(15,10))

p123 = sns.lineplot(data=denuncias_mes_19_df)
sns.lineplot(data=denuncias_mes_17_df)

sns.lineplot(data=denuncias_mes_18_df)

sns.lineplot(data=denuncias_mes_19_df)

#rename axis
meses = []



def func(anio):

    for k in range(1,13,1):

        if (k<10):

            meses.append( str(anio) +"-0" + str(k))

        else:

            meses.append(str(anio) + "-" + str(k))

    return meses

        

func(2017)

func(2018)

func(2019)



meses

denuncias_meses = pd.Series(denuncias_mes_17+denuncias_mes_18+denuncias_mes_19, index=meses)

len(denuncias_meses)



plt.figure(figsize=(40,20))

sns.lineplot(data=denuncias_meses)

plt.title("Número de denuncias por mes")

plt.ylabel("Denuncias")

plt.xlabel("Fecha")
pd.Series(denuncias_mes_17).mean()

pd.Series(denuncias_mes_18).mean()

pd.Series(denuncias_mes_19).mean()

k = datos17.groupby(["hecho_provincia","hecho_localidad"]).hecho_pais.value_counts()

k.loc["Buenos Aires"," LA PLATA"]


datos17["hecho_provincia"] = datos17["hecho_provincia"].fillna("No Aplica").replace(["Unknown","No aplica"],["No Aplica","No Aplica"])

#Primero con las provincias...

datos17 = datos17.replace(["Mendoza ","SALTA"], ["Mendoza","Salta"])

np.sort(pd.unique(datos17["hecho_provincia"].values))

porProvincia = datos17["hecho_provincia"].value_counts().sort_values(ascending=False)

#voy a limpiar los casos de denuncias donde el hecho es en paises extranjeros



porProvincia_soloArg = porProvincia[porProvincia.index!="No Aplica"]

porProvincia_soloArg = porProvincia_soloArg.sort_index()

plt.figure(figsize=(40,10))

sns.barplot(x=porProvincia_soloArg.index,y=porProvincia_soloArg.values)
dicc_poblacion_porprovincia = {"Caba":2890151,

"Buenos Aires":15625084,

"Catamarca": 367828,

"Chaco":1055259,

"Chubut": 509108,

"Córdoba":3308876,

"Corrientes":992595,

"Entre Ríos":1235994,

"Formosa":530162,

"Jujuy":673307,

"La Pampa":318951,

"La Rioja":333642,

"Mendoza":1738929,

"Misiones":1101593,

"Neuquén":551266,

"Río Negro":638645,

"Salta":1214441,

"San Juan":681055,

"San Luis":432310,

"Santa Cruz":273964,

"Santa Fe":3194537,

"Santiago del Estero":874006,

"Tierra del Fuego":127205,

"Tucumán":1448188,

}

provs_poblacion = pd.Series(dicc_poblacion_porprovincia).sort_index()

provs_poblacion = provs_poblacion.sort_index()

provs_poblacion
hechosypobl = pd.DataFrame([porProvincia_soloArg,provs_poblacion]).T



hechosypobl = hechosypobl.rename(columns={"Unnamed 0":"poblacion"})

hechosypobl

tasadenuncias_serie = hechosypobl.hecho_provincia/hechosypobl.poblacion

tasadenuncias_df = pd.DataFrame(tasadenuncias_serie).rename(columns={0:"Tasa de denuncias por población"})

tasadenuncias_df = tasadenuncias_df.sort_values(by="Tasa de denuncias por población", ascending=False)

tasadenuncias_df

plt.figure(figsize=(40,10))

sns.barplot(x=tasadenuncias_df.index,y=tasadenuncias_df["Tasa de denuncias por población"].values)
#luego con las localidades

datos17["hecho_localidad"] = datos17["hecho_localidad"].fillna("NaN")

datos17 = datos17.replace(["ALMAGRO ",'ALEN /MISíONES',"ADROGUÉ"," BARRIO FLORES", " CERRITO", " CIUDAD DE VILLA ABREGÚ",' CONCEPCION DEL URUGUAY', ' Caba', ' EL CHAÑAR',

       ' INGENIERO ALDOLFO SOURDEAUX', ' LA PLATA',

       ' PARTIDO ESTEBAN ECHEVERRÍA',],

                ["ALMAGRO",'ALEM - MISíONES',"ADROGUE","BARRIO FLORES","CERRITO", "CIUDAD DE VILLA ABREGÚ",'CONCEPCION DEL URUGUAY', 'Caba', 'EL CHAÑAR',

       'INGENIERO ALDOLFO SOURDEAUX', 'LA PLATA',

       'PARTIDO ESTEBAN ECHEVERRÍA',])

datos17["hecho_localidad"] = (datos17["hecho_localidad"].str).lstrip().str.rstrip()

datos17["hecho_localidad"] = datos17["hecho_localidad"].str.replace(r'\s\s+', ' ')

datos17["hecho_localidad"].value_counts()
condicion_victima = datos1819.apply(lambda x: "victima" in x.name ,axis=0)

datos1819[datos1819.columns[condicion_victima]]
condicion_edad = datos_todos.apply(lambda x: "etario" in x.name ,axis=0)

datos_edades_df = datos_todos[datos_todos.columns[condicion_edad]]

datos_edades_df = datos_edades_df.fillna("Sin Datos")

datos_edades_df["victima_rango_etario"].value_counts()
victima_poredad_ordered = pd.DataFrame(datos18["victima_rango_etario"].value_counts(), index = ['0 a 13', 

                                                                                                '14 a 16','16 a 17 (específico para trabajo adolescente)', 

                                                                                                '18 a 25','26 a 40','41 a 60',

       

         '61 en adelante', 'Sin datos','No refiere'],

      dtype='object')

victima_poredad_ordered = victima_poredad_ordered.rename_axis("Edad aproximada")

victima_poredad_ordered.rename_axis(2018,axis="columns")


datos19["victima_identificada_rango_etario"].value_counts().unique()
#en el año 2017 no se registró el género de las víctimas

pd.DataFrame(datos18["victima_genero"].value_counts()+datos18["victima_identificada_genero"].value_counts()).sort_values(by=0,ascending=False).rename(columns={0:"Numero"}).head(5)
datos1819["victima_discapacidad"] = datos1819["victima_discapacidad"].fillna("Sin datos")

datos1819["victima_discapacidad"].value_counts().rename_axis("Discapacidad", axis="rows")
datos1819["victima_embarazada"] = datos1819["victima_embarazada"].fillna("Sin datos")

datos1819["victima_embarazada"].value_counts().rename_axis("¿Embarazada?", axis="rows")
datos18["llamante_anonimo"].value_counts()



datos19["llamante_anonimo"] = datos19["llamante_anonimo"].replace(["Anónima", "No ", "sí"],

                                                                  ["Sí","No","Sí"])

display(datos18["llamante_anonimo"].value_counts().rename_axis(2018),

        datos19["llamante_anonimo"].value_counts().rename_axis(2019))
datos_todos["llamante_es_victima"] = datos_todos["llamante_es_victima"].replace(["No es Víctima", "No es víctima","No es víctima","no es Víctima","no es víctima","Víctima Directa","VÍCTIMA DIRECTA" ],["No","No","No","No","No","Sí","Sí"])

datos_todos["llamante_es_victima"].value_counts().rename_axis("¿Es víctima?")

(datos17["hecho_denuncia_previa"]).value_counts()

denunciados_previa_2018 = datos18["denuncia_previa"].value_counts()

tot18 = denunciados_previa_2018.sum()

denunciados_previa_2018

porcentaje_denunciados_previamente18 = str(denunciados_previa_2018.loc["Sí"]*100/tot18) + "% de casos denunciados previamente"

porcentaje_denunciados_previamente18

porcentaje_casos_nuevos_18 = str(100-(denunciados_previa_2018.loc["Sí"]*100/tot18)) + "% de casos nuevos en 2018"

porcentaje_casos_nuevos_18 
datos19["denuncia_previa"].value_counts()

nuevas19 = len(datos19["denuncia_previa"][(datos19["denuncia_previa"]=="Nueva") 

                                          | (datos19["denuncia_previa"]=="nueva") |

                                   (datos19["denuncia_previa"]=="NUEVA") 

                                          | (datos19["denuncia_previa"]=="nUEVA")])

tot19 = datos19["denuncia_previa"].value_counts().sum()

porcentaje_casos_relacionados_19 = str((tot19-nuevas19)*100/tot19) + "% de casos relacionados"

porcentaje_casos_relacionados_19

porcentaje_casos_nuevos_19 = str(nuevas19*100/tot19 ) + "% de casos nuevos en 2019"

porcentaje_casos_nuevos_19
display(datos1819['denuncia_via_ingreso'].value_counts(),datos1819['acercamiento_tipo'].value_counts())
condicion_llamante = datos1819.apply(lambda x: "llamante" in x.name ,axis=0)

datos1819[datos1819.columns[condicion_llamante]]
condicion_denunciado = datos_todos.apply(lambda x: "denunciado" in x.name,axis=0)

df_denunciados = (datos_todos[datos_todos.columns[condicion_denunciado]]).fillna("Sin datos")

df_denunciados
df_denunciados["denunciado_provincia"] = df_denunciados["denunciado_provincia"].replace(["BUENOS AIRES", "CABa","CHACO","CORRIENTES","Caba","Ciudad Autónoma de Buenos Aires","CÓRDOBA","JUJUY","MENDOZA","MISIONES"],["Buenos Aires","CABA","Chaco","Corrientes","CABA","CABA","Córdoba","Jujuy","Mendoza","Misiones"])

df_denunciados["denunciado_provincia"].value_counts().sort_index()
df_denunciados["denunciado_genero"].value_counts()
df_denunciados["denunciado_rango_etario"].value_counts()
condicion_complicidad = datos_todos.apply(lambda x: "connivencia" in x.name,axis=0)

df_complicidad = (datos_todos[datos_todos.columns[condicion_complicidad]]).fillna("Sin datos")

df_complicidad = df_complicidad.replace(["NO","SÍ","no","sí","nO","No Refiere","No Refiere "],["No","Sí","No","Sí","No","Sin datos","Sin datos"])

numeros_complicidad_seguridad = df_complicidad.iloc[:,0].value_counts().rename_axis("Complicidad (fuerzas de seguridad)")

numeros_complicidad_politicos = df_complicidad.iloc[:,1].value_counts().rename_axis("Complicidad (poder político)")

display(numeros_complicidad_seguridad,numeros_complicidad_politicos)

sns.barplot(x=numeros_complicidad_seguridad.index,y=numeros_complicidad_seguridad.values)

sns.barplot(x=numeros_complicidad_politicos.index,y=numeros_complicidad_politicos.values)
series_tipo_explotacion = datos_todos["denuncia_tipo_explotacion"].value_counts().sort_values(ascending=False)

series_tipo_explotacion