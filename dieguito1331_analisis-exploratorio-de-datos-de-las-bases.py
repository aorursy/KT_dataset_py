import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

train = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_inicial_train/ib_base_inicial_train.csv")

digital = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_digital/ib_base_digital.csv")

rcc = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_rcc/ib_base_rcc.csv")

reniec = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_reniec/ib_base_reniec.csv")

vehicular = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_vehicular/ib_base_vehicular.csv")
print("# Filas: {0}".format(str(train.shape[0])))

print("# Columnas: {0}".format(str(train.shape[1])))



print("Compruebo tipos de datos de las columnas")

train.dtypes

print("compruebo valores nulos")

train.isna().sum()
variables = ["margen", "cem", "ingreso_neto", "linea_ofrecida"]



for variable in variables:

    print("calculo deciles para variable {0}".format(variable))

    print("----------------------------------")

    dfAux = train.groupby(["codmes", pd.qcut(train[variable], 10, duplicates = "drop"), "codtarget"]).size().reset_index().rename(columns = {0 : "qCasos"})

    crosstab = pd.crosstab(index = [dfAux.codmes, dfAux[variable]], columns = dfAux.codtarget, values = dfAux["qCasos"], aggfunc = "sum").reset_index()

    crosstab["totales"] = crosstab[0]+crosstab[1]

    crosstab["%target0"] = crosstab[0]/crosstab["totales"]

    crosstab["%target1"] = crosstab[1]/crosstab["totales"]

    print(crosstab[crosstab.codmes == 201901])

print("Observamos las proporciones del target")

print(train.groupby("codtarget").size() / len(train))



print("calculamos quienes son rentables y quienes son, margen > 0 es rentable")

train["esRentable"] = np.where(train.margen > 0, 1, 0)

print(train.groupby("esRentable").size() / len(train))



print("vemos quienes son rentables y compraron una TC")

print(train.groupby(["codtarget","esRentable"]).size() / len(train))

variables = ["margen", "cem", "ingreso_neto", "linea_ofrecida"]



for variable in variables:

    print("calculo deciles para variable {0}".format(variable))

    print("----------------------------------")

    dfAux = train.groupby(["codmes", pd.qcut(train[variable], 4, duplicates = "drop"), "codtarget"]).size().reset_index().rename(columns = {0 : "qCasos"})

    crosstab = pd.crosstab(index = [dfAux.codmes, dfAux[variable]], columns = dfAux.codtarget, values = dfAux["qCasos"], aggfunc = "sum").reset_index()

    crosstab["totales"] = crosstab[0]+crosstab[1]

    crosstab["%target0"] = crosstab[0]/crosstab["totales"]

    crosstab["%target1"] = crosstab[1]/crosstab["totales"]

    print(crosstab[crosstab.codmes == 201901])

a = train[train.codtarget == 1].groupby("codmes").agg({"linea_ofrecida": ["min", "max", "mean", "median"], 

                                                   "cem": ["min", "max", "mean", "median"], 

                                                   "ingreso_neto": ["min", "max", "mean", "median"], 

                                                   "margen": ["min", "max", "mean", "median"]})
array = []

for column in a.columns:

    array.append(column[0]+"_"+column[1])

a.columns = array

 

a
train[train.codtarget == 0].groupby("codmes").agg({"linea_ofrecida": ["min", "max", "mean", "median"], 

                                                   "cem": ["min", "max", "mean", "median"], 

                                                   "ingreso_neto": ["min", "max", "mean", "median"], 

                                                   "margen": ["min", "max", "mean", "median"]})
train["ratioLineaIngreso"] = train["linea_ofrecida"] / train["ingreso_neto"]

train.groupby(["codmes", "codtarget"]).agg({"ratioLineaIngreso": ["min", "max", "mean", "median"]})
variables = ["ratioLineaIngreso", "ingreso_neto", "cem"]



for variable in variables:

    print("calculo deciles para variable {0}".format(variable))

    print("----------------------------------")

    dfAux = train.groupby(["codmes", pd.qcut(train[variable], 10, duplicates = "drop"), "codtarget"]).size().reset_index().rename(columns = {0 : "qCasos"})

    crosstab = pd.crosstab(index = [dfAux.codmes, dfAux[variable]], columns = dfAux.codtarget, values = dfAux["qCasos"], aggfunc = "sum").reset_index()

    crosstab["totales"] = crosstab[0]+crosstab[1]

    crosstab["%target0"] = crosstab[0]/crosstab["totales"]

    crosstab["%target1"] = crosstab[1]/crosstab["totales"]

    print(crosstab[crosstab.codmes == 201901])
print("valido la cantidad de veces que una persona aparece en las bases")

print("----------------------------------------------------------------")

a = train.groupby("id_persona").size().reset_index().rename(columns ={0:"qApariciones"})

b = train.groupby("id_persona").codtarget.max().reset_index()

c = pd.merge(a,b, on ="id_persona", how = "left")

print(c[c.qApariciones > 1].groupby("codtarget").size() / len(c[c.qApariciones > 1]))

print("valido segun las apariciones en la base como se comporta segun la venta de los productos")

print("---------------------------------------------------------------------------------------")



print(c.groupby(["codtarget", "qApariciones"]).size() / len(c))



print("valido los porcentajes de target cuando aparece solo una vez el cliente")

print("-----------------------------------------------------------------------")



print(c[c.qApariciones == 1].groupby("codtarget").size() / len(c[c.qApariciones == 1]))
meses = train.codmes.drop_duplicates().values

for codmes in meses:

    print("cantidad de registros para el mes {0} son {1}".format(codmes, len(train[train.codmes == codmes])))

    a = train[train.codmes < codmes].groupby("id_persona").size().reset_index().rename(columns ={0:"qApariciones"})

    b = train[train.codmes == codmes].groupby("id_persona").codtarget.max().reset_index()

    c = pd.merge(b,a, on ="id_persona", how = "left").fillna(0)

    print("valido el paso anterior pero cada uno de los meses separados")

    print("------------------------------------------------------------")

    print(c.groupby(["codtarget", "qApariciones"]).size() / len(c))



    print("con solo la aparicion del mes en curso")

    print("--------------------------------------")

    print(c[c.qApariciones == 0].groupby(["codtarget"]).size() / len(c[c.qApariciones == 0]))

    print("con una aparicion ademas del mes en curso")

    print("-----------------------------------------")

    print(c[c.qApariciones == 1].groupby(["codtarget"]).size() / len(c[c.qApariciones == 1]))



    print("con al menos una aparicion ademas del mes en curso")

    print("--------------------------------------------------")

    print(c[c.qApariciones == 1].groupby(["codtarget"]).size() / len(c[c.qApariciones == 1]))



    



print("revisamos para aquellos que hayan tenido una aparicion anteriormente como se comportan en base a los limites otorgados anteriormente y ahora")

print("el campo mayorLimite es 1 entonces se otorga en el mes en curso mayor limite que en otra aparicion")

print("--------------------------------------------------------------------------------------------------")

meses = train.codmes.drop_duplicates().values

#meses = [201904]

for codmes in meses:

    a = train[train.codmes < codmes].groupby("id_persona").size().reset_index().rename(columns ={0:"qApariciones"})    

    b = train[train.codmes == codmes].groupby("id_persona").agg({"codtarget": "max", "linea_ofrecida": "max"}).reset_index().rename(columns = {"linea_ofrecida": "ultimaLinea"})

    c = train[train.codmes < codmes].groupby("id_persona").linea_ofrecida.max().reset_index()

    d = pd.merge(b,a, on ="id_persona", how = "left").fillna(0)

    e = pd.merge(d, c, on = "id_persona", how = "left").fillna(0)

    e = e[e.qApariciones > 0]

    e["mayorLimite"] = np.where(e.ultimaLinea > e.linea_ofrecida, 1, 0)

    print("como se comporta la cartera mes a mes en base al target y si se aumento o no el limite")

    print("--------------------------------------------------------------------------------------")

    print(e.groupby(["mayorLimite", "codtarget"]).size() / len(e))

    print("como se comportan aquellos que la linea ofrecida disminuye")

    print(e[e.mayorLimite == 0].groupby(["mayorLimite", "codtarget"]).size() / len(e[e.mayorLimite == 0]))

    

    print("como se comportan aquellos que la linea ofrecida aumenta")

    print(e[e.mayorLimite == 1].groupby(["mayorLimite", "codtarget"]).size() / len(e[e.mayorLimite == 1]))

    

   

   
digital["codmes"] = digital.apply(lambda x: x.codday // 100, axis = 1).astype(int)

digital = digital.fillna(0)

agregatedColumns = ['simu_prestamo', 'benefit', 'email', 'facebook',

       'goog', 'youtb', 'compb', 'movil', 'desktop', 'lima_dig', 'provincia_dig', 'extranjero_dig', 'n_sesion',

       'busqtc', 'busqvisa', 'busqamex', 'busqmc', 'busqcsimp', 'busqmill',

       'busqcsld', 'busq', 'n_pag', 'android', 'iphone']



aggFunctions = dict()

for column in agregatedColumns:

    aggFunctions[column] = "sum"



digital = digital.groupby(["codmes", "id_persona"]).agg(aggFunctions).reset_index()

mesesDigital = {

    201901: (201808, 201810),

    201902: (201809, 201811),

    201903: (201810, 201812),

    201904: (201811, 201901),

    201905: (201812, 201902),

    201906: (201901, 201903),

    201907: (201902, 201904)

}
auxDigital = digital[digital.codmes.between (mesesDigital[201902][0], mesesDigital[201902][1])]
for column in auxDigital.columns[2:]:

    print("Columnas con Valores distintos a 0 {0} {1} {2} %".format(column, auxDigital[auxDigital[column] > 0][column].count(), \

                                                             round(auxDigital[auxDigital[column] > 0][column].count() / auxDigital[column].count(),2)))
rcc.head()
rcc.groupby("rango_mora").size()
rcc.groupby("id_persona").producto.size().sort_values(ascending = False)
mesesMora = mesesDigital = {

    201901: (201803, 201809),

    201902: (201804, 201810),

    201903: (201805, 201811),

    201904: (201806, 201812),

    201905: (201807, 201901),

    201906: (201808, 201902),

    201907: (201809, 201903)

}
auxRcc = rcc[rcc.codmes.between (mesesMora[201902][0], mesesMora[201902][1])]
auxRcc.isna().sum()/ auxRcc.count()
auxRcc.rango_mora.min()
auxRcc["rango_mora"] = auxRcc["rango_mora"].fillna(0).astype(int)

moraValues = auxRcc.rango_mora.drop_duplicates().values



auxRccRangoMora = auxRcc.groupby("id_persona").agg({"rango_mora": "max", "mto_saldo": "mean"}).join(auxRcc.groupby("id_persona").cod_banco.nunique())

auxRccRangoMora = auxRccRangoMora.join(auxRcc.groupby("id_persona").producto.nunique())

reniecColumns = reniec.columns[2:-1]

reniecDummies = pd.concat([pd.get_dummies(reniec.set_index("id_persona")[col]) for col in reniecColumns], axis=1, keys=reniecColumns)

reniecDummies.columns = [col[0]+"_"+str(col[1]) for col in reniecDummies.columns]

reniecDummies



reniec.set_index("id_persona").join(reniecDummies).drop(columns = reniecColumns)
vehicular.marca.value_counts().to_frame().sort_values(by = "marca", ascending = False).head(50)
vehicular1 = vehicular.groupby(["id_persona", "marca"]).veh_var1.sum().unstack(level=1, fill_value=0).astype("float32")

vehicular1.columns = [c + "_v1" for c in vehicular1.columns]



vehicular2 = vehicular.groupby(["id_persona", "marca"]).veh_var1.sum().unstack(level=1, fill_value=0).astype("float32")

vehicular2.columns = [c + "_v2" for c in vehicular2.columns]