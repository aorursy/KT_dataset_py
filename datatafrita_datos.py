import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

plt.style.use('default') # haciendo los graficos un poco mas bonitos en matplotlib
#plt.rcParams['figure.figsize'] = (20, 10)

sns.set(style="whitegrid") # seteando tipo de grid en seaborn
types= {
    "event": "category", 
    "model": "category", 
    "condition": "category", 
    "color": "category", 
    "storage": "category"
}
df_events = pd.read_csv("events.csv", dtype=types, low_memory=False)


models = df_events[["person","model","storage","event","color","condition"]]
models = models.loc[models["event"] == "conversion" ,:]
#models

df = df_events.loc[(df_events["country"].isnull() == False) & (df_events["country"] != "Unknown"),:]
df = df[['person','country','region']]

df = df.drop_duplicates(subset='person')


#df2 = df[["person","region","country"]]
models = models.merge(df, on='person')
#coalese o fillna
#models
#s16_GB = models.loc[models["storage"] == "16GB"]
s16_GB = models["country"].value_counts()
s16_GB
brazil = models.loc[models["country"] == "Brazil"]
sbrazil = brazil["storage"].value_counts()
sbrazil.index = [16,32,8,64,128,4,0.512,256]
sbrazil = sbrazil.sort_index()
sbrazil
uk = models.loc[models["country"] == "United Kingdom"]
uk = uk["storage"].value_counts()
uk.index = [16,8,64,0.512,4,32,256,128]
uk = uk.sort_index()
uk

eeuu = models.loc[models["country"] == "United States"]
eeuu = eeuu["storage"].value_counts()
eeuu.index = [16,8,64,0.512,4,32,256,128]
eeuu = eeuu.sort_index()
eeuu
nombres = ["512MB","4GB","8GB","16GB","32GB","64GB","128GB","256GB"]

X = np.arange (len(nombres))
Y1 = sbrazil.values
Y2 = uk.values
Y3 = eeuu.values

plt.axes([0.025,0.025,1.5,1.5])
plt.bar(X-0.15 ,Y1,facecolor = "#28A820",label = "Brasil", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X,Y2,facecolor = "#0C11BB",label = "Reino Unido", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X+0.15 ,Y3,facecolor = "#FF0017",label = "Estados Unidos", width = 0.15, align = "center",edgecolor = "white")



plt.xticks(X,nombres,fontsize = 10)
plt.xlabel("Almacenamiento interno",fontsize = 18)
plt.ylabel('Cantidad de compras', fontsize = 18)

plt.title('Cantidad de compras de los disntintos almacenamiento en los distintos paises\n', fontsize = 18)

plt.legend(loc="upper right", fontsize = 12)
plt.show()
#brazil = models.loc[(models["country"] == "Brazil")&(models["region"] != "Unknown")&(models["event"] == "conversion" )]
#sao_pablo = brazil.loc[brazil["region"] == "Sao Paulo"]
brazil = brazil.loc[brazil["region"] != "Unknown"]
top_5_regiones = brazil["region"].value_counts().head()
top_5_regiones
sao_pablo = brazil.loc[(brazil["region"] == "Sao Paulo")]
sao_pablo = sao_pablo["storage"].value_counts()
sao_pablo.index = [16,32,8,64,128,0.512,4,256]
sao_pablo = sao_pablo.sort_index()
sao_pablo
minas_gerais  = brazil.loc[brazil["region"] == "Minas Gerais"]
minas_gerais = minas_gerais["storage"].value_counts()
minas_gerais.index = [16,32,8,64,128,0.512,4,256]
minas_gerais = minas_gerais.sort_index()
minas_gerais
rio  = brazil.loc[brazil["region"] == "Rio de Janeiro"]
rio = rio["storage"].value_counts()

rio.index = [16,32,8,64,4,0.512,256,128]
rio = rio.sort_index()
rio
bahia  = brazil.loc[brazil["region"] == "Bahia"]
bahia = bahia["storage"].value_counts()
bahia.index = [16,8,32,64,0.512,4,128,256]
bahia  = bahia.sort_index()
bahia 
maranhao  = brazil.loc[brazil["region"] == "Maranhao"]
maranhao = maranhao["storage"].value_counts()
maranhao.index = [16,32,4,8,64,0.512,256,128]
maranhao  = maranhao.sort_index()
maranhao 
nombres = ["512MB","4GB","8GB","16GB","32GB","64GB","128GB","256GB"]

X = np.arange (len(nombres))
Y1 = sao_pablo.values
Y2 = minas_gerais.values
Y3 = bahia.values
Y4 = rio.values
Y5 = maranhao.values



plt.axes([0.025,0.025,1.5,1.5])
plt.bar(X-0.3 ,Y1,facecolor = "#000000",label = "Sao Paulo", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X-0.15,Y2,facecolor = "#ff0000",label = "Minas Gerais", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X ,Y3,facecolor = "#0a14c8",label = "Bahia", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X+0.15,Y4,facecolor = "#4795e0",label = "Rio de Janiero", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X+0.3 ,Y5,facecolor = "#fcf10c",label = "Maranhao", width = 0.15, align = "center",edgecolor = "white")


plt.xticks(X,nombres,fontsize = 10)
plt.xlabel("Almacenamiento interno",fontsize = 18)
plt.ylabel('Cantidad de compras', fontsize = 18)

plt.title('Comparacion el almacenamiento mas vendidos en las regiones mas polulares de Brasil\n', fontsize = 18)

plt.legend(loc="upper right",fontsize = 12)
plt.show()


# importacion general de librerias y de visualizacion (matplotlib y seaborn)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

plt.style.use('default') # haciendo los graficos un poco mas bonitos en matplotlib
#plt.rcParams['figure.figsize'] = (20, 10)

sns.set(style="whitegrid") # seteando tipo de grid en seaborn
types= {
    "event": "category", 
    "model": "category", 
    "condition": "category", 
    "color": "category", 
    "storage": "category"
}
df_events = pd.read_csv('./fiuba-trocafone-tp1-final-set/events.csv', dtype=types, low_memory=False)
#Cantidad de campaign y de search engine
filter_colum = df_events[["person", "event", "campaign_source", "search_engine", "timestamp"]]
filter_colum
hit_contains = filter_colum.loc[filter_colum["event"].str.contains("hit")]
hit_contains
campaing = hit_contains.loc[filter_colum["event"] == "ad campaign hit"]
search = hit_contains.loc[filter_colum["event"] == "search engine hit"]
campaing["event"].value_counts()
search["event"].value_counts()
group = filter_colum.groupby(["person", "event"])
group.count()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

plt.style.use('default') # haciendo los graficos un poco mas bonitos en matplotlib
#plt.rcParams['figure.figsize'] = (20, 10)

sns.set(style="whitegrid") # seteando tipo de grid en seaborn
types= {
    "event": "category", 
    "model": "category", 
    "condition": "category", 
    "color": "category", 
    "storage": "category"
}
df_events = pd.read_csv("events.csv", dtype=types, low_memory=False)
galaxy_j5 = df_events.loc[(df_events["model"] == "Samsung Galaxy J5") & (df_events["event"] == "conversion"),:]
galaxy_j5 = galaxy_j5[["city","region","country","event","timestamp","condition"]]
galaxy_j5.head()
condition = galaxy_j5["condition"].value_counts()

ax = condition.plot(kind = 'bar', title = "Cantidad de compras vs Condicion del smatphone",figsize =(6,4),legend = False, fontsize = 12)
ax.set_xlabel("Condicion del smatphone", fontsize = 18)
ax.set_ylabel("Cantidad de compras",fontsize = 18)
ax.set_xticklabels(["Bueno","Muy Bueno","Excelente","Nuevo"],rotation = "horizontal")
plt.show()
#dias
galaxy_j5['dia_mes'] = galaxy_j5['timestamp'].apply(lambda x: x.split(' ')[0][-5:])
#galaxy_j5.sort_values(by = galaxy)
fechas = galaxy_j5['dia_mes'].value_counts()
#fechas  = fechas.to_frame(name = "Fechas").reset_index()
fechas
#fechas.columns= [["Fecha","Frecuencia"]]
#fechas.sort_values(fechas["Fecha"])
fechas
#fechas.sort(fechas.index)

fechas = galaxy_j5.sort_values(by = "timestamp",ascending = True)["timestamp"]
meses=pd.to_datetime(fechas).dt.month
dias=pd.to_datetime(fechas).dt.day
serie_final= meses.map(lambda x: str(x))+'-'+dias.map(lambda x: str(x) if x>9 else "0"+str(x))
serie_final = serie_final.value_counts().sort_index()



#galaxy_j5["mes"] = pd.to_datetime(galaxy_j5["timestamp"]).dt.month
#galaxy_j5["dia"] = pd.to_datetime(galaxy_j5["timestamp"]).dt.day
#galaxy_j5
ax = serie_final.plot(kind = 'bar', title = "Cantidad de compras vs Condicion del smatphone",figsize =(30,5),legend = False, fontsize = 14)

#plt.rc('figure', dpi=350)
#g=sns.barplot(x=serie_final.index,y=serie_final.values)
#g.set_xticklabels(g.get_xticklabels(), rotation = 90)
plt.show()

serie_final
serie_final_t = serie_final.to_dict()
serie_final_t
fin = False
for i in range (1,7):
    if fin:
        break
    for j in range (1,32):
        if j <10:
            fecha = str(i)+'-0'+str(j)
        else:
            fecha = str(i)+'-'+str(j)
        
        if i == 2 and j <= 29:
            break
        if i in [4,6] and j == 31:
            break
        if i == 6 and j >= 10:
            fin = True
            break
        if not fecha in serie_final_t:
            serie_final_t[fecha] = 0

serie_final_t = pd.Series(serie_final_t)
serie_final_t
ax = serie_final_t.plot(kind = 'line', title = "Cantidad de compras vs Condicion del smatphone",figsize =(20,10),legend = False, fontsize = 8)
lista = serie_final_t.index.tolist()

plt.show()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

plt.style.use('default') # haciendo los graficos un poco mas bonitos en matplotlib
#plt.rcParams['figure.figsize'] = (20, 10)

sns.set(style="whitegrid") # seteando tipo de grid en seaborn
types= {
    "event": "category", 
    "model": "category", 
    "condition": "category", 
    "color": "category", 
    "storage": "category"
}
df_events = pd.read_csv("events.csv", dtype=types, low_memory=False)
#df_events.columns
#df_events.head()

models = df_events.loc[df_events["storage"].isnull() == False,:]
models = models[["model","storage","event","color","condition"]]
models = models.loc[models["event"] == "conversion" ,:]
models.head()
models.count()
storage = models["storage"].value_counts()
storage.index = [16,32,64,8,128,4,256,0.512]
storage = storage.sort_index()
#storage.index = ["256GB","128GB","64GB","32GB","16GB","8GB","4GB","512MB"]
storage
ax = storage.plot(kind = 'bar', title = "Cantidad de compras vs Almacenamiento interno",figsize =(10,8),legend = False, fontsize = 12)
ax.set_xlabel("Almacenamiento interno", fontsize = 18)
ax.set_ylabel("Cantidad de compras",fontsize = 18)
ax.set_xticklabels(["512MB","4GB","8GB","16GB","32GB","64GB","128GB","256GB"], rotation = "horizontal")
plt.show()
model = models["model"].value_counts()
model = model.head(20)
model.head(20)
ax = model.plot(kind = 'barh', title = "Models vs compras",figsize =(10,8),legend = False, fontsize = 10)
ax.set_xlabel("Cantidad de compras",fontsize = 18)
ax.set_ylabel("Modelos",fontsize = 18)
plt.show()
color = models["color"].value_counts()
color = color.head(30)
color
colores = {"Preto" : "Negro", "Dourado" : "Dorado", "Branco" : "Blanco", "Cinza espacial" : "Negro", "Prateado" : "Plata", "Ouro Rosa" : "Rosa",
"Rosa": "Rosa", "Cinza" : "Plata", "Azul" : "Azul", "Preto Vermelho" : "Negro", "Prata" : "Plata", "Platinum" : "Plata", "Preto Matte" : "Negro",
"Branco Vermelho" : "Blanco", "Ouro" : "Dorado", "Titânio" : "Plata", "Ametista" : "Otros", "Preto Brillhante" : "Negro", "Indigo" : "Otros",
"Amarelo" : "Otros", "Vermelho" : "Otros", "Bambu" : "Otros", "Cabernet" : "Otros", "Preto Azul" : "Negro", "Couro Vintage" : "Otros", "Azul Topázio" : "Azul"}

models["in_color"] = models['color'].apply(lambda x: colores.get(x, 'Basura'))

models = models.loc[models['in_color'] != 'Basura']
colores = models['in_color'].value_counts()
colores

#colores = {"Negro": color["Preto"] + color["Cinza espacial"] + color["Preto Matte"] + color["Preto Brilhante"] ,
#           "Plata": color["Prateado"] + color["Prata"] + color["Platinum"] + color["Cinza"] + color["Titânio"] ,
#           "Dorado": color["Ouro"] + color["Dourado"],
#            "Rosa": color["Ouro Rosa"] + color["Rosa"],
#           "Azul": color["Azul"], "Otros" : color["Ametista"] + color["Vermelho"] +  color["Bambu"] +  color["Preto Vermelho"] + color["Verde"]}
#colores

#plt.bar(range(len(colores)), list(colores.values()))
#plt.xticks(range(len(colores)), list(colores.keys()))
         
ax = colores.plot(kind = 'bar', title = "Colores de smartphones vs Cantidad de compras",figsize =(10,8),legend = False, rot=0, fontsize = 12,color = [["#000000","#DCC6AE","#F6F7F9","#DFE0E2","#F1C1BD" ,"#3F5796","#C72233"]])
ax.set_xlabel("Colores",fontsize = 18)
ax.set_ylabel("Cantidad de compras",fontsize = 18)
#ax.set_xticklabels(ax.getaxis.rotation = "horizontal")


plt.show()
condition = models["condition"].value_counts()
condition
ax = condition.plot(kind = 'bar', title = "Cantidad de compras vs Condicion del smatphone",figsize =(10,8),legend = False, fontsize = 12)
ax.set_xlabel("Condicion del smatphone", fontsize = 18)
ax.set_ylabel("Cantidad de compras",fontsize = 18)
ax.set_xticklabels(["Bueno","Muy Bueno","Excelente","Bueno - sin Touch ID","Nuevo"],rotation = "horizontal")
plt.show()

# importacion general de librerias y de visualizacion (matplotlib y seaborn)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

plt.style.use('default') # haciendo los graficos un poco mas bonitos en matplotlib
#plt.rcParams['figure.figsize'] = (20, 10)

sns.set(style="whitegrid") # seteando tipo de grid en seaborn
types= {
    "event": "category", 
    "model": "category", 
    "condition": "category", 
    "color": "category", 
    "storage": "category"
}
df_events = pd.read_csv('./fiuba-trocafone-tp1-final-set/events.csv', dtype=types, low_memory=False)
df_events.columns
df_events.head()
#Ranking del sitio web que utiliza el usuario para ingresar a Trocafone 
trafico = df_events.groupby('campaign_source').agg({'person': 'count'})
trafico.reset_index(inplace=True)
trafico.columns =['campaing_source','count']
ax = trafico.plot(kind = 'bar', title = "Cantidad vs Sitio web",figsize =(10,8),legend = False, fontsize = 12,  color=[plot.cm.Paired(np.arange(len(trafico)))])
ax.set_xlabel("Sitio web",fontsize = 18)
ax.set_ylabel("Log de cantidad de ingresos",fontsize = 18)
ax.set_xticklabels(trafico['campaing_source'].values)
plt.show()
trafico['count'] = np.log(trafico['count'])
ax = trafico.plot(kind = 'bar', title = "Cantidad vs Sitio web",figsize =(10,8),legend = False, fontsize = 12)
ax.set_xlabel("Sitio web",fontsize = 18)
ax.set_ylabel("Log de cantidad de ingresos",fontsize = 18)
ax.set_xticklabels(trafico['campaing_source'].values)
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

plt.style.use('default') # haciendo los graficos un poco mas bonitos en matplotlib
#plt.rcParams['figure.figsize'] = (20, 10)

sns.set(style="whitegrid") # seteando tipo de grid en seaborn
types= {
    "event": "category", 
    "model": "category", 
    "condition": "category", 
    "color": "category", 
    "storage": "category"
}
df_events = pd.read_csv("events.csv", dtype=types, low_memory=False)
df = df_events[["person","city","region","country"]]
df = df.loc[(df["city"].isnull() == False) & (df["city"] != "Unknown"),:]

df = df.drop_duplicates(subset='person')

#_df = df_events.loc[(df_events['person'].isin(usuarios)) & (df_events["city"].isnull() == False) & (df_events["city"] != "Unknown") ] 

df2 = df_events[["person","search_term","sku","event","model"]]
dfs = df2.merge(df, on='person')
#coalese o fillna
dfs["country"].value_counts()
dfs = dfs.loc[dfs["search_term"].isnull() == False]
dfs["search_term"] = dfs['search_term'].apply(lambda x: x.lower())
dfs["search_term"].value_counts().head(20)
search_term = dfs["search_term"].value_counts().head(20)
ax = search_term.plot(kind = 'bar', title = "Cantidad de Search Term",figsize =(10,6),legend = False, fontsize = 12)
ax.set_xlabel("Tipos de Search terms", fontsize = 18)
ax.set_ylabel("Cantidad",fontsize = 18)


ax.set_xticklabels(["iPhone 6","iPhone","iPhone 6s","iPhone 7","iPhone 5s","Motorola","J5","J7","iPhone 6 Plus","S7","S8","iPhone SE","Samsung","iPhone 5","iPhone 6s Plus","iPhone 7 Plus","S6","J7 Prime","Moto G5","Moto G"])
plt.show()
dfs_brazil = dfs.loc[dfs["country"] == "Brazil"]
#dfs_brazil["search_term"] = dfs_brazil['search_term'].apply(lambda x: x.lower())
dfs_brazil["search_term"].value_counts().head(20)

dfs_usa = dfs.loc[dfs["country"] == "United States"]
dfs_usa["search_term"].value_counts().head(10)
search_term_usa = dfs_usa["search_term"].value_counts().head(10)
ax = search_term_usa.plot(kind = 'bar', title = "Cantidad de Search Term de Estados Unidos",figsize =(10,6),legend = False, fontsize = 12)
ax.set_xlabel("Tipos de Search terms", fontsize = 18)
ax.set_ylabel("Cantidad",fontsize = 18)

ax.set_xticklabels(["Note 3","Note","Samsung","iPhone 5s", "Motorola","Sony","iPhone 6","iPhone","Moto G4 Plus","Samung Celular"])

plt.show()
dfs["region"].value_counts().head()

dfs_sp = dfs.loc[dfs["region"] == "Sao Paulo"]
sao_paulo = dfs_sp["search_term"].value_counts().head(10)

lista = sao_paulo.index.tolist()
sao_paulo = sao_paulo.sort_index()
sao_paulo
dfs_mn = dfs.loc[(dfs["region"] == "Minas Gerais")&(dfs["search_term"].isin(lista))]
minas_gerais = dfs_mn["search_term"].value_counts().head(10).sort_index()
minas_gerais
dfs_rj = dfs.loc[(dfs["region"] == "Rio de Janeiro")&(dfs["search_term"].isin(lista))]
rio_de_janeiro = dfs_rj["search_term"].value_counts().head(10).sort_index()
rio_de_janeiro
dfs_ba = dfs.loc[(dfs["region"] == "Bahia")&(dfs["search_term"].isin(lista))]
bahia = dfs_ba["search_term"].value_counts().head(10).sort_index()
bahia
dfs_per = dfs.loc[(dfs["region"] == "Pernambuco")&(dfs["search_term"].isin(lista))]
pernambuco = dfs_per["search_term"].value_counts().head(10).sort_index()
pernambuco
X = np.arange (len(lista))
Y1 = sao_paulo.values
Y2 = minas_gerais.values
Y3 = bahia.values
Y4 = rio_de_janeiro.values
Y5 = pernambuco.values


plt.axes([0.025,0.025,1.5,1.5])
plt.bar(X-0.3 ,Y1,facecolor = "#000000",label = "Sao Paulo", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X-0.15,Y2,facecolor = "#ff0000",label = "Minas Gerais", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X ,Y3,facecolor = "#0a14c8",label = "Bahia", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X+0.15,Y4,facecolor = "#4795e0",label = "Rio de Janiero", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X+0.3 ,Y5,facecolor = "#fcf10c",label = "Pernambuco", width = 0.15, align = "center",edgecolor = "white")

nombres = ["iPhone","iPhone 5s","iPhone 6","iPhone 6s","iPhone 7","iPhone 7 Plus","iPhone SE","J7","Motorola","Samsung"]

plt.xticks(X,nombres,fontsize = 10)
plt.xlabel("Terminos de busqueda",fontsize = 18)
plt.ylabel('Cantidad de apariciones', fontsize = 18)

plt.title('Cantidad de los terminos de busquedas en las 5 regiones mas populares de Brasil\n', fontsize = 18)

plt.legend(loc="upper right",fontsize = 12)
plt.show()
dfs_c = dfs.loc[(dfs["event"] == "conversion")&(dfs["country"] == "Brazil")]
models = dfs_c["region"].value_counts().head(10)
models
models_sp = dfs_c.loc[dfs_c["region"] == "Sao Paulo"]
sao_paulo = models_sp["model"].value_counts().head(10)

"""modelos = pd.Series( [0,0,0,0,0,0,0,0,0,0],
    index = ['Samsung Galaxy J5', 'Samsung Galaxy J7', 'Samsung Galaxy S5', 'Samsung Galaxy S5', 'Samsung Galaxy S6 Edge', 'iPhone 5c', 'iPhone 5s', 'iPhone 6',
     'iPhone 6S', 'iPhone SE'])"""
lista = sao_paulo.index.tolist()
sao_paulo = sao_paulo.sort_index()
sao_paulo
models_mn = dfs_c.loc[(dfs_c["region"] == "Minas Gerais")&(dfs_c["model"].isin(lista))]
minas_gerais = models_mn["model"].value_counts().head(8)

modelos = pd.Series( [0,0],
    index = ["iPhone 5c","iPhone SE"])
minas_gerais  = minas_gerais.append(modelos)
minas_gerais  = minas_gerais.sort_index()
minas_gerais 

models_rj = dfs_c.loc[(dfs_c["region"] == "Rio de Janeiro")&(dfs_c["model"].isin(lista))]
rio_de_janeiro = models_rj["model"].value_counts().head(7)

modelos = pd.Series( [0,0,0],
    index = ["Samsung Galaxy J7 Prime","Samsung Galaxy J7","Samsung Galaxy S6 Edge"])

rio_de_janeiro = rio_de_janeiro.append(modelos)
rio_de_janeiro = rio_de_janeiro.sort_index()
rio_de_janeiro
models_ba = dfs_c.loc[(dfs_c["region"] == "Bahia")&(dfs["model"].isin(lista))]
bahia = models_ba["model"].value_counts().head(7)

modelos = pd.Series( [0,0,0],
    index = ["Samsung Galaxy J7 Prime",'iPhone 6S', 'iPhone SE'])

bahia = bahia.append(modelos)
bahia = bahia.sort_index()
bahia
models_per = dfs_c.loc[(dfs_c["region"] == "Pernambuco")&(dfs["model"].isin(lista))]
pernambuco = models_per["model"].value_counts().head(3)

modelos = pd.Series( [0,0,0,0,0,0,0],
    index = ['Samsung Galaxy J5', 'Samsung Galaxy J7', 'Samsung Galaxy S5', 'Samsung Galaxy S5', 'Samsung Galaxy S6 Edge',
     'iPhone 6S', 'iPhone SE'])


pernambuco = pernambuco.append(modelos)
pernambuco = pernambuco.sort_index()
pernambuco
X = np.arange (len(lista))
Y1 = sao_paulo.values
Y2 = minas_gerais.values
Y3 = bahia.values
Y4 = rio_de_janeiro.values
Y5 = pernambuco.values



plt.axes([0.025,0.025,1.5,1.5])
plt.bar(X-0.3 ,Y1,facecolor = "#000000",label = "Sao Paulo", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X-0.15,Y2,facecolor = "#ff0000",label = "Minas Gerais", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X ,Y3,facecolor = "#0a14c8",label = "Bahia", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X+0.15,Y4,facecolor = "#4795e0",label = "Rio de Janiero", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X+0.3 ,Y5,facecolor = "#fcf10c",label = "Pernambuco", width = 0.15, align = "center",edgecolor = "white")

nombres = ['Galaxy J5', 'Galaxy J7', 'Galaxy S5', 'Galaxy S5', 'Galaxy S6 Edge', 'iPhone 5c', 'iPhone 5s', 'iPhone 6',
     'iPhone 6S', 'iPhone SE']

plt.xticks(X,nombres,fontsize = 10)
plt.xlabel("Modelos",fontsize = 18)
plt.ylabel('Cantidad de compras', fontsize = 18)

plt.title('Comparacion de los telefonos mas vendidos en Sao Pablo segun las regiones mas polulares\n', fontsize = 18)

plt.legend(loc="upper right",fontsize = 12)
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

plt.style.use('default') # haciendo los graficos un poco mas bonitos en matplotlib
#plt.rcParams['figure.figsize'] = (20, 10)

sns.set(style="whitegrid") # seteando tipo de grid en seaborn
types= {
    "event": "category", 
    "model": "category", 
    "condition": "category", 
    "color": "category", 
    "storage": "category"
}

df_events = pd.read_csv("events.csv", low_memory=False)
users = df_events[["person","new_vs_returning","city","region","country","event","timestamp"]]
users = users.loc[users["new_vs_returning"].isnull() == False]
users["new_vs_returning"].value_counts()
#users = users.drop_duplicates(subset='person')
user = users["new_vs_returning"].value_counts()
user
ax = user.plot(kind = 'pie', title = "Nuevos usuarios vs Retuning",figsize =(8,8),legend = False, fontsize = 12 )
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
users = users.loc[users["country"] != "Unknown"]
usersN = users.loc[users["new_vs_returning"] == "New"]
usersN = usersN["country"].value_counts().head(4)
usersR = users.loc[users["new_vs_returning"] == "Returning"]
usersR = usersR["country"].value_counts().head(4)
usersR
X = np.arange (len(usersN.index))
Y1 = usersN.values
Y2 = usersR.values


plt.axes([0.025,0.025,1.5,1.5])
plt.bar(X-0.125 ,Y1,facecolor = "#0080FF",label = "Nuevos", width = 0.25, align = "center",edgecolor = "white")
plt.bar(X+0.125,Y2,facecolor = "#000000",label = "Viejos", width = 0.25, align = "center",edgecolor = "white")

plt.ylim(0,300)
plt.xticks(X,usersN.index,fontsize = 10)
plt.xlabel("Paises",fontsize = 18)
plt.ylabel('Cantidad de usuarios', fontsize = 18)

plt.title('Nuevos y viejos usuarios por Paises\n', fontsize = 18)

plt.legend(loc="upper right")
plt.show()
X = np.arange (len(usersN.index))
Y1 = usersN.values
Y2 = usersR.values


plt.axes([0.025,0.025,1.5,1.5])
plt.bar(X-0.125 ,Y1,facecolor = "#0080FF",label = "Nuevos", width = 0.25, align = "center",edgecolor = "white")
plt.bar(X+0.125,Y2,facecolor = "#000000",label = "Viejos", width = 0.25, align = "center",edgecolor = "white")

plt.ylim(0,60000)
plt.xticks(X,usersN.index,fontsize = 10)
plt.xlabel("Paises",fontsize = 18)
plt.ylabel('Cantidad de usuarios', fontsize = 18)

plt.title('Nuevos y viejos usuarios por Paises\n', fontsize = 18)

plt.legend(loc="upper right")
plt.show()
usersN = np.log(usersN)
usersR = np.log(usersR)
X = np.arange (len(usersN.index))
Y1 = usersN.values
Y2 = usersR.values


plt.axes([0.025,0.025,1.5,1.5])
plt.bar(X-0.125 ,Y1,facecolor = "#0080FF",label = "Nuevos", width = 0.25, align = "center",edgecolor = "white")
plt.bar(X+0.125,Y2,facecolor = "#000000",label = "Viejos", width = 0.25, align = "center",edgecolor = "white")

plt.ylim(0,12)
plt.xticks(X,usersN.index,fontsize = 10)
plt.xlabel("Paises",fontsize = 18)
plt.ylabel('Log de Cantidad de usuarios', fontsize = 18)

plt.title('Nuevos y viejos usuarios por Paises\n', fontsize = 18)

plt.legend(loc="upper right")
plt.show()


        

    
#g = sns.barplot(x = usersR.index, y = usersR.values, orient='v', palette=['red'], alpha=1)
#g = sns.barplot(x = usersN.index, y = usersN.values, orient='v', palette=['green'], alpha=0.5)
#g.legend(['Returning','New'],ncol=2, loc='upper right');
#g.set_title("Nuevos usuarios vs viejos usuarios", fontsize=18)
#g.set_xlabel("Paises",fontsize=18)
#g.set_ylabel("Log de Cantidad", fontsize=18)

#usersN



#plt.show()

#sns.countplot(x = usersN.index, y = usersN.values)
#sns.countplot(usersR)

#ax = userN.plot(kind = 'barh', title = "Nuevos usuarios por Paises",figsize =(15,10),legend = False, fontsize = 12 )
#ax.subplt(userR ,color ="r")
#ax.set_xlabel("Paises", fontsize = 18)
#ax.set_ylabel("Cantidad nuevos de usuarios",fontsize = 18)
#users = np.log(user)
#ax = user.plot(kind = 'barh', title = "Nuevos usuarios por Paises",figsize =(15,10),legend = False, fontsize = 12 )
#ax.set_xlabel("Paises", fontsize = 18)
#ax.set_ylabel("Log cantidad nuevos de usuarios",fontsize = 18)
usersB = users.loc[users["country"] == "Brazil"]
usersB = usersB.loc[users["region"] != "Unknown"]
userB = usersB["region"].value_counts().head(10)
userB
ax = userB.plot(kind = 'bar', title = "Nuevos usuarios vs Retuning",figsize =(13,8),legend = False, fontsize = 12 )
ax.set_xlabel("Region", fontsize = 18)
ax.set_ylabel("Cantidad de Eventos",fontsize = 18)
plt.show()
users = users.loc[users["country"] == "Argentina"]
users = users.loc[users["region"] != "Unknown"]
usersA = users["region"].value_counts().head(10)
usersA
ax = usersA.plot(kind = 'bar', title = "Nuevos usuarios vs Retuning",figsize =(13,8),legend = False, fontsize = 12 )
ax.set_xlabel("Region", fontsize = 18)
ax.set_ylabel("Cantidad de Eventos",fontsize = 18)
plt.show()
usersN = users.loc[users["new_vs_returning"] == "New"]
#usersN = usersN["person"]
#usersN["event"].value_counts()
usuarios_nuevos = usersN.drop_duplicates(subset='person')['person'].tolist()

_users = df_events.loc[(df_events['event'] == 'conversion') & (df_events['person'].isin(usuarios_nuevos))]
_users = _users[["person","model","condition"]]
#_users
_users_condition = _users["condition"].value_counts()
_users_model = _users["model"].value_counts().head(10)
_users_model
ax = _users_model.plot(kind = 'bar', title = "Cantidad de compras realizadas por nuevos usuarios",figsize =(13,8),legend = False, fontsize = 12 )
ax.set_xlabel("Modelos", fontsize = 18)
ax.set_ylabel("Cantidad de compras",fontsize = 18)
plt.show()
ax = _users_condition.plot(kind = 'bar', title = "Condicion de las compras de los nuevos usuarios",figsize =(13,8),legend = False, fontsize = 12 )
ax.set_xlabel("Condicion", fontsize = 18)
ax.set_ylabel("Cantidad de compras",fontsize = 18)
ax.set_xticklabels(["Bueno","Muy Bueno","Excelente","Bueno - sin Touch ID"],rotation = "horizontal")
plt.show()
usersR = users.loc[users["new_vs_returning"] == "Returning"]
#usersR["event"].value_counts()
usuarios_viejos = usersR.drop_duplicates(subset='person')['person'].tolist()

_users = df_events.loc[(df_events['event'] == 'conversion') & (df_events['person'].isin(usuarios_viejos))]
_users = _users.loc[_users['person'].isin(usuarios_nuevos)]


_users = _users[["person","model","condition"]]
_users
_users_condition = _users["condition"].value_counts()
_users_model = _users["model"].value_counts().head(10)
_users_model
lista = []
repetidos = []
for usuario in usuarios_nuevos:
    if not usuario in usuarios_viejos:
        lista.append(usuario)
    else:
        repetidos.append(usuario)
        
repetidos



import pandas as pd
import matplotlib as mlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np

mylist = []

for chunk in pd.read_csv('../../events.csv', low_memory=False, chunksize=20000):
    mylist.append(chunk)

df_events = pd.concat(mylist, axis= 0)
del mylist


df_events.columns
df_events['event'].value_counts()
# 1) Cantidad de visitas segun pais

df_1 = df_events[(df_events["event"]=="visited site") | (df_events["event"]=="ad campaign hit")]
df_1[df_1["event"]=="ad campaign hit"]["country"].value_counts()

## Esto lleva a no saber en qué paises conviene poner publicidad,
## dado que falta la información del pais para el evento "ad campaign hit"

## Si se tuviera esa información, se podría invertir más plata en publicidades
## para aquellos países que tienen un mayor ratio personas <-> publicidades clickeadas,
## o también visitas al sitio <-> publicidades clickeadas

## 2) Publicidades más clickeadas

df_events[df_events["event"]=="ad campaign hit"]["url"].value_counts().head(6)
df_events[df_events["event"]=="ad campaign hit"].head()
## Eventos vender
df_events["es_vender"] = df_events[df_events["event"]=="ad campaign hit"]["url"].str.contains("/vender")
df_vendiendo = df_events[df_events["es_vender"]==True]

## Personas vendiendo
df_vendiendo["person"].value_counts()

## Modelos a vender
df_vendiendo["url"].value_counts()

## Podemos ver que las url a las que llegaron los usuarios son de brasil, por lo que tal vez
## personas de otros países no esten queriendo/intentando/pudiendo vender (argentina no puede por ej.)

# Los modelos a vender son más viejos que los que se quieren comprar, no hay muchas ventas de modelos nuevos
# Eventos comprar
df_events["es_comprar"] = df_events[df_events["event"]=="ad campaign hit"]["url"].str.contains("/comprar")
df_comprando = df_events[df_events["es_comprar"]==True]

# Personas comprando
# df_comprando["person"].value_counts()

# Modelos a comprar
publicidades_comprar = df_comprando["url"].value_counts().head(10)

## Vemos que las marcas que insinúan a la gente a clickear las publicidades son
## iphone, samsung y motorola (en orden)
# Los menos interesantes
df_comprando["url"].value_counts().tail(10)
fig, ax = plt.subplots()
ax.set(xlabel='Cantidad de clicks', ylabel='URLs',
       title='Publicidades más clickeadas para comprar dispositivos')
ax.plot(publicidades_comprar.values, publicidades_comprar.index.values)
# 3) Checkout y conversion - Patrones segun usuarios

checkyconv_events = df_events[(df_events["event"]=="conversion") | (df_events["event"]=="checkout")]

persons_array = checkyconv_events["person"].value_counts().head(100).index.values
df_personas_checkyconv = df_events[df_events["person"].isin(persons_array)]
# El evento "visited site" es el unico que recopila informacion sobre el tipo de dispositivo
personas_smartphone = df_personas_checkyconv[df_personas_checkyconv["device_type"]=="Smartphone"]["person"].value_counts().index.values
personas_computer = df_personas_checkyconv[df_personas_checkyconv["device_type"]=="Computer"]["person"].value_counts().index.values
df_personas_smartphone = df_personas_checkyconv[df_personas_checkyconv["person"].isin(personas_smartphone)]
df_personas_smartphone[(df_personas_smartphone["event"]=="conversion")]["person"].value_counts().values.sum()
df_personas_computer = df_personas_checkyconv[df_personas_checkyconv["person"].isin(personas_computer)]
df_personas_computer[(df_personas_computer["event"]=="conversion")]["person"].value_counts().values.sum()
computer_group = df_personas_computer.groupby("event").agg({"person": "count"})
computer_group
smartphone_group = df_personas_smartphone.groupby("event").agg({"person": "count"})
smartphone_group
ratio_smart_computer = computer_group/smartphone_group
ratio_smart_computer
# Un ratio menor a 1 significa que es mas prevalente el grupo de smartphone, y viceversa
# 3) 1. Parecidos
# Vemos que la gente, para clickear publicidades, añadir al carrito, comprar, ver un listado generico, busqueda de productos,
# y visitar paginas estaticas (como terminos y condiciones, etc.) no varía tanto según si usan un dispositivo móvil o una computadora.
from statsmodels.graphics.mosaicplot import mosaic

I = [1,5,6,9,10]
parecidos_computer = computer_group.values
np.delete(parecidos_computer, I)
parecidos_smartphone = smartphone_group.values
np.delete(parecidos_smartphone, I)

parecidos_array = []
computer_array = []
for valor in parecidos_computer:
    computer_array.append((1, valor[0]))
smart_array = []
for valor in parecidos_smartphone:
    smart_array.append((2, valor[0]))
parecidos_array.append(computer_array)
parecidos_array.append(smart_array)
del smart_array
del computer_array

# ax = plt.gca()
# ax.text(0.8, 0, 'Smartphone',
#         horizontalalignment='right',
#         verticalalignment='bottom',
#         transform=ax.transAxes)
# ax.text(0.2, 0, 'Computer',
#         horizontalalignment='left',
#         verticalalignment='bottom',
#         transform=ax.transAxes)

# labelizer = lambda k: ""

# mosaic(parecidos_array, 
#        title="Comparacion entre eventos comunes para usuarios con Smartphones y Computadoras", 
#        axes_label=None,
#        ax=ax,
#       labelizer=labelizer)
# 3) 2.
# Diferencias entre eventos desde smartphone (más en lead y search engine hit)
# y computadoras (más en brand listing, viewed product y visited site)
computer_values = []
for value in computer_group.values:
    computer_values.append(value[0])
np.delete(computer_values, [0,1,2,3,4,7,8,9,10])

smartphone_values = []
for value in smartphone_group.values:
    smartphone_values.append(value[0])
np.delete(smartphone_values, [0,2,3,4,5,6,7,8])

df = pd.DataFrame()
df["computer"]=computer_values
df["smartphone"]=smartphone_values
df["events"]=smartphone_group.index.values
df
ratio_computer_smart = 1/ratio_smart_computer
events = df["events"]
values = df["computer"]
values2 = df["smartphone"]
plt.scatter(events, values, s=(ratio_smart_computer)**3*(1000), c="red", alpha=0.5)
plt.scatter(events, values2, s=(ratio_computer_smart)**3*(1000), c="blue", alpha=0.5)
# Rotula y dibuja el gráfico
plt.xlabel('Eventos')
plt.ylabel('Cantidad')
plt.title('Relación entre eventos según dispositivo utilizado')
plt.legend(markerscale=0.3,fontsize=12)
plt.show()

# 4) Click en publicidad - desde qué dispositivo
# Esta informacion tambien falta, no se puede saber desde que dispositivo se accedio a la publicidad
# Esto hace que se pierda informacion sobre donde poner publicidades, si conviene invertir mas en un diseño para mobile
# o si la gente no suele clickear publicidades desde sus celulares o tablets.
# c013417a    277
# 5af7e2bc    220
# 875eb866    207
# 5107ab49    167
# 14752aa3    152
# 13d3dbee    147
# ff9dc4b8    144
# 3952fd6f    144
# 7433a87f    133
# e2b0ce1b    131
# ba102035    127
# d9251b63    122
# 778fcfbd    119

df_events[df_events["person"]=="5107ab49"]["device_type"].value_counts()
# 5) Pais del que provienen los que clickean ads

persons_ads = df_events[df_events["event"]=="ad campaign hit"]["person"].value_counts().index.values


countries_ads = df_events[df_events["person"].isin(persons_ads)]["country"].value_counts().head(8).index.values
values_countries_ads = df_events[df_events["person"].isin(persons_ads)]["country"].value_counts().head(8).values
df_events[df_events["person"].isin(persons_ads)]["country"].value_counts().head(8)
values_countries_total = df_events[df_events["country"].isin(countries_ads)]["country"].value_counts().values
df_events[df_events["country"].isin(countries_ads)]["country"].value_counts()
ratio_total_ads = values_countries_total/values_countries_ads

porcentaje_eventos = values_countries_total

cmap = mlib.cm.viridis
norm = mlib.colors.Normalize(vmin=0, vmax=84308)

fig = plt.figure(figsize=(8, 6)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
ax0 = plt.subplot(gs[0])
# Dibuja los circulos segun la data de eventos, variando el radio segun el porcentaje 

plt.scatter(100/ratio_total_ads, countries_ads, s=(porcentaje_eventos)**(0.4)*10, c=cmap(norm(values_countries_total)))

# Rotula y dibuja el gráfico
plt.xlabel('Porcentaje visitas:publicidades')
plt.ylabel('Pais')
plt.title('Visitas a partir de publicidades')
plt.axis([35, 105, -1, 8])


ax1 = plt.subplot(gs[1])
# Colorbar
cb1 = mlib.colorbar.ColorbarBase(ax1, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
cb1.set_label('Cantidad total de visitas')

plt.show()

# importacion general de librerias y de visualizacion (matplotlib y seaborn)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

plt.style.use('default') # haciendo los graficos un poco mas bonitos en matplotlib
#plt.rcParams['figure.figsize'] = (20, 10)

sns.set(style="whitegrid") # seteando tipo de grid en seaborn
types= {
    "event": "category", 
    "model": "category", 
    "condition": "category", 
    "color": "category", 
    "storage": "category"
}
df_events = pd.read_csv('./fiuba-trocafone-tp1-final-set/events.csv', dtype=types, low_memory=False)
df_events = df_events[["person",'timestamp','event','campaign_source','search_engine']]
df_events['is_new_session'] = np.where((df_events["event"] == "visited site"),True,False)
checkout_events = df_events.sort_values(by=['person','timestamp','is_new_session'], ascending=[True,True,False])\
        .loc[(df_events["event"] == "checkout") | (df_events["event"] == "ad campaign hit") \
             | (df_events["event"] == "visited site")]
session_id = (checkout_events["event"] == "visited site").cumsum()

checkout_events['session_id'] = session_id
checkouts_per_session = checkout_events[['event','session_id']].loc[(checkout_events["event"] == "checkout")]\
                    .groupby(['session_id'])\
                    .agg({'event' : 'count'})\
                    .rename(columns={'session_id' : 'session_id' , 'event' : 'checkouts_qty'})
checkouts_per_session
campaign_source_per_session = checkout_events[['event','session_id','campaign_source']]\
            .loc[(checkout_events["event"] == "ad campaign hit")].groupby(['session_id'])['campaign_source'].first()
campaign_source_per_session
checkouts_and_campaign_source_per_session = pd.concat([checkout_events['session_id'],checkouts_per_session,campaign_source_per_session], axis=1)
checkouts_qty_per_campaign = checkouts_and_campaign_source_per_session\
                        .fillna(0)\
                        .groupby(['campaign_source','checkouts_qty'])\
                        .size()
checkouts_qty_per_campaign


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

plt.style.use('default') # haciendo los graficos un poco mas bonitos en matplotlib
#plt.rcParams['figure.figsize'] = (20, 10)

sns.set(style="whitegrid") # seteando tipo de grid en seaborn
types= {
    "event": "category", 
    "model": "category", 
    "condition": "category", 
    "color": "category", 
    "storage": "category"
}
df_events = pd.read_csv("events.csv", dtype=types, low_memory=False)
models = df_events[["person","model","storage","event","color","condition"]]
models = models.loc[models["event"] == "conversion" ,:]
df = df_events.loc[(df_events["country"].isnull() == False) & (df_events["country"] != "Unknown"),:]
df = df[['person','country','region']]

df = df.drop_duplicates(subset='person')


#df2 = df[["person","region","country"]]

models = models.merge(df, on='person')
colores = {"Preto" : "Negro", "Dourado" : "Dorado", "Branco" : "Blanco", "Cinza espacial" : "Negro", "Prateado" : "Plata", "Ouro Rosa" : "Rosa",
"Rosa": "Rosa", "Cinza" : "Plata", "Azul" : "Azul", "Preto Vermelho" : "Negro", "Prata" : "Plata", "Platinum" : "Plata", "Preto Matte" : "Negro",
"Branco Vermelho" : "Blanco", "Ouro" : "Dorado", "Titânio" : "Plata", "Ametista" : "Otros", "Preto Brillhante" : "Negro", "Indigo" : "Otros",
"Amarelo" : "Otros", "Vermelho" : "Otros", "Bambu" : "Otros", "Cabernet" : "Otros", "Preto Azul" : "Negro", "Couro Vintage" : "Otros", "Azul Topázio" : "Azul"}

models["in_color"] = models['color'].apply(lambda x: colores.get(x, 'Basura'))

models = models.loc[models['in_color'] != 'Basura']
colores = models['in_color'].value_counts().sort_index()
colores
brazil = models.loc[models["country"] == "Brazil"]
brazil = brazil.loc[brazil["region"] != "Unknown"]

top_5_regiones = brazil["region"].value_counts().head()
sao_pablo = brazil.loc[(brazil["region"] == "Sao Paulo")]
sao_pablo = sao_pablo["in_color"].value_counts().sort_index()
sao_pablo
minas_gerais  = brazil.loc[brazil["region"] == "Minas Gerais"]

"""colores_f = pd.Series( [0,0,0,0,0,0,0],
    index = ["azul","Blanco","Dorado","Negro","Otros","Plata","Rosa")"""

        

minas_gerais = minas_gerais["in_color"].value_counts().sort_index()
minas_gerais
rio  = brazil.loc[brazil["region"] == "Rio de Janeiro"]

colores_f = colores_f = pd.Series( [0],
    index = ["Otros"])
rio = rio["in_color"].value_counts()
rio = rio.append(colores_f) 

rio = rio.sort_index()
rio
maranhao  = brazil.loc[brazil["region"] == "Maranhao"]
maranhao = maranhao["in_color"].value_counts().sort_index()
maranhao 
bahia  = brazil.loc[brazil["region"] == "Bahia"]

bahia = bahia["in_color"].value_counts()

colores_f = pd.Series( [0],
    index = ["Azul"])

bahia = bahia.append(colores_f)

bahia = bahia.sort_index()
bahia 
nombres = ["Azul","Blanco","Dorado","Negro","Otros","Plata","Rosa"]

X = np.arange (len(nombres))
Y1 = sao_pablo.values
Y2 = minas_gerais.values
Y3 = bahia.values
Y4 = rio.values
Y5 = maranhao.values



plt.axes([0.025,0.025,1.5,1.5])
plt.bar(X-0.3 ,Y1,facecolor = "#000000",label = "Sao Paulo", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X-0.15,Y2,facecolor = "#ff0000",label = "Minas Gerais", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X ,Y3,facecolor = "#0a14c8",label = "Bahia", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X+0.15,Y4,facecolor = "#4795e0",label = "Rio de Janiero", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X+0.3 ,Y5,facecolor = "#fcf10c",label = "Maranhao", width = 0.15, align = "center",edgecolor = "white")


plt.xticks(X,nombres,fontsize = 10)
plt.xlabel("Colores",fontsize = 18)
plt.ylabel('Cantidad de compras', fontsize = 18)

plt.title('Comparacion los colores mas vendidos en las regiones mas polulares de Brasil\n', fontsize = 18)

plt.legend(loc="upper right",fontsize = 12)
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

plt.style.use('default') # haciendo los graficos un poco mas bonitos en matplotlib
#plt.rcParams['figure.figsize'] = (20, 10)

sns.set(style="whitegrid") # seteando tipo de grid en seaborn
types= {
    "event": "category", 
    "model": "category", 
    "condition": "category", 
    "color": "category", 
    "storage": "category"
}
df_events = pd.read_csv("events.csv", dtype=types, low_memory=False)
#compradores = df_events.loc[df_events["event"] == "conversion",:]
compradores = df_events[["person","model","event","country"]]
#compradores_ch = compradores.loc[compradores["event"] == "checkout"]
compradores_con = compradores.loc[compradores["event"] == "conversion"]
a = compradores_con["person"].value_counts().head(20).sort_index()
a
#compradores_ti = compradores_con.drop_duplicates(subset='person')['person'].tolist()
#compradores_ti
compradores_ch.count()
#usuarios_nuevos = compradores_ch.drop_duplicates(subset='person')['person'].tolist()

#compradores_ch = compradores.loc[(compradores['person'].isin(usuarios_nuevos))&(compradores["event"] == "checkout")]
compradores_ti = a.index.tolist()
compradores_cch = compradores.loc[(compradores['person'].isin(compradores_ti))&(compradores["event"] == "checkout")]

b = compradores_cch["person"].value_counts().head(20).sort_index()
b
"""a = compradores_con["person"].value_counts().sort_index().head(20)
a"""
X = np.arange (len(compradores_ti))
Y1 = a.values
Y2 = b.values

plt.axes([0.025,0.025,1.5,1.5])
plt.bar(X-0.15 ,Y1,facecolor = "#F6728D",label = "Conversion", width = 0.3, align = "center",edgecolor = "white")
plt.bar(X+0.15,Y2,facecolor = "#6C5878",label = "Checkouts", width = 0.3, align = "center",edgecolor = "white")


plt.xticks(X,compradores_ti,fontsize = 10,rotation='vertical')
plt.xlabel("Usuarios",fontsize = 18)
plt.ylabel('Cantidad', fontsize = 18)

plt.title('Checkouts vs conversiones de los 20 usuarios que mas compraron\n', fontsize = 18)

plt.legend(loc="upper right")
plt.show()

# importacion general de librerias y de visualizacion (matplotlib y seaborn)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

plt.style.use('default') # haciendo los graficos un poco mas bonitos en matplotlib
#plt.rcParams['figure.figsize'] = (20, 10)

sns.set(style="whitegrid") # seteando tipo de grid en seaborn
types= {
    "event": "category", 
    "model": "category", 
    "condition": "category", 
    "color": "category", 
    "storage": "category"
}
df_events = pd.read_csv('./fiuba-trocafone-tp1-final-set/events.csv', dtype=types, low_memory=False)
df_events[['campaign_source','search_engine', 'device_type', 'new_vs_returning']].head(20)
users1 = df_events[["new_vs_returning", "person", "event", "campaign_source", "search_engine", "device_type"]]
users1["person"].value_counts()
users2 = users1.loc[(users1["event"] == "ad campaign hit") & (users1["campaign_source"] == "google")]
users2["person"].value_counts()
#Cantidad de personas que visitan la pagina mediante una computadora
filter_col = df_events[[ "person", "event", "device_type"]]
filter_row = filter_col.loc[(filter_col["event"] == "visited site") & (filter_col["device_type"] == "Computer")]
filter_row
#Cantidad de personas que visitan la pagina mediante un smartphone
filter_row = filter_col.loc[(filter_col["event"] == "visited site") ]
filter_row["device_type"].value_counts()
#filter_row.loc[(filter_row["device_type"] == "Unknown")]
filter_row
filter = (filter_row.loc[(filter_row["device_type"] != "Unknown")])["device_type"].value_counts()
#filter_row = filter_row["device_type"].value_counts()
filter
ax = filter.plot.pie(title = "Dispositivos mas utilizados para visitar la página", figsize =(7,7), colors = ['#66b3ff','#ff6666','#99ff99'], legend = False, fontsize = 12 )
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

plt.style.use('default') # haciendo los graficos un poco mas bonitos en matplotlib
#plt.rcParams['figure.figsize'] = (20, 10)

sns.set(style="whitegrid") # seteando tipo de grid en seaborn
types= {
    "event": "category", 
    "model": "category", 
    "condition": "category", 
    "color": "category", 
    "storage": "category"
}
df_events = pd.read_csv("events.csv", dtype=types, low_memory=False)
df_events = df_events[["person","device_type","screen_resolution","operating_system_version","browser_version"]]
df_events.count()
models = df_events.loc[(df_events["device_type"].isnull() == False) & (df_events["device_type"] != "Unknown"),:]
models = models.drop_duplicates(subset='person')
device = models["device_type"].value_counts()
device
ax = device.plot(kind = 'pie', title = "Distintos tipo de dispositivos que usan trocafone.com",figsize =(6,6),legend = False, fontsize = 12)

my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
ax = device.plot(kind = 'bar', title = "Distintos tipo de dispositivos que usan trocafone.com",figsize =(6,6),legend = False, fontsize = 12,color =[["#4d70b2","#57a76a","#c84d52"]])
ax.set_xlabel("Dispositivo", fontsize = 18)
ax.set_ylabel("Cantidad",fontsize = 18)
plt.show()
celus = df_events.loc[df_events["device_type"] == "Smartphone",:]
celus = celus.drop_duplicates(subset='person')

celus["system_op"] = celus['operating_system_version'].apply(lambda x: x.split('.')[0])
celus
so_smart = celus["system_op"].value_counts().head(10)
so_smart
ax = so_smart.plot(kind = 'bar', title = "Distintos SO que ingresan en trocafone.com",figsize =(10,10),legend = False, fontsize = 12, color =[["#77C159","#77C159","#77C159","#77C159",'#FD166A',"#77C159",'#FD166A','#FD166A','#681B7B','#681B7B']])
ax.set_xlabel("Sistema operativo", fontsize = 18)
ax.set_ylabel("Cantidad",fontsize = 18)
plt.show()
compus = df_events.loc[df_events["device_type"] == "Computer",:]
compus = compus.drop_duplicates(subset='person')

#celus["system_op"] = celus['operating_system_version'].apply(lambda x: x.split('.')[0])
#celus
so_compus = compus["operating_system_version"].value_counts().head(10)
so_compus
ax = so_compus.plot(kind = 'bar', title = "Distintos SO en computadoras que ingresan en trocafone.com",figsize =(10,10),legend = False, fontsize = 12,color = [["#00ADEF","#00ADEF","#00ADEF","#00ADEF","#00ADEF","#FFB700","#FFB700","#F64A15","#00ADEF","#F64A15"]])
ax.set_xlabel("Sistema operativo", fontsize = 18)
ax.set_ylabel("Cantidad",fontsize = 18)
plt.show()
resolucion_s = df_events.loc[df_events["device_type"] == "Smartphone",:]
resolucion_s = resolucion_s.drop_duplicates(subset='person')
resolucion_s = resolucion_s["screen_resolution"].value_counts().head(20)
resolucion_s
resolucion_c = df_events.loc[df_events["device_type"] == "Computer",:]
resolucion_c = resolucion_c.drop_duplicates(subset='person')
resolucion_c = resolucion_c["screen_resolution"].value_counts().head(10)
resolucion_c
ax = resolucion_s.plot(kind = 'bar', title = "Distintas resolucion de pantalla de los distintos smartphone",figsize =(10,8),legend = False, fontsize = 12)
ax.set_xlabel("Resolucion de pantalla", fontsize = 18)
ax.set_ylabel("Cantidad",fontsize = 18)
plt.show()
ax = resolucion_c.plot(kind = 'bar', title = "Distintas resolucion de pantalla de los distintas computadoras",figsize =(8,8),legend = False, fontsize = 12)
ax.set_xlabel("Resolucion de pantalla", fontsize = 18)
ax.set_ylabel("Cantidad",fontsize = 18)
plt.show()

# importacion general de librerias y de visualizacion (matplotlib y seaborn)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

# TODO: ver que onda los warnings
import warnings
warnings.filterwarnings('ignore')


plt.style.use('default') 
plt.rcParams['figure.figsize'] = (20, 10)

sns.set(style="whitegrid") # seteando tipo de grid en seaborn
def plot_heatmap(df, x, y, z, label_x='x', label_y='y', label_z='z', title='titulo', color='hot_r', invert_color=False):
    fig, ax = plt.subplots(figsize=(14,14))
    graph = sns.heatmap(df.pivot_table(index=y,columns=x,values=z),\
    linewidths=.5,cmap=color, ax=ax, cbar_kws={'label': label_z}, annot=False)
    ax.set_xlabel(label_x);
    ax.set_ylabel(label_y);
    ax.set_title(title)
    if invert_color:
        ax.invert_yaxis()
types= {
    "event": "category", 
    "model": "category", 
    "condition": "category", 
    "color": "category", 
    "storage": "category",
}
parse_dates = ['timestamp']
df_events = pd.read_csv('../../events.csv', dtype=types, low_memory=False, parse_dates=parse_dates)
df_events['year'] = df_events['timestamp'].dt.year
df_events['month'] = df_events['timestamp'].dt.month
df_events['day'] = df_events['timestamp'].dt.day
df_events['time'] = df_events['timestamp'].dt.time
df_events.columns
df_horaPico = df_events[['timestamp', 'day', 'event']]
df_horaPico['hour'] = df_horaPico['timestamp'].dt.hour
df_horaPico.drop('timestamp', axis='columns', inplace=True)
df_horaPico = df_horaPico.groupby(['hour', 'day']).agg({'day':'count'})
df_horaPico.columns = ['count']
df_horaPico.reset_index(inplace=True)
df_horaPico

plot_heatmap(df_horaPico,  x='hour', y='day', z='count', title='Cantidad de eventos según día y hora', label_x='Hora',label_y='Dia',label_z='Cantidad de eventos',invert_color=True)
df_horaPico = df_events[['timestamp', 'day', 'event']]
df_horaPico['hour'] = df_horaPico['timestamp'].dt.hour
df_horaPico.drop('timestamp', axis='columns', inplace=True)
df_conversions = df_horaPico[df_horaPico['event']=='conversion']
conversions_ser = df_conversions.groupby('hour').agg({'event': 'count'})
conversions_ser.reset_index(inplace=True)
conversions_ser.columns = ['hour', 'count']

df_viewed = df_horaPico[df_horaPico['event']=='viewed product']
viewed_ser = df_viewed.groupby('hour').agg({'event': 'count'})
viewed_ser.reset_index(inplace=True)
viewed_ser.columns = ['hour','count']
index = viewed_ser['hour'].values

fig, ax = plt.subplots()
bar_width = 0.35

opacity = 0.4

rects1 = ax.bar(index, conversions_ser['count'].values, bar_width,
                alpha=opacity, color='b',
                label='conversions')

rects2 = ax.bar(index + bar_width, viewed_ser['count'].values, bar_width,
                alpha=opacity, color='r',
                label='viewed product')

ax.set_xlabel('Hora')
ax.set_ylabel('Cantidad')
ax.set_title('Cantidad de evento segun la hora')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(index)
ax.legend()

fig.tight_layout()
plt.show()
k = (viewed_ser['count']/conversions_ser['count']).mean()
index = viewed_ser['hour'].values

fig, ax = plt.subplots()
bar_width = 0.35

opacity = 0.4

rects1 = ax.bar(index, conversions_ser['count'].values * k, bar_width,
                alpha=opacity, color='b',
                label='conversions')

rects2 = ax.bar(index + bar_width, viewed_ser['count'].values, bar_width,
                alpha=opacity, color='r',
                label='viewed product')

ax.set_xlabel('Hora')
ax.set_ylabel('Cantidad')
ax.set_title('Cantidad de evento segun la hora')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(index)
ax.legend()

fig.tight_layout()
plt.show()
df_mayoristas = df_events[df_events['event'] == 'conversion']['person'].value_counts()
df_mayoristas.head(10)
adicto = df_events[(df_events['person'] == '252adec6') & (df_events['event'] == 'conversion')]
[
    adicto['timestamp'].max()-
    adicto['timestamp'].min(),
    df_events['timestamp'].min(),
    df_events['timestamp'].max(),
]
adicto['timestamp'];

adicto2 = df_events[(df_events['person'] == '4200bdee') & (df_events['event'] == 'conversion')]
[
    adicto2['timestamp'].max()-
    adicto2['timestamp'].min(),
    df_events['timestamp'].min(),
    df_events['timestamp'].max(),
]
adicto2['timestamp'];
adicto3 = df_events[(df_events['person'] == 'a0d4baef') & (df_events['event'] == 'conversion')]
[
    adicto3['timestamp'].max()-
    adicto3['timestamp'].min(),
    df_events['timestamp'].min(),
    df_events['timestamp'].max(),
]
adicto3['timestamp'];
# Ver como mostrar adicto, adicto2, adicto3
# Un grafico cada uno?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

plt.style.use('default') # haciendo los graficos un poco mas bonitos en matplotlib
#plt.rcParams['figure.figsize'] = (20, 10)

sns.set(style="whitegrid") # seteando tipo de grid en seaborn
types= {
    "event": "category", 
    "model": "category", 
    "condition": "category", 
    "color": "category", 
    "storage": "category"
}
df_events = pd.read_csv("events.csv", dtype=types, low_memory=False)
models = df_events[["person","model","storage","event","color","condition"]]
models = models.loc[models["event"] == "conversion" ,:]
df = df_events.loc[(df_events["country"].isnull() == False) & (df_events["country"] != "Unknown"),:]
df = df[['person','country','region']]

df = df.drop_duplicates(subset='person')


#df2 = df[["person","region","country"]]

models = models.merge(df, on='person')
brazil = models.loc[models["country"] == "Brazil"]
brazil = brazil.loc[brazil["region"] != "Unknown"]

#top = brazil["region"].value_counts()

top_10_t = brazil["model"].value_counts().head(10)
top_10 = top_10_t.index.tolist()
top_10.sort()
top_10

sao_pablo = brazil.loc[(brazil["region"] == "Sao Paulo")&(brazil["model"].isin(top_10))]
sao_pablo = sao_pablo["model"].value_counts().head(10).sort_index()
minas_gerais = brazil.loc[(brazil["region"] == "Minas Gerais")&(brazil["model"].isin(top_10))]

minas_gerais = minas_gerais["model"].value_counts().head(9)

modelos_f = pd.Series( [0], index = ['iPhone 5c'])

minas_gerais = minas_gerais.append(modelos_f)

minas_gerais = minas_gerais.sort_index()
minas_gerais
rio = brazil.loc[(brazil["region"] == "Rio de Janeiro")&(brazil["model"].isin(top_10))]
rio = rio["model"].value_counts().head(7)

modelos_f = pd.Series( [0,0,0], index = ['Motorola Moto G3 4G','Samsung Galaxy Gran Prime Duos TV','Samsung Galaxy J7'])

rio = rio.append(modelos_f)
rio = rio.sort_index()

rio
bahia = brazil.loc[(brazil["region"] == "Bahia")&(brazil["model"].isin(top_10))]
bahia = bahia["model"].value_counts().head(8)

modelos_f = pd.Series( [0,0], index = ['Samsung Galaxy S6 Flat','iPhone 6S'])

bahia = bahia.append(modelos_f)
bahia = bahia.sort_index()

bahia

maranhao = brazil.loc[(brazil["region"] == "Maranhao")&(brazil["model"].isin(top_10))]
maranhao = maranhao["model"].value_counts().head(4)

modelos_f = pd.Series( [0,0,0,0,0,0],
    index = [ 'Motorola Moto G4 Plus','Samsung Galaxy Gran Prime Duos TV', 'Samsung Galaxy J5','Samsung Galaxy J7',
 'iPhone 5c','iPhone 6'])

maranhao = maranhao.append(modelos_f)
maranhao = maranhao.sort_index()

maranhao



X = np.arange (len(top_10))
Y1 = sao_pablo.values
Y2 = minas_gerais.values
Y3 = bahia.values
Y4 = rio.values
Y5 = maranhao.values



plt.axes([0.025,0.025,1.5,1.5])
plt.bar(X-0.3 ,Y1,facecolor = "#000000",label = "Sao Paulo", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X-0.15,Y2,facecolor = "#ff0000",label = "Minas Gerais", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X ,Y3,facecolor = "#0a14c8",label = "Bahia", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X+0.15,Y4,facecolor = "#4795e0",label = "Rio de Janiero", width = 0.15, align = "center",edgecolor = "white")
plt.bar(X+0.3 ,Y5,facecolor = "#fcf10c",label = "Maranhao", width = 0.15, align = "center",edgecolor = "white")


plt.xticks(X,top_10,fontsize = 8,rotation = "vertical")
plt.xlabel("Modelos",fontsize = 18)
plt.ylabel('Cantidad de compras', fontsize = 18)

plt.title('Comparacion los modelos mas vendidos en las regiones mas polulares de Brasil\n', fontsize = 18)

plt.legend(loc="upper right",fontsize = 12)
plt.show()


