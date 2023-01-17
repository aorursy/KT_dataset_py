import pandas as pd

import numpy as np

import seaborn as sns

path = ''

df= pd.read_csv('../input/train.csv')

#Fuente: https://es.investing.com/currencies/usd-mxn-historical-data

#AGREGAR EL PATH CORRESPONDIENTE

path1 = ''

dolar= pd.read_csv('../input/Datos histricos USD_MXN.csv')

dolar.columns = ["fecha", "last", "opening", "max", "min", "std"]

dolar.drop(["std", "opening", "opening", "max", "min"], axis=1 ,inplace=True)

#Paso a formato fecha correctamente y luego creo columnas mes y año

dolar['date'] = pd.to_datetime(dolar['fecha'], format="%b %Y")

dolar.drop(["fecha"], axis=1, inplace=True)

dolar.columns= ['price', "date"]

#El csv ponia los numeros con punto en vez de coma.

dolar['price'] = dolar['price'].replace(',', '.', regex=True).astype(float)

dolar["year"] = dolar.date.dt.year

dolar['month'] = dolar.date.dt.month

dolar.drop(["date"], axis=1, inplace=True)

df['fecha'] = pd.to_datetime(df['fecha'])

df['year'] = df['fecha'].dt.year

df['month']= df['fecha'].dt.month

#Agrego la columna precio en dolar por propiedad, en el dataframe original

newDf= pd.merge(df, dolar, on=['year', 'month'], how='left')

newDf["dollar_price"] = newDf.apply(lambda row: row["precio"]/row["price"], axis=1)

#Filtro dejando las cosas útiles.

#Para agregar una columna, se debe agregar aca y luego agregarla al merge

df = newDf.filter(["id", "tipodepropiedad", "habitaciones",'garages', 'banos', 'ciudad',

                   'provincia','metroscubiertos', 'metrostotales', 'idzona', 'fecha',

                   'gimnasio', 'usosmultiples', 'piscina', 'escuelascercanas',

                   'centroscomercialescercanos', 'dollar_price','precio' ])

def asignarMetros(metroscubiertos, metrostotales):

    if (metroscubiertos != 0 and metrostotales == 0):

        return metroscubiertos

    else:

        return metrostotales

    

#Arreglando, no pueden haber propiedades con metros cubiertos pero sin metros totales

df['metrostotales'].fillna(0, inplace=True)

df['metrostotales'] = df.apply(lambda x: asignarMetros(x['metroscubiertos'],x['metrostotales']),axis=1)

df.head()

#Filtrando el garage, lotes, hospedaje, otros

filtroTerreno = df["tipodepropiedad"].isin(["Huerta", "Nave industrial", "Terreno", "Terreno comercial", 

                                            "Bodega comercial", "Terreno industrial"])

dfTerreno = df[filtroTerreno]

dfTerreno.tipodepropiedad.value_counts()

#Dropeo todos los terrenos que tienen habitaciones ya que si las tienen no serian terrenos.

dfTerreno = dfTerreno[dfTerreno["habitaciones"].isna()]

dfTerreno.tipodepropiedad.value_counts()

#Limpiando las propiedades tipo casa, no pueden no tener al menos una habitacion o baño y no pueden

#tener metros no cubiertos

filtro1 = df["tipodepropiedad"].isin(["Apartamento", "Casa", "Casa en condominio",

                                        "Casa uso de suelo", "Rancho", "Quinta Vacacional"])

dfCasas = df[filtro1]

dfCasas = dfCasas[np.isfinite(dfCasas['habitaciones'])]

dfCasas = dfCasas[np.isfinite(dfCasas['banos'])]

dfCasas = dfCasas[np.isfinite(dfCasas['metroscubiertos'])]

dfCasas.garages.fillna(0, inplace=True)

df = pd.merge(dfCasas, dfTerreno, on=['id', 'tipodepropiedad', 'habitaciones', 'garages', 'banos', 'metroscubiertos'

                                      ,'ciudad','provincia','metrostotales','idzona','fecha', 'gimnasio','usosmultiples',

                                       'piscina','escuelascercanas','centroscomercialescercanos', 'dollar_price', 'precio'], how='outer')

filtro = newDf['provincia'].isin(['Distrito Federal','Veracruz','Oaxaca'])

df=newDf[filtro]

sns.lineplot(x="year",y='dollar_price',data=df,hue='provincia',sizes=100)
filtro = newDf['provincia'].isin(['Distrito Federal','Veracruz','Oaxaca'])

df=newDf[filtro]

sns.lineplot(x="year",y='precio',data=df,hue='provincia')
filtro1 = df["tipodepropiedad"].isin(["Apartamento"])



dfCasa = df[filtro1]



g = dfCasa[["habitaciones"]].plot.hist(bins=10)

g.set_title("Cantidad de habitaciones en apartamentos ", fontsize=18)

g.set_xlabel("Habitaciones",fontsize=18)

g.set_ylabel("Frecuencia", fontsize=18)
#Arranco 

df["year"] = df.fecha.dt.year

df['month'] = df.fecha.dt.month

grouped = df.groupby('month').agg({'year':'count'})

grouped.reset_index(inplace=True)

grouped.columns = ['month', 'totalSold']

#Cantidad de ventas por mes

sns.set(font_scale=2)

sns.barplot(x="month", y='totalSold',data=grouped, linewidth=2.5)
grouped = df.groupby(['month', 'year'])

forPlot = grouped.agg({'year':{'totalSold':'count'}})

#sns.relplot(x="year", col_wrap=5,y='dollar_price',kind='line',

#            data=df,hue='provincia',col='tipodepropiedad',height=3, linewidth=2.5)



forPlot.reset_index(inplace=True)

forPlot.columns = ['month', 'year', 'totalSoldPerYear']

sns.set(font_scale=2)

sns.relplot(x="month", col_wrap=5,y='totalSoldPerYear',kind='line',

            data=forPlot,col='year',height=10, aspect=1)
g = sns.catplot(x="month", y="totalSoldPerYear",

                 hue="year",

                 data=forPlot, kind="bar",

                 height=6, aspect=2);
filtro = forPlot["year"].isin(["2012", "2013", "2014", "2015"])

forPlot1215= forPlot[filtro]

g = sns.catplot(x="month", y="totalSoldPerYear",

                 hue="year",

                 data=forPlot1215, kind="bar",

                 height=6, aspect=2);
g = sns.catplot(x="month", y="totalSoldPerYear",

                 data=forPlot1215, kind="bar",

                 height=6, aspect=2);
pivot = forPlot1215.pivot("month", "year", "totalSoldPerYear")

sns.set(font_scale=1.5)

ax = sns.heatmap(pivot, cmap="YlGnBu", linewidths=.5)
sns.set(font_scale=2)

sns.relplot(x="month", col_wrap=5,y='totalSoldPerYear',kind='line',

            data=forPlot1215,col='year',height=10, aspect=1)
id = df.dollar_price.idxmax(axis = 1, skipna = True) 

df.iloc[id]
d0=df.groupby(['ciudad','provincia']).agg({'precio':'mean'})

sns.set_context("paper", font_scale=1.9)

d0.reset_index(inplace=True)

d0.sort_values(by=['precio'],ascending = [False],inplace = True)

d0.reset_index(inplace=True)

dh = d0.iloc[0:20:]

display(dh)

g0=sns.catplot(y="ciudad",x='precio',kind='strip',data=dh,orient='h',hue='provincia',height=6,aspect=2.1,sizes=(100,10))

g0.set_axis_labels("Precio moneda mexicana","Ciudad")


dg = newDf.groupby('antiguedad').agg({'precio':'mean'})

dg.reset_index(inplace=True)



dh=dg.iloc[0:10].mean()

dh['01 / 10 Años'] = dh['precio']

dh.drop('antiguedad',inplace=True)

dh.drop('precio',inplace=True)

dh.to_frame()





dh2=dg.iloc[10:20].mean()

dh2['10 / 20 Años'] = dh2['precio']

dh2.drop('antiguedad',inplace=True)

dh2.drop('precio',inplace=True)

dh2.to_frame()



dh3=dg.iloc[20:30].mean()

dh3['20 / 30 Años'] = dh3['precio']

dh3.drop('antiguedad',inplace=True)

dh3.drop('precio',inplace=True)

dh3.to_frame()





dh4=dg.iloc[30:40].mean()

dh4['30 / 40 Años'] = dh4['precio']

dh4.drop('antiguedad',inplace=True)

dh4.drop('precio',inplace=True)

dh4.to_frame()





dh5=dg.iloc[40:50].mean()

dh5['40 / 50 Años'] = dh5['precio']

dh5.drop('antiguedad',inplace=True)

dh5.drop('precio',inplace=True)

dh5.to_frame()





dh8=dg.iloc[50:60].mean()

dh8['50 / 60 Años'] = dh8['precio']

dh8.drop('antiguedad',inplace=True)

dh8.drop('precio',inplace=True)

dh8.to_frame()





dh6=dg.iloc[60:70].mean()

dh6['60 / 70 Años'] = dh6['precio']

dh6.drop('antiguedad',inplace=True)

dh6.drop('precio',inplace=True)

dh6.to_frame()





dh7=dg.iloc[70:80].mean()

dh7['70 / 80 Años'] = dh7['precio']

dh7.drop('antiguedad',inplace=True)

dh7.drop('precio',inplace=True)

dh7.to_frame()







frames = [dh, dh2, dh3,dh4,dh5,dh6,dh7,dh8]



result = pd.concat(frames)



apa = result.to_frame()

apa.reset_index(inplace=True)

apa.columns = ['Antiguedad','Precio']





apa.sort_values(by=['Precio'],ascending = [True],inplace = True)

sns.set(style="darkgrid")

sns.set_context("paper", font_scale=5.9)

sns.catplot(x="Antiguedad",y='Precio',kind='point',data=apa,height=14,aspect=3.6)

#CANTIDAD DE PISCINAS POR ZONA, EN FORMA ASCENDENTE.

da = df.groupby('provincia').agg({'piscina':'sum'})

da.sort_values(by=['piscina'],ascending = [False],inplace = True)

da.reset_index(inplace=True)

sns.set_context("paper", font_scale=1.9) 

g4=sns.catplot(y='provincia',x='piscina',kind='bar',

            data=da,height=8, aspect=2.2,orient='h')



g4.set_axis_labels( "Cantidad De Piscinas","Provincia")
filtro1 = df["tipodepropiedad"].isin(["Apartamento"])



dfCasa = df[filtro1]



g = sns.distplot(dfCasa['metrostotales'], bins = 70, color='orange')

g.set_title("Apartamentos", fontsize=18)

g.set_xlabel("Metros Totales",fontsize=18)

g.set_ylabel("Frecuencia", fontsize=18)
df2 = df

df2['fecha_month'] = df2['fecha'].dt.month

df2['fecha_year'] = df2['fecha'].dt.year

for_heatmap_mexican = df2.pivot_table(index='fecha_year', columns='fecha_month', values='dollar_price', aggfunc='mean')

for_heatmap_dollar = df2.pivot_table(index='fecha_year', columns='fecha_month', values='precio', aggfunc='mean')



h = sns.heatmap(for_heatmap_dollar, cmap="YlGnBu")
def cambiarNombre(word):

    if(word == 1.0):

        return "Con gimnasio"

    else:

        return "Sin gimnasio"

    

filtroGimnasios = df["gimnasio"].isin([1.0])

dfGim = df[filtroGimnasios]

#g = sns.catplot(x='tipodepropiedad', y='dollar_price', kind="bar",  data=dfGim, height=4,aspect=2)

group = dfGim.groupby("tipodepropiedad").agg({"tipodepropiedad":{"amount" : "count"}})

group.reset_index(inplace=True)

group.columns = ["tipodepropiedad", "amount"]

group = group[group["tipodepropiedad"].isin(["Apartamento", "Casa", "Casa en condominio"])]



g = sns.catplot(x='tipodepropiedad', y='amount',

                kind="bar",  data=group, height=4,aspect=2)
dfPools = df[df.tipodepropiedad.isin(["Apartamento", "Casa", "Casa en condominio"])]

dfPools["gimnasio"] = dfPools.apply(lambda x: cambiarNombre(x['gimnasio']),axis=1)

sns.set(font_scale=1.3)

g = sns.catplot(x='tipodepropiedad', y='dollar_price', hue = "gimnasio",

                kind="boxen",  data=dfPools, height=4,aspect=2)

g.fig.set_size_inches(15,10)
dfUsosMultiples=df



def cambiarNombre(word):

    if(word == 1.0):

        return "Con usos multiples"

    else:

        return "Sin usos multiples"

    

usosMultiples = dfUsosMultiples["usosmultiples"].isin([1.0])

dfUM = dfUsosMultiples[usosMultiples]

group = dfUM.groupby("tipodepropiedad").agg({"tipodepropiedad":{"amount" : "count"}})

group.reset_index(inplace=True)

group.columns = ["tipodepropiedad", "amount"]

group = group[group["tipodepropiedad"].isin(["Apartamento", "Casa", "Casa en condominio"])]



g = sns.catplot(x='tipodepropiedad', y='amount',

                kind="bar",  data=group, height=4,aspect=2)
dfUM = df[df.tipodepropiedad.isin(["Apartamento", "Casa", "Casa en condominio"])]

dfUM["usosmultiples"] = dfUM.apply(lambda x: cambiarNombre(x['usosmultiples']),axis=1)

sns.set(font_scale=1.3)

g = sns.catplot(x='tipodepropiedad', y='dollar_price', hue = "usosmultiples",

                kind="boxen",  data=dfUM, height=4,aspect=2)

g.fig.set_size_inches(15,10)
corr = df.corr()



s = sns.heatmap(corr, cmap="YlGnBu")
filtro1 = df["tipodepropiedad"].isin(["Apartamento"])

df2 = df[filtro1]



g = sns.lineplot(x="metrostotales",y='dollar_price',data=df2)



g.set_title("Apartamentos", fontsize=18)

g.set_xlabel("Metros totales",fontsize=12)

g.set_ylabel("Precio en dolares", fontsize=12)
#Limpiando las propiedades tipo casa, no pueden no tener al menos una habitacion o baño y no pueden

#tener metros no cubiertos

filtro1 = df["tipodepropiedad"].isin(["Casa", "Apartamento", "Edificio", "Casa en condominio", 

                                      "Casa uso de suelo", "Quinta Vacacional", "Duplex", "Villa", "Rancho"])

dfCasas = df[filtro1]

dfCasas.tipodepropiedad.value_counts()



dfCasas = dfCasas[np.isfinite(dfCasas['habitaciones'])]

dfCasas = dfCasas[np.isfinite(dfCasas['banos'])]

dfCasas = dfCasas[np.isfinite(dfCasas['metroscubiertos'])]

dfCasas.garages.fillna(0, inplace=True)

df = pd.merge(dfCasas, dfTerreno, on=['id', 'tipodepropiedad', 'habitaciones', 'garages', 'banos', 'metroscubiertos'

                                      ,'ciudad','provincia','metrostotales','idzona','fecha', 'gimnasio','usosmultiples',

                                       'piscina','escuelascercanas','centroscomercialescercanos', 'dollar_price'], how='outer')



group = dfCasas.groupby("tipodepropiedad").agg({"banos":{"Promedio" :"mean"}, "habitaciones": {"habitacionesPromedio":"mean"},

                                           "tipodepropiedad":{"CantidadDePropiedades":"count"}, 

                                           "dollar_price":{"precioEnDolar":"mean"}})

group.reset_index(inplace=True)

group.columns = ["tipodepropiedad", "banosPromedio", "habitacionesPromedio", "cantidadDePropiedades", "precioDolarPromedio"]



plotting1 = group.sort_values('banosPromedio')

g = sns.catplot(x='tipodepropiedad', y='banosPromedio',

                kind="bar",  data=plotting1, height=4,aspect=2)

g.set_xticklabels(rotation=65, horizontalalignment='right')

g.set(xlabel='Tipos de domicilios', ylabel='Cant de baños promedio')

plotting2 = group.sort_values('habitacionesPromedio')

g = sns.catplot(x='tipodepropiedad', y='habitacionesPromedio',

                kind="bar",  data=plotting2, height=4,aspect=2)

g.set_xticklabels(rotation=65, horizontalalignment='right')

g.set(xlabel='Tipos de domicilios', ylabel='Cant de habitaciones promedio')
filtroBanos = dfCasas["tipodepropiedad"].isin(["Apartamento","Casa"])

dfBanos = dfCasas[filtroBanos]

sns.set(font_scale=1.3)

g.fig.set_size_inches(20,15)

g = sns.catplot(x='tipodepropiedad', y='dollar_price', hue = "banos", palette="bright",

                kind="boxen",  data= dfBanos, height=4,aspect=2)

g.set(xlabel='', ylabel='Precio en dolar')
#CANTIDAD DE PISCINAS POR ZONA, EN FORMA ASCENDENTE.

da = df.groupby('provincia').agg({'piscina':'sum'})

da.reset_index(inplace=True)

d0=da[(da['piscina']>0)&(da['piscina']<8)]

d1=da[(da['piscina']>=8)&(da['piscina']<70)]

sns.catplot(x='provincia',y='piscina',kind='bar',

            data=d0,height=4.5, aspect=2.9)

sns.catplot(x='provincia',y='piscina',kind='bar',

            data=d1,height=5.5, aspect=2.9)