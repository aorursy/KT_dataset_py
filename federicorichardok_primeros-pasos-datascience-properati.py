# Importar bibliotecas en esta celda



import numpy as np

import pandas as pd

import matplotlib.pylab as plt

import seaborn as sns
# La siguiente línea es para ver las imagenes dentro del notebook

%matplotlib inline

# Acá configuramos el tamaño de las figuras

plt.rcParams['figure.figsize'] = (12,8)

# Seteamos opciones de pandas sobre las columnas y su ancho

pd.set_option('max_columns', 120)

pd.set_option('max_colwidth', 5000)

df = pd.read_csv('../input/datos-properati/datos_properati.csv', parse_dates = True)

print(df.shape)

df.head(3)
# mostrá las categorías del campo property_type y mostrá la cant. de valores en esta celda

#df=pd.DataFrame(df)



df.property_type.unique()

# Filtrá el dataframe en esta celda

df_filtered = df.copy()







df_filtered = df_filtered[(df_filtered['property_type'] == 'house')  | (df_filtered['property_type'] == 'PH') | (df_filtered['property_type'] == 'apartment')] 

df_filtered.head ()



df.property_type.unique()
df_filtered.property_type.unique()
# Visualizá las categorías en esta celda



ax = sns.countplot(x="property_type", data=df_filtered)



df_filtered_nonull = df_filtered.copy()
# Quitá las instancias con valores nules en esta celdaa

df_filtered_nonull = df_filtered_nonull.dropna(subset=['surface_total_in_m2', 'surface_covered_in_m2', 'rooms', 'price_aprox_usd','price_usd_per_m2', ])

df_filtered_nonull
print(df_filtered_nonull.shape)
print(df_filtered.shape)
inst_descartadas = len(df_filtered) - len(df_filtered_nonull)

inst_descartadas
# Mostrá en esta celda los otros atributos con valores faltantes



df_filtered_nonull.columns[df_filtered_nonull.isna().any()].tolist()
# Mostrá en esta celda cuántas instancias tienen errores en la superficie

wrong_surface_rows = df_filtered_nonull[df_filtered_nonull['surface_total_in_m2']<df_filtered_nonull['surface_covered_in_m2']]

wrong_surface_rows
df_filtered = df_filtered_nonull
df_filtered.at[wrong_surface_rows.index, 'surface_total_in_m2'] = wrong_surface_rows.surface_covered_in_m2

df_filtered.at[wrong_surface_rows.index, 'surface_covered_in_m2'] = wrong_surface_rows.surface_total_in_m2

conteo_puntos = df_filtered.groupby(['lat', 'lon']).size()

conteo_puntos[conteo_puntos > 1].sort_values(ascending=False).head(10)
conteo_puntos.name = 'conteo'



df_filtered = df_filtered.join(conteo_puntos, on=['lat', 'lon'])



df_filtered = df_filtered[df_filtered.conteo <= 5]

df_filtered
df_filtered.drop('conteo', inplace=True, axis=1)
total_propiedades = df_filtered.groupby('barrio')['property_type'].count().values

total_propiedades
porcentaje_casas_ph = df_filtered[df_filtered.property_type.isin(['PH', 'house'])].groupby('barrio').count().property_type/total_propiedades

porcentaje_edificios = df_filtered[df_filtered.property_type == 'apartment'].groupby('barrio').count().property_type/total_propiedades
print("Barrios con mayor porcentaje de edificios: \n", porcentaje_edificios.sort_values()[::-1][:5])

print("Barrios con mayor porcentaje de casas y phs: \n ", porcentaje_casas_ph.sort_values()[::-1][:5])
barrios_casas = porcentaje_casas_ph.sort_values()[::-1][:5].index

barrios_edificios = porcentaje_edificios.sort_values()[::-1][:5].index

fig, axs = plt.subplots(1,2,figsize=(14,7))

df_filtered[df_filtered.barrio.isin(barrios_edificios)].property_type.value_counts().plot(x=None,y=None, kind='pie', startangle=30, ax=axs[0],autopct='%1.2f%%')

df_filtered[df_filtered.barrio.isin(barrios_casas)].property_type.value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')
df_filtered.price_usd_per_m2.describe()
# Mostrá en esta celda los cálculos



minimo=df_filtered.price_usd_per_m2.min()

maximo=df_filtered.price_usd_per_m2.max()

media=df_filtered.price_usd_per_m2.mean()

mediana=df_filtered.price_usd_per_m2.median()

desv_estandar=df_filtered.price_usd_per_m2.std()



print('maximo = {a}\nminimo: {b}\nmedia= {c}\nmediana= {d}\ndesvio estandar = {e}' .format(a=maximo, b=minimo, c=media, d=mediana, e=desv_estandar))
# Mostrá los cuartiles en esta celda

df_filtered.price_usd_per_m2.quantile([0.25,0.75])
Q1= df_filtered.price_usd_per_m2.quantile(0.25)

Q3= df_filtered.price_usd_per_m2.quantile(0.75)

IQR = Q3 - Q1

IQR
mask = df_filtered['price_usd_per_m2']

p10 = np.percentile(mask, 10)

p90 = np.percentile(mask, 90)



df_filtered = df_filtered[(df_filtered.price_usd_per_m2 >= p10) & (df_filtered.price_usd_per_m2 <= p90)]



# Realizá los calculos en esta celda

maximo=df_filtered.price_usd_per_m2.max()

media=df_filtered.price_usd_per_m2.mean()

mediana=df_filtered.price_usd_per_m2.median()

desv_estandar=df_filtered.price_usd_per_m2.std()



print('maximo = {a}\nminimo: {b}\nmedia= {c}\nmediana= {d}\ndesvio estandar = {e}' .format(a=maximo, b=minimo, c=media, d=mediana, e=desv_estandar))
plt.xticks(rotation = 90)

barrios = df_filtered.groupby('barrio').mean().sort_values('price_usd_per_m2', ascending=False).index

sns.barplot(x="barrio", y="price_usd_per_m2", order=barrios, data=df_filtered);
table = pd.pivot_table(df, values='price_usd_per_m2', index=['barrio'], aggfunc=np.mean)

mas_caros = table.sort_values('price_usd_per_m2', ascending=False).iloc[:5]

mas_caros
table = pd.pivot_table(df, values='price_usd_per_m2', index=['barrio'], aggfunc=np.mean)

mas_baratos = table.sort_values('price_usd_per_m2', ascending=True).iloc[:5]

mas_baratos
barrios_caros = mas_caros.index 

barrios_baratos = mas_baratos.index
g = sns.distplot(df_filtered[df_filtered.barrio.isin(barrios_baratos)].surface_total_in_m2, label="Barrios baratos")

sns.distplot(df_filtered[df_filtered.barrio.isin(barrios_caros)].surface_total_in_m2, label="Barrios caros")

g.set_xlim(0, 400)

plt.legend()
g = sns.distplot(df_filtered[df_filtered.barrio.isin(barrios_baratos)].rooms, label="Barrios baratos")

sns.distplot(df_filtered[df_filtered.barrio.isin(barrios_caros)].rooms, label="Barrios caros")

g.set_xlim(0, 10)

plt.legend()
g = sns.distplot(df_filtered[df_filtered.barrio.isin(barrios_baratos)].price_usd_per_m2, label="Barrios baratos")

sns.distplot(df_filtered[df_filtered.barrio.isin(barrios_caros)].price_usd_per_m2, label="Barrios caros")

plt.legend()
df_filtered['place_with_parent_names'].apply(lambda x: len(x.split("|"))).unique()
split_place_filter = df_filtered['place_with_parent_names'].apply(lambda x: len(x.split("|"))!=4)

df_filtered = df_filtered.loc[split_place_filter]
df_filtered['barrio_publicado'] = df_filtered['place_with_parent_names'].apply( lambda x: x.split("|")[3].upper())

df_filtered['barrio_publicado'] = df_filtered['barrio_publicado'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
barrios_match = (df_filtered['barrio_publicado'] != df_filtered['barrio'])

df_filtered_barrio_not_match = df_filtered.loc[barrios_match]
table = pd.crosstab(df_filtered_barrio_not_match.barrio, df_filtered_barrio_not_match.barrio_publicado)

table.idxmax(axis=1)