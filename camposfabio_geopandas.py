import pandas as pd

import matplotlib.pyplot as plt

import geopandas as gpd
# set the filepath and load

file_shp = "/kaggle/input/bc250-municipio-a/BC250_Municipio_A.shp"



#reading the file stored in variable fp

map_df = gpd.read_file(file_shp)



# check data type so we can see that this is not a normal dataframe, but a GEOdataframe



map_df.head()
# Substitui os valores faltantes por 0

map_df['id_produto'].fillna(0, inplace = True)



# Remove as colunas abaixo que possuem apenas um valor

del map_df['anoderefer']

del map_df['id_element']
map_df.info()
# Ajuste dos tipos de dados do DataFrame

map_df['nome'] = map_df['nome'].astype('str')

map_df['nomeabrev'] = map_df['nomeabrev'].astype('str')

map_df['geometriaa'] = map_df['geometriaa'].astype('category')

map_df['geocodigo'] = map_df['geocodigo'].astype('int')

map_df['id_produto'] = map_df['id_produto'].astype('int')
plt.tight_layout()

plt.rcParams['figure.figsize'] = (15,10)

#plotting the map of the shape file preview of the maps without data in it

#fig, ax = plt.subplots(1, figsize=(15, 10))

fig = map_df.plot().get_figure()
# Localiza o municipio 'Alta Floresta D'Oeste' (código retirado da tabela Media_Alunos_Turma - Municipios_2018.xlsx)

map_df[map_df.geocodigo == 1100015]