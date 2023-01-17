import pandas as pd

import geopandas

import numpy as np

import matplotlib

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# importando o dataframe com os dados do corona vírus



df = pd.read_csv("../input/world_covid.csv", index_col=0)



# renomeando a coluna 'name'



df = df.rename(columns={'Country_or_Region': 'name'})



# mudando o nome de países



df.loc[df.name.isin(['Hong Kong', 'US', 'Mainland China', 'UK'])]

df.at[205, 'name'] = 'United States of America'

df.at[122, 'name'] = 'China'

df.at[91, 'name'] = 'Hong Kong'

df.at[204, 'name'] = 'United Kingdom'
# mostrar quais datasets estão disponíveis



geopandas.datasets.available
# importando o dataset naturalearth_lowres



mundo = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

mundo
# inserindo a coluna geometry do naturalearth_lowres no dataframe do corona vírus



df_geo = pd.merge(df, mundo[['name', 'geometry']], on='name')

df_geo
# transformando o dataframe em um geodataframe



gdf = geopandas.GeoDataFrame(df_geo)



gdf
# criando a imagem (fig) e o gráfico (ax)



fig, ax = plt.subplots(1, 1, figsize=(10, 10))



# criando a barra de cores da legenda



from mpl_toolkits.axes_grid1 import make_axes_locatable



divider = make_axes_locatable(ax)

cax = divider.append_axes("bottom", size="10%", pad=0.1)



# código das cores personalizadas

# as cores podem ser cores padrão do matplotlib ou um hex

# cria uma escala de cores entre as cores escolhidas



cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#63bddb", "#691585"])



# caso eu queira deixar a escala 'desproporcional'



#cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(0, "#ffeded"),(0.75, "#ff0000"), (1, "#000000")])



# define os limites e normaliza os valores



#bounds = np.linspace(0, 30, 10) # retorna um array com 10 números entre 0 e 30

#norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)



norm = matplotlib.colors.BoundaryNorm([0,5,10,15,20,25,30], cmap.N) # o cmap.N faz com que todas as cores da escala apareçam



# criar um título para o gráfico



ax.set_title('% de mortes por COVID-19 por país', color='#4a4a4a', fontsize=20, pad=10)



# criando um plot do geodataframe de fundo (com todos os países e sem a antártida)



mundo[mundo.continent != 'Antarctica'].plot(ax=ax, facecolor='#d4d4d4', edgecolor='#ffffff', linewidth=0.4)



# criando um plot do geodataframe com os dados



gdf.plot(column='% deaths', ax=ax, legend=True, cax=cax, cmap=cmap, edgecolor='#ffffff', linewidth=0.4,

        legend_kwds={'label': "% de mortes por país", 'orientation': "horizontal"},

         norm=norm) # plot com o geodataframes com informações



plt.show()