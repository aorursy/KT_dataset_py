# importando pacotes



import geopandas

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt
# mostrar quais datasets estão disponíveis



geopandas.datasets.available
# mostrar onde está armazenado odataset naturalearth_lowres



geopandas.datasets.get_path('naturalearth_lowres')
# importando o dataset naturalearth_lowres



mundo = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

mundo
# importando o dataframe com os dados do corona vírus



df = pd.read_csv("../input/covid19-teste/south_america_covid.csv", index_col=0)

df = df.rename(columns={'Country_or_Region': 'name'})

df
# inserindo a coluna geometry do naturalearth_lowres no dataframe do corona vírus



df_geo = pd.merge(df, mundo[['name', 'geometry']], on='name')

df_geo
# transformando o dataframe em um geodataframe



gdf = geopandas.GeoDataFrame(df_geo)



gdf
#código da legenda



from mpl_toolkits.axes_grid1 import make_axes_locatable



fig, ax = plt.subplots(1, 1, figsize=(10, 10)) # o figsize tem que ficar aqui

divider = make_axes_locatable(ax)

cax = divider.append_axes("bottom", size="10%", pad=0.1) # bottom, top, right, left



# código das cores personalizadas

# as cores podem ser cores padrão do matplotlib ou um hex

# cria uma escala de cores entre as cores escolhidas



cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#63bddb", "#691585"])



# caso eu queira deixar a escala 'desproporcional'

#cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(0, "#ffeded"),(0.75, "#ff0000"), (1, "#000000")])



# define os limites e normaliza os valores



bounds = np.linspace(0, 15, 5) # retorna um array com 10 números entre 0 e 30

norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

#norm = matplotlib.colors.BoundaryNorm([0,5,10,15], cmap.N) # o cmap.N faz com que todas as cores da escala apareçam



# criar um título para o gráfico



ax.set_title('% de mortes por COVID-19 por país', color='#4a4a4a', fontsize=20, pad=10)



# código do plot



mundo[mundo.continent == ('South America')].plot(ax=ax, facecolor='#d4d4d4', edgecolor='#ffffff', linewidth=0.4)



gdf.plot(column='% deaths', ax=ax, legend=True, cax=cax, cmap=cmap, edgecolor='#ffffff',

        legend_kwds={'label': "% de mortes por país", 'orientation': "horizontal"}, norm=norm)



plt.show()