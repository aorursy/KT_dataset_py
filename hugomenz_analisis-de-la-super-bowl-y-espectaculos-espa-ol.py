#importar pandas

import pandas as pd
# Cargar los archivos CSV como DataFrames

super_bowls = pd.read_csv("../input/super_bowls.csv")

tv =pd.read_csv("../input/tv.csv") 

halftime_musicians = pd.read_csv("../input/halftime_musicians.csv")
#Muestra las primeras 5 filas de cada DataFrame

display(super_bowls.head())

display(tv.head())

display(halftime_musicians.head())
#Resumen de los datos a inspeccionar de cada DataFrame

super_bowls.info()

tv.info()

halftime_musicians.info()
# Para poder graficar algo, tendremos que cargar pyplotlib

from matplotlib import pyplot as plt

%matplotlib inline



#Utilizamos el estilo de 'seaborn' por ejemplo

plt.style.use('seaborn')

# BUSCAR COMANDO PARA ACCEDER A TODOS LOS ESTILOS



#Creamos un histograma con los puntos combinados

#para ello accedemos a la columna 'combined_pts' del DataFrame 'super_bowl'

#asi vemos una distribucion de los puntos y podemos elegir cuales son los 

#puntos extremos

plt.hist(super_bowls.combined_pts)

plt.xlabel('Combined Points')

plt.ylabel('Number of Super Bowls')

plt.show()
# Muestra las Super Bowls con las puntuaciones mas altas y mas bajas

# de puntos combinados

display(super_bowls[super_bowls['combined_pts'] > 70])

display(super_bowls[super_bowls['combined_pts'] < 30])
plt.hist(super_bowls.difference_pts)



plt.xlabel("Point Difference")

plt.ylabel("Super Bowl game")

plt.title("Point Difference Distribution")



plt.show()
max_dif=super_bowls[super_bowls['difference_pts'] > 40]

min_dif=super_bowls[super_bowls['difference_pts'] < 5]



display(max_dif)

display(min_dif)
def busca_busca(df,columna):

    indice_min= df[columna].idxmin()

    objeto_min= display(df[indice_min : indice_min + 1])

    

    indice_max = df[columna].idxmax()

    objeto_max=display(df[indice_max : indice_max + 1])

    

    return objeto_min,objeto_max 
display(busca_busca(super_bowls,'difference_pts'))
# Unir los datos de los partidos y la television omitiendo SB I porque se

#dividio en 2 canales. (ver index 51 y 52)

games_tv = pd.merge(tv[tv['super_bowl'] > 1], super_bowls, on='super_bowl')



#Import seaborn

import seaborn as sns



# Create a scatter plot with a linear regression model fit

sns.regplot(x=games_tv.difference_pts, y=games_tv.share_household, data=games_tv)
# Creamos una figura con 3x1 subplot y activamos el subplot superior

plt.subplot(3, 1, 1)

plt.plot(tv.super_bowl, tv.avg_us_viewers, color='#648FFF')

plt.title('Average Number of US Viewers')



# activamos el subplot intermedio

plt.subplot(3, 1, 2)

plt.plot(tv.super_bowl, tv.rating_household, color='#DC267F')

plt.title('Household Rating')



# activamos el subplot inferior

plt.subplot(3, 1, 3)

plt.plot(tv.super_bowl, tv.ad_cost, color='#FFB000')

plt.title('Ad Cost')

plt.xlabel('SUPER BOWL')



# modificamos el espacio entre subplots..

plt.tight_layout()
# Mostramos todos los musicos que han actuado en los halftime shows

halftime_musicians
# Contar todas las apariciones en halftime show para cada artista

# y los ordenamos por numero de apariciones

halftime_appearances = halftime_musicians.groupby('musician').count()['super_bowl'].reset_index()

halftime_appearances = halftime_appearances.sort_values('super_bowl', ascending=False)



# Mostramos los musicos con mas de una aparicion en los halftime shows

halftime_appearances[halftime_appearances.super_bowl>1]
# Filtramos la mayor parte de las bandas

no_bands = halftime_musicians[~halftime_musicians.musician.str.contains('Marching')]

no_bands = no_bands[~no_bands.musician.str.contains('Spirit')]



# Creamos un histograma con el numero de canciones por actuacion

most_songs = int(max(no_bands['num_songs'].values))

most_songs
plt.hist(no_bands.num_songs.dropna(), bins=most_songs)

plt.xlabel('Number of Songs Per Halftime Show Performance')

plt.ylabel('Number of Musicians')

plt.show()
# Ordenamos los musicos non-band por numero de canciones por aparicion

no_bands = no_bands.sort_values('num_songs', ascending=False)

# ...y mostramos el top 15

display(no_bands.head(15))