import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

datos=pd.read_csv('../input/imdb-data/IMDB-Movie-Data.csv')

df=pd.DataFrame(datos)
df.groupby("Genre")["Votes"].sum().plot(kind = 'barh', legend ='reverse',figsize=(4,40), color = "aquamarine")

plt.title("Votos por Genero de Pelicula")
df.Votes.groupby(df.Year).sum().plot(kind = 'pie',cmap= 'viridis',figsize=(8,8))

plt.title("Votos por Año")

sns.lmplot(x='Year',y= 'Runtime (Minutes)', data=df,fit_reg = False ,hue ="Runtime (Minutes)", legend= False,palette="rocket_r",size = 8)

plt.title("Tiempo de pelicula vs Años ")
relacion=sns.lmplot(x= 'Revenue (Millions)',y= 'Votes' ,data= df,size=8,line_kws={'color': 'orange'})

plt.title("Relación número de Votos con el Ingreso")
sns.lmplot(x='Year',y= 'Rating', data=df,fit_reg = False ,hue ="Runtime (Minutes)", legend= False,palette="rocket_r",size = 8)

plt.title("Rating vs año")
relacion=sns.lmplot(x= 'Revenue (Millions)',y= 'Runtime (Minutes)' ,data= df,size=8, line_kws={'color': 'violet'})

plt.title("Relación entre el Tiempo de Pelicula y su Ingreso")