import pandas as pd

import numpy as np

import nltk
from IPython.display import display_html

from nltk.corpus import stopwords
# nltk.download('stopwords')
# La creacion de la lista de stopsWords customizada se debe a que en cada topico existe un grupo de palabras que no 

# aportan informacion relevante para el analisis, en este caso, por ejemplo, la palabra wine, no tiene relevancia

# ya es un pablara que hace referencia al objeto de analisis direcamente sin destacar niguna propiedad del el

customStopWords = ['wine', 'flavors']
#Stop Words que incluyen por defecto la libreria NLTK

stopWords = stopwords.words('english') + customStopWords
# Esta funcion nos va a permitir mas adelante

# visualizar algunos DataFrames de manera simultanea

def mydisplay(dfs, names=[], index=False):



    count = 0

    maxTables = 6 

    

    if len(dfs) < maxTables:

        maxTables = len(dfs)

        

    

    if not names:

        names = [x for x in range(len(dfs))]



    html_str = ''

    html_th = ''

    html_td = ''

    

    # contruccion del esqueleto html para mostrar dataFrames simultaneamente

    # de esta manera se facilita la visualizacion para evaluar los datos y hacer comparaciones

    for df, name in zip(dfs, names):

        if count < (maxTables):

            html_th += (''.join(f'<th style="text-align:center">{name}</th>'))

            html_td += (''.join(f'<td style="vertical-align:top"> {df.to_html(index=index)}</td>'))

            # Este contador lo uso para cerrar las filas en un maximo de 6 columnas

            count += 1 

        else:

            #Cuando el contador llega a 6 se incluyen las columnas a la tabla

            html_str += f'<tr>{html_th}</tr><tr>{html_td}</tr>'

            # Se reinicia el string con el dato que esta en las variables del for, para no perderlo

            html_th = f'<th style="text-align:center">{name}</th>'

            html_td = f'<td style="vertical-align:top"> {df.to_html(index=False)}</td>'

            # Finalmente reinicio el contador

            count = 1

    

    # Este condicionl esta aca por que probablemente el for termine de recorrer los datos y el contador no quede

    # en cero, esto implica que la tabla html que estamos construyendo no quede apropiadamente cerrada

    # para prevenir esto, aca puse el condicional que va a cerrar el tag cuando el contador no sea cero

    # aclaroo que esta funcion tiene un bug, si lo econtras y/o lo arreglas escribime arzanico@hotmail.com

    html_str += f'<tr>{html_th}</tr><tr>{html_td}</tr>'

        

        

    

    html_str += f'<table>{html_str}</table>'

    html_str = html_str.replace('table','table style="display:inline"')

    display_html(html_str, raw=True)
# df = pd.read_csv('../data/winemag-data-130k-v2.csv', index_col = None)

df = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv', index_col = None)



df.drop(columns=['Unnamed: 0'], inplace = True)
df.info()
#Consultamos si hay algun missing data, verificacion, por columna de al menos uno.

pd.DataFrame(pd.isnull(df).any()).rename(columns={0:'Bool'}).reset_index(drop=False).Bool.any()
mydisplay([pd.DataFrame(pd.isnull(df[c]).value_counts()) for c in df.columns], [c for c in df.columns], index=True)    
#Pasamos algunos datos de texto a minuscula

df.description = df.description.apply(lambda x : x.lower())

df.dropna(how='any', inplace=True)

df.variety = df.variety.apply(lambda x : x.lower())
#Sample data of all columns

df.sample(1)
#Amount of uniques varieties

print('Se encuentrar ',df.variety.unique().size, ' variedades diferentes')

print('Hay vinos de ',df.country.unique().size, ' paises', end=' ')

print('y',df.winery.unique().size, ' bodegas diferentes')
top20Rw = pd.DataFrame(df.variety.value_counts()[:20]).reset_index(drop=False).rename(columns={'index':'varietal','variety':'recuento'})
top20Rw
#Aqui quitamos las stopsWords

desc = df[['description', 'variety']].copy()

desc.description =  desc.description.apply(lambda x : ' '.join([w for w in x.split() if w.lower() not in stopWords]))
print('Numero medio de palabras con las que se describen los varietales')

round(desc.description.apply(lambda x : len(x.split())).mean(), 2)
pinotNoir = pd.Series((' '.join([x for x in desc.groupby('variety').get_group('pinot noir').description]).split()))
print('Evaluacion de las descripciones del varital Pinot Noir')

print('Cantidad de palabras distintas usadas :', pinotNoir.unique().size)

print('Aparicion promedio de palabras : ', round(pinotNoir.value_counts().mean(), 2))

print('Lista ordenada de las palabras usadas (top 5):')

pinotNoir.value_counts()[:5]
#Palabras econtradas en los varietales

df1 = pd.DataFrame(pd.Series(' '.join([w for w in desc.variety]).split()).value_counts()).reset_index(drop=False).rename(columns={'index':'words',0:'recuento'})
mydisplay([top20Rw,df1[:20],df1[20:40],df1[40:60],df1[60:80],df1[80:101]], index=True)
#Varietales que contienen las palabras mas usadas

dfs = list()

for w in df1.words:

    word = w

    data = pd.DataFrame(pd.Series([x for x in desc.variety if w in x]).value_counts()).reset_index(drop=False).rename(columns={'index':'word {}'.format(w),0:'recuento'})

    

    dfs.append((data,word))
#Varietals that contains the most common words. Aca la funcion de visualizacion

mydisplay([d[0] for d in dfs])#, [w[1] for w in dfs])
top20Rw.plot(kind='hist', bins=50, figsize=(12,6));
# El siguiente analisis se va a enfocar en las descripciones de los varitales.

df2 = pd.DataFrame(pd.Series(' '.join([w for w in desc.description]).split()).value_counts()).reset_index(drop=False).rename(columns={'index':'words',0:'recuento'})
print('Numero medio de palabras con las que se describen los varietales')

round(desc.description.apply(lambda x : len(x.split())).mean(), 2)
print('Cantidad de palabras distintas usadas :', df2.words.unique().size)

print('Aparicion promedio de palabras : ', round(df2.words.value_counts().mean(), 2))

print('Lista ordenada de las palabras usadas (top 5):')

df2[:5]