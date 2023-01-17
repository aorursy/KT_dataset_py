#Libraries to Data Analysis
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#Getting .csv file
df = pd.read_csv('../input/denzel-imdb-movies/filmes_denzel_V2.csv',sep=',', encoding='ISO-8859-1')
df.head(4)
#df.shape
#df.isnull().any()
import matplotlib.pyplot as plt
#%matplotlib inline
df.shape[0]
#Excluindo anos sem informação
dfGraphAno = df['Ano'] 
dfGraphAno = dfGraphAno[dfGraphAno != 0]
dfGraphAno = dfGraphAno.sort_values(ascending=True)

plt.rcParams['figure.figsize'] = (11,7)
plt.figure();
plt.title('Productions by Year')
plt.ylabel('Qtt of Productions')
plt.yticks(range(0,len(dfGraphAno)))
plt.xlabel('Year')
    
dfGraphAno.value_counts().sort_index(ascending=True).plot(kind='bar');
#Decade function, based on year
def pegaDecada(ano):
    if ano >= 1970 and ano < 1980:
        decada = 1970
    elif ano >= 1980 and ano < 1990:
        decada = 1980
    elif ano >= 1990 and ano < 2000:
        decada = 1990
    elif ano >= 2000 and ano < 2010:
        decada = 2000
    elif ano >= 2010 and ano < 2020:
        decada = 2010
    elif ano >= 2020 and ano < 2030:
        decada = 2020
    else: 
        decada = 0
        
    return decada    

#x = pegaDecada(1985)
#x
#Creating Decade column on dataframe
qtdProd = df.shape[0]
listDecada = []

for i in range(qtdProd):
    x = pegaDecada(df['Ano'][i])
    listDecada.append(x)
    #print (x)

df['Decada'] = listDecada
#df
#Deleting years without data
dfGraphDec = df['Decada'] 
dfGraphDec = dfGraphDec[dfGraphDec != 0]
dfGraphDec = dfGraphDec.sort_values(ascending=True)

plt.rcParams['figure.figsize'] = (11,7)
plt.figure();
plt.title('Productions by Decade')
plt.ylabel('Qtt of Productions')
plt.yticks(range(0,len(dfGraphDec)))
plt.xlabel('Decade')
    
dfGraphDec.value_counts().sort_index().plot(kind='bar');
#dfGraphDec.value_counts()
#Creating 'Others' category in Director column, based on qtt of movies(<=2)
dfDiretor = df 
dfDiretor.loc[dfDiretor['Diretor'].isin((dfDiretor['Diretor'].value_counts()[dfDiretor['Diretor'].value_counts() < 2]).index), 'Diretor'] = 'Outros'
#dfDiretor
#df
#Deleting Directors with less than 2 movies
dfGraphDir = dfDiretor[dfDiretor['Diretor'] != 'Outros']
dfGraphDir = dfGraphDir[dfGraphDir['Diretor'].notnull()]
#Director's graph
plt.rcParams['figure.figsize'] = (11,7)
plt.figure();
plt.title('Directors that made movies with Denzel')
plt.xlabel('Qtt of Movies')
plt.ylabel('Directors')
dfGraphDir['Diretor'].value_counts(ascending=True).plot(kind='barh');
#Movies/diretor list
dfGraphDir[['Diretor','Titulo','Ano']].sort_values(['Diretor','Ano'], ascending=[False, False])
#Movies that Denzel acted and directed simultaneously
dfDirDenz = dfGraphDir[dfGraphDir['Diretor'] == 'Denzel Washington']
dfDirDenz[['Diretor','Titulo','Ano']].sort_values(['Diretor','Ano'], ascending=[False, False])
dfGraphGen = df
dfGraphGen = dfGraphGen[dfGraphGen['Ano'] != 0]
#dfGraphGen
#Most common movie genres
#Graph considering the first genre of the movie...
plt.rcParams['figure.figsize'] = (11,7)
plt.figure();
plt.title('Genre of Productions')
plt.xlabel('Genre')
plt.ylabel('Quantity')
plt.yticks(range(0,len(dfGraphGen)))
dfGraphGen['Genero'].value_counts(ascending=False).plot(kind='bar');
#dfGraphGen
#Creating crosstable with genre and decade in order to create the graph
tabGenDec = pd.crosstab(index=dfGraphGen['Decada'], columns=dfGraphGen['Genero'])
#tabGenDec
#Creating graph...
tabGenDec.plot(kind="line",figsize=(11,7),stacked=False)
#TOP 3 longest movies
dfDurFilm = dfGraphDir[dfGraphDir['Duracao'] != 0]
dfDurFilm[['Titulo','Duracao']].sort_values(by='Duracao', ascending=False).head(3)
#TOP 3 shortest movies
dfDurFilm[['Titulo','Duracao']].sort_values(by='Duracao', ascending=True).head(3)
#TOP 10 better grades from IMDB
#df.sort_values(by='Nota IMDB', ascending=False)
dfGraphDir[['Titulo','Nota IMDB']].sort_values(by='Nota IMDB', ascending=False).head(10)
#TOP 10 worst grades from IMDb
dfGraphDir[['Titulo','Nota IMDB']].sort_values(by='Nota IMDB', ascending=True).head(10)
#TOP 10 better grades from MetaScore
dfGraphDir[['Titulo','Nota Metascore']].sort_values(by='Nota Metascore', ascending=False).head(10)
dfNotas1 = dfGraphDir[dfGraphDir['Nota Metascore'] != 0]
dfNotas1[['Titulo','Nota Metascore']].sort_values(by='Nota Metascore', ascending=True).head(10)
##dfGraphGen
##Putting grades in same scale...
dfGraphGen['Nota Metascore'] = dfGraphGen['Nota Metascore']/10
#Keeping only Movies with grades in IMDb and Metacritic simutaneously
dfNotas = dfGraphGen[['Titulo','Nota IMDB','Nota Metascore']]
dfNotas2 = dfNotas[dfNotas['Nota Metascore'] != 0]
#dfNotas2
#dfNotas2.plot(x=dfNotas2['Titulo'],  kind='bar', 
#              figsize=(11,15),title='IMDB x METASCORE', yticks=(range(0,10)), stacked=False);
dfNotas2.plot(x='Titulo', kind='bar', 
              figsize=(11,15), title='IMDB x METASCORE', yticks=(range(0,10)), stacked=False);
from matplotlib import cm
cmap = cm.get_cmap('Spectral')
#Gerando tabela cruzada(somente com gênero e nota IMDb) para criar o grafico
tabGenIMDb = pd.crosstab(index=dfGraphGen['Nota IMDB'], columns=dfGraphGen['Genero'])
#tabGenIMDb
#Creating graph...
tabGenIMDb.plot(kind="barh",figsize=(11,7), legend='reverse',  stacked=True)
#Creating crosstable with genre and metascore in order to create graph
dfx = dfGraphGen[dfGraphGen['Nota Metascore'] != 0]
tabGenIMDb = pd.crosstab(index=dfx['Nota Metascore'], columns=dfx['Genero'])
#tabGenIMDb
#Creating graph...
tabGenIMDb.plot(kind="barh",figsize=(11,7), stacked=True)
#Wordcloud and stopword libraries
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from os import path
#Read synopsis and creating unique string
sinopses = str(df['Sinopse'])
#sinopses
#Defining stopwords list
stopwords= set(STOPWORDS)

#Adding portuguese stopwords
new_words = []
with open("../input/stopwords-portuguese/stopwords_portuguese.txt", encoding='latin-1', mode='r') as f:
    [new_words.append(word) for line in f for word in line.split()]

new_stopwords = stopwords.union(new_words)
def printWordCloud(x):
    plt.figure(figsize=(20,10))
    wc = WordCloud(min_font_size=10, 
               max_font_size=300, 
               background_color='black', 
               mode="RGB",
               width=2000, 
               height=1000,
               stopwords=new_stopwords,
               normalize_plurals= True).generate(x)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
#Printing wordcloud
printWordCloud(sinopses)
titulo = str(df['Titulo'])
#titulo
printWordCloud(titulo)
