import pandas as pd



df = pd.read_csv('../input/The_top-5000_frequent_Spanish_words_in_Twitter_for_331_cities_in_the_Spanish-speaking_world.csv')
df.head()
df.shape
df = pd.read_csv('../input/The_top-5000_frequent_Spanish_words_in_Twitter_for_331_cities_in_the_Spanish-speaking_world_2.csv')
df.head()
from collections import Counter



cnt = Counter()



for country in df['Country'].tolist():

     cnt[country] += 1



print(cnt)
for item in list(zip(df['Country'],df['city_ascii'])):

    print (item[0]+', '+item[1])
import folium



m = folium.Map(tiles= 'cartodbpositron')  

m
cities = list(zip(df['city_ascii'],df['lat'],df['lng']))
cities[0]
for city in cities:

    folium.Marker(

    popup=city[0],

    location=[city[1],city[2]]

    ).add_to(m)

m
df['lines']
from gensim import corpora

df['tokens'] = df['lines'].str.split(',')
df['tokens']
dictionary = corpora.Dictionary(df['tokens'])
print(dictionary)
df['bow'] = df['tokens'].apply(lambda x: dictionary.doc2bow(x))
df['bow']
from gensim import models

tfidf = models.TfidfModel(df['bow']) 
df['tfidf'] = df['bow'].apply(lambda x: tfidf[x]) 
df['tfidf']
def get_top_tfidf(line,dictionary):

    line = [(dictionary[a],b) for (a,b) in line]

    line = [(float(b),a) for (a,b) in line]

    line = list(sorted(line,reverse=True))[:10] 

    line = [b for (a,b) in line]

    return line
df['top_tfidf'] = df['tfidf'].apply(lambda line: get_top_tfidf(line,dictionary))
import random

for i in random.sample((list(zip(df['Country'],df['city_ascii'],df['top_tfidf']))),10):

    print(i)
lsi = models.LsiModel(df['tfidf'],num_topics=200)
from gensim import similarities

index = similarities.MatrixSimilarity(lsi[df['tfidf']]) 
query = 'Somos pacífico, estamos unidos Nos une la región La pinta, la raza y el don del sabor Somos pacífico, estamos unidos Nos une la región La pinta, la raza y el don del sabor  Ok! si por si acaso usted no conoce En el pacífico hay de todo para que goce Cantadores, colores, buenos sabores Y muchos santos para que adores Es toda una conexión Con un corrillo chocó, valle, cauca Y mis paisanos de nariño Todo este repertorio me produce orgullo Y si somos tantos Porque estamos tan al cucho (en la esquina) Bueno, dejemos ese punto a un lado Hay gente trabajando pero son contados Allá rastrillan, hablan jerguiados Te preguntan si no has janguiado (hanging out) Si estas queda’o Si lo has copiado, lo has vacilado Si dejaste al que está malo o te lo has rumbeado Hay mucha calentura en buenaventura Y si sos chocoano sos arrecho por cultura, ey!  Somos pacífico, estamos unidos Nos une la región La pinta, la raza y el don del sabor Somos pacífico, estamos unidos Nos une la región La pinta, la raza y el don del sabor Unidos por siempre, por la sangre, el color Y hasta por la tierra No hay quien se me pierda Con un vínculo familiar que aterra Característico en muchos de nosotros Que nos reconozcan por la mamá Y hasta por los rostros Étnicos, estilos que entre todos se ven La forma de caminar  El cabello y hasta por la piel Y dime quién me va a decir que no Escucho hablar de san pacho Mi patrono allá en quibdo, ey! Donde se ven un pico y juran que fue un beso Donde el manjar al desayuno es el plátano con queso Y eso que no te he hablado de buenaventura Donde se baila el currulao, salsa poco pega’o Puerto fiel al pescado Negras grandes con gran tumba’o Donde se baila aguabajo y pasillo  En el lado del río (ritmo folclórico) Con mis prietillos Somos pacífico, estamos unidos Nos une la región La pinta, la raza y el don del sabor Somos pacífico, estamos unidos Nos une la región La pinta, la raza y el don del sabor Es del pacífico, guapi, timbiquí, tumaco El bordo cauca Seguimos aquí con la herencia africana Más fuerte que antes Llevando el legado a todas partes De forma constante Expresándonos a través de lo cultural Música, artes plástica, danza en general Acento golpia’o al hablar El 1, 2,3 al bailar Después de eso seguro hay muchísimo más Este es pacífico colombiano Una raza un sector Lleno de hermanas y hermanos Con nuestra bámbara y con el caché (bendición, buen espíritu) Venga y lo ve usted mismo Pa vé como es, y eh! Piense en lo que se puede perder, y eh! Pura calentura y yenyeré, y eh!'.split()
query_bow = dictionary.doc2bow(query)
query_bow
query_lsi = lsi[query_bow] 
sim_scores = index[query_lsi]
df['sim_score'] = sim_scores
df = df.dropna()
df = df.sort_values(by=['sim_score'], ascending=False)[:10]
similar_cities = list(zip(df['Country'],df['city_ascii'],df['lat'],df['lng'],df['sim_score']))

for city in similar_cities:

    print(city)
m = folium.Map(tiles= 'cartodbpositron')  



for city in similar_cities:

    folium.Marker(

    popup=city[0],

    location=[city[2],city[3]]

    ).add_to(m)



m.fit_bounds(m.get_bounds())



m