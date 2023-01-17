import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import zscore

from sklearn.preprocessing import MaxAbsScaler

from sklearn.neighbors import NearestNeighbors

from scipy.stats import randint

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from scrapy import Selector

import requests

from scrapy.crawler import CrawlerProcess

import scrapy

from scrapy.utils.project import get_project_settings
anime = pd.read_csv("../input/anime.csv")

rating = pd.read_csv("../input/rating.csv")
print(anime.shape)

print(anime.drop_duplicates().shape)

print(anime.info())

anime.tail(15)
anime.replace("Unknown", np.nan, inplace=True)

anime["episodes"] = anime["episodes"].astype(float)
print(anime.isnull().sum())

print(anime[anime.isnull().any(axis=1)].shape)

anime[anime.isnull().any(axis=1)].head()



nombres= anime[anime.isnull().any(axis=1)]

nombres = nombres["name"].values.tolist()



tipoAnime=pd.get_dummies(anime["type"]).columns

tipoAnime=tipoAnime.str.strip().unique().tolist()



genero=anime["genre"].str.get_dummies(sep=",").columns

genero=genero.str.strip().unique().tolist()
buscarURL = 'https://myanimelist.net/search/all?q='

urlAnime = []

for i in nombres:

    urlAnime.append(buscarURL + i)





class AnimeFcSpider(scrapy.Spider):

    name = 'anime_fc'



    def start_requests(self):  # start_requests method

        for url2 in urlAnime:

            yield scrapy.Request(url=url2,

                                 callback=self.parse_front)



    def parse_front(self, response):  # First parsing method

        course_links = response.xpath('//div[@class="picSurround di-tc thumb"]/a/@href')

        yield response.follow(url=course_links[0],

                              callback=self.parse_pages)



    def parse_pages(self, response):  # Second parsing method

        crs_name = response.xpath('//h1[@class="h1"]/span/text()').extract_first()

        crs_episodes = response.xpath('//td[@class="spaceit"]/span[@id="curEps"]/text()').extract_first()

        crs_rating = response.xpath('//span[@itemprop="ratingValue"]/text()').extract_first()

        crs_id = response.xpath('//input[@name="aid"]/@value').extract_first()



        crs_genre = response.xpath('//div/a/@title').extract()

        crs_genre = np.intersect1d(crs_genre, genero)

        crs_genre = ','.join(map(str, crs_genre))



        crs_type = response.xpath('//div/a/text()').extract()

        crs_type = np.intersect1d(crs_type,tipoAnime)

        crs_type = ','.join(map(str, crs_type))

       



        list_name.append(crs_name)

        list_genre.append(crs_genre)

        list_type.append(crs_type)

        list_episodes.append(crs_episodes)

        list_rating.append(crs_rating)

        list_id.append(crs_id)







list_name = list()

list_genre = list()

list_type = list()

list_episodes = list()

list_rating = list()

list_id = list()



s = get_project_settings()

s['CONCURRENT_REQUESTS_PER_IP'] = 16

s['CONCURRENT_REQUESTS_PER_DOMAIN '] = 16

s['DOWNLOAD_DELAY'] = 2.5

s['CONCURRENT_REQUESTS'] = 32

s['CONCURRENT_REQUESTS'] = 32





process = CrawlerProcess(s)  # Run the Spider

process.crawl(AnimeFcSpider)

process.start()

DataNa = pd.DataFrame({"anime_id":list_id, "name":list_name,"genre":list_genre,

                       "type":list_type, "episodes":list_episodes, "rating":list_rating})



DataNa.replace("", np.nan, inplace=True)

DataNa.replace('?', np.nan, inplace=True)



print(DataNa.shape)

print(DataNa.isnull().sum())

DataNa.head(10)
DataNa["anime_id"] = DataNa["anime_id"].astype(float)

DataNa["episodes"] = DataNa["episodes"].astype(float)

DataNa["rating"] = DataNa["rating"].astype(float)

DataNa.info()
dataNueva= pd.merge(anime, DataNa,left_on="anime_id",right_on="anime_id", how="left")

dataNueva.info()

print(anime.isnull().sum())

print(anime[anime.isnull().any(axis=1)].shape)

dataNueva.loc[dataNueva["genre_x"].isna(),"genre_x"] = dataNueva["genre_y"]

dataNueva.loc[dataNueva["type_x"].isna(),"type_x"] = dataNueva["type_y"]

dataNueva.loc[dataNueva["episodes_x"].isna(),"episodes_x"] = dataNueva["episodes_y"]

dataNueva.loc[dataNueva["rating_x"].isna(),"rating_x"] = dataNueva["rating_y"]

dataNueva.drop(["name_y", "genre_y", "type_y", "episodes_y", "rating_y"],axis=1,inplace=True)

dataNueva.columns = dataNueva.columns.str.replace('_x', '')



print(dataNueva.isnull().sum())

print(dataNueva[dataNueva.isnull().any(axis=1)].shape)
anime=dataNueva.copy()

print(anime.groupby("type")["episodes"].describe())



anime.loc[(anime["type"]=="OVA") & (anime["episodes"].isna()),"episodes"] = anime.loc[(anime["type"]=="OVA") ,"episodes"].median()

anime.loc[(anime["type"]=="Movie") & (anime["episodes"].isna()),"episodes"] = anime.loc[(anime["type"]=="Movie") ,"episodes"].median()

anime.loc[(anime["type"]=="Music") & (anime["episodes"].isna()),"episodes"] = anime.loc[(anime["type"]=="Music") ,"episodes"].median()

anime.loc[(anime["type"]=="ONA") & (anime["episodes"].isna()),"episodes"] = anime.loc[(anime["type"]=="ONA") ,"episodes"].median()

anime.loc[(anime["type"]=="Special") & (anime["episodes"].isna()),"episodes"] = anime.loc[(anime["type"]=="Special") ,"episodes"].median()

anime.loc[(anime["type"]=="TV") & (anime["episodes"].isna()),"episodes"] = anime.loc[(anime["type"]=="TV") ,"episodes"].median()

anime.loc[(anime["type"].isna()) & (anime["episodes"].isna()),"episodes"] = anime["episodes"].median()



print(anime[anime.isnull().any(axis=1)].shape)

print(anime.isnull().sum())
anime["type"].replace(np.nan, "notype", inplace=True)

print(anime.isnull().sum())
anime["genre"].replace(np.nan, "nogenre", inplace=True)

print(anime.isnull().sum())
def impute_median(series):

    return series.fillna(series.median())



anime.rating = anime.groupby(['type', 'episodes'])[["rating"]].transform(impute_median)

anime.rating = anime.groupby(['genre', 'episodes'])[["rating"]].transform(impute_median)

anime["rating"]=anime["rating"].fillna(anime["rating"].median())

print(anime.isnull().sum())

anime=anime.reset_index()
anime_data = pd.concat([anime["genre"].str.get_dummies(sep=","),

                           anime["type"].str.get_dummies(sep=","),anime[["rating"]],

                            anime[["members"]],anime["episodes"]],axis=1)



anime_data.head()

anime_data = MaxAbsScaler().fit_transform(anime_data)

anime_data
KNNanime = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(anime_data)

distances, indices = KNNanime.kneighbors(anime_data)
def nombres_indices(name):  # Toma el nombre del anime y devuelve su indice correspondiente

    return anime[anime["name"]==name].index.tolist()[0] 

def recomendados_por_anime(nombre):  # Muestra el grupo de animes más cercanos al consultado

     found_id = nombres_indices(nombre)

     for id in indices[found_id][1:]:

            print(anime.loc[id]["name"])

            

recomendados_por_anime("Naruto")

        

       
print(rating.shape)

print(rating.isnull().sum())

rating.head()
merge = pd.merge(anime, rating, on="anime_id", how="left")

merge.head()
def similar_animes(id_anime):  # Trae todos los id_anime relacionados con un id_anime dado

    

    id_list=[]

    found_id = anime[anime["anime_id"]==id_anime].index.tolist()[0]  # Indice del id ingresado

    for id in indices[found_id][1:]:

            id_list.append(anime.loc[id]["anime_id"])

            

    return id_list  

        

            

def similar_animes_usuarios(id_user):  # Crea una lista con todos los animes relacionados con los animes visto por el usuario

    

    a = merge[merge["user_id"]==id_user].anime_id.values

    lista = []

    for i in range(len(a)):

        lista.append(similar_animes(a[i]))

    return lista

            

        

def similar_animes_usuarios_freq(id_user): # Crea una lista con los 6 anime más recomendados del usuario

    a=similar_animes_usuarios(id_user)

    r= np.array([])

    for i in range(5):

        f1 = pd.Series( (v[i] for v in a))

        r = np.append(r,f1)

        

    gh = merge[merge["user_id"]==id_user].anime_id.values

    rdiff=np.setdiff1d(r, gh)

    kk = pd.DataFrame({'Column1':rdiff})

    pda = pd.crosstab(index=kk["Column1"].astype(int), columns= "count")

    pda2 = pda.sort_values("count", ascending=False).head(6).index.tolist() 

    

    return pda2

        

    

def recomendados_usuario(id_user):  # Pasa de anime_id a los nombres de los animé

    

    a=similar_animes_usuarios_freq(id_user)

    for id in a:

        print(anime[anime["anime_id"]==id]["name"].values)

        

recomendados_usuario(3454)
recomendados_usuario(8765)
recomendados_por_anime("Dragon Ball Z")
recomendados_por_anime("Pokemon")