%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import random
import os
import shutil
import folium
from folium.plugins import MarkerCluster
from folium.plugins import HeatMap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import seaborn as sns

# Disabilitarv avisos
import warnings
warnings.filterwarnings('ignore')
#Sahpe das UF brasileiras
BRASIL=gpd.read_file('../input/olistaggregate/BRUFE250GC_SIR.shp')

populacao=[881935,3337357,845731,14873064,9132078,3015268,4018650,7018354,7075181,3484466,2778986,21168791,8602865,4018127,
9557071,3273227,17264943,3506853,11377239,605761,7164788,45919049,2298696,1572866,11433957,1777225,4144597]
BRASIL['pop']=populacao
BRASIL['area']=BRASIL.area
BRASIL.head(27)
#Tabela na forma original
df=pd.read_excel('../input/olistaggregate/DADOS AGREADOS.xlsx')

#Separar colunas relevantes
df=df[['product_category_name', 'price', 'review_comment_message',
       'review_score', 'avaliation', 'customer_region',
       'delivery_time_days','delay', 'geolocation_lat', 'geolocation_lng']]

#Traduzir pra portugês 

dic={'product_category_name':'Produto', 
     'price':'Preço', 
     'review_comment_message':'Comentário',
     'review_score':'Nota',
     'customer_region':'Região',
     'avaliation':'Avaliação',
     'customer_state':'UF',
     'customer_city':'Cidade', 
     'geolocation_lat' :'latitude', 
     'geolocation_lng':'longitude',
     'points_distance_km':'Distância de Entrega', 
     'delivery_time_days':'Tempo de Entrega', 
     'reaction_time_days':'Tempo de Comentário',
     'answer_time_days':'Tempo de Resposta da  Loja',
     'delay':'Atraso na Entrega'}
#Renomear colunas

df.rename(columns=dic,inplace=True)
df.sort_values('Nota',inplace=True)
df.reset_index(drop=True,inplace=True)

# Tranformar em geo data frame
x=zip(df.longitude,df.latitude)
geometry=[Point(x) for x in zip(df.longitude,df.latitude) ]

#crs defualt
crs = {'proj': 'latlong', 'ellps': 'GRS80', 'datum': 'WGS84', 'no_defs': True}
geo_df=gpd.GeoDataFrame(df,crs=crs,geometry=geometry)
geo_df.head()
#Igular crs 
BRASIL=BRASIL.to_crs('+proj=utm +zone=23 +south +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=km +no_defs')
geo_df= geo_df.to_crs('+proj=utm +zone=23 +south +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=km +no_defs')


# Separar por avaliação 
geo_negativa=geo_df[geo_df['Avaliação']=='negativa']
geo_regular =geo_df[geo_df['Avaliação']=='regular']
geo_positiva =geo_df[geo_df['Avaliação']=='positiva']
paleta= 'cividis'
#'inferno' #'gist_heat'#'cividis'#'afmhot'#'Spectral' #'RdGy'  #'RdBu' 

# #Fechar plot anterior 
plt.clf()
plt.cla()
plt.close()


#BASE=BRASIL.plot(color='k', edgecolor='w',alpha=0.85,figsize=(20,15),legend=True)

BASE=BRASIL.plot(column='pop',edgecolor='w',alpha=0.85,legend=False,cmap=paleta,figsize=(20,15))

MAPA=geo_negativa.plot(ax=BASE,marker='X',color="#9d9d9d", edgecolor='k',alpha=1,markersize=75,legend=True)
MAPA=geo_regular.plot(ax=BASE,marker='^',color="#ec008b", edgecolor='k',alpha=1,markersize=75,legend=True)
MAPA=geo_positiva.plot(ax=BASE,marker='o',color="#00BFFF", edgecolor='k',alpha=1,markersize=75,legend=True)

#Legenda

legenda=plt.legend(title='Avaliação', loc='upper right', 
                   labels=['Negativa - 29.95 %',
                           'Regular  -  9.12 %',
                           'Positiva - 60.94 %']
                   ,prop={'family': 'Arial','size':14},frameon=False)
plt.setp(legenda.get_title(),fontsize=14,family='Arial')

BASE.set_axis_off()


#Color bar
mn = 605761
mx = 45919049
norm = plt.Normalize(vmin=mn, vmax=mx)
n_cmap = cm.ScalarMappable(norm=norm, cmap=paleta)
n_cmap.set_array([])
BASE.get_figure().colorbar(n_cmap, ax=BASE, orientation='vertical')


MAPA=MAPA
BASE=BASE
# Encurtar titulos
geo_df.rename(columns={'Comentário':'text','Avaliação':'avaliation','Produto':'prod'},inplace=True)
geo_negativa=geo_df[geo_df['avaliation']=='negativa']
geo_regular =geo_df[geo_df['avaliation']=='regular']
geo_positiva =geo_df[geo_df['avaliation']=='positiva']

# Definir crs padrão
crs = {'init': 'epsg:4326'}
BRASIL.to_crs(crs,inplace=True)

#Cacular centroide
y=BRASIL.centroid.y.iloc[5] #Brasilia
x=BRASIL.centroid.x.iloc[5] #Brasilia



# Trabalhar com sample do data set 
geo_sample=geo_df.sample(5001,random_state=42)

geo_sample_neg=geo_negativa.sample(5001,random_state=42)
geo_sample_reg=geo_regular.sample(5001,random_state=42)
geo_sample_pos=geo_positiva.sample(5001,random_state=42)

#Montar a base
base = folium.Map([y, x], zoom_start=4, tiles='OpenStreetMap')
base.choropleth(BRASIL,
               name="BRASIL",
               line_color="Black",
               line_weight=0.5,
               fill_opacity=0)

#Clueterizar 
cluster_1 = MarkerCluster()
cluster_2 = MarkerCluster()
cluster_3 = MarkerCluster()

#Por marcadores
for item in geo_sample_neg.itertuples():
        
    cluster_1.add_child(folium.Marker(location=[item.latitude, item.longitude],
                                popup="<h5>"+str(item.text)+"</li> </h5>" + 
                                      "<li> Produto: " +str(item.prod)+
                                      "<li> Avalição: " +str(item.avaliation),
                                       icon=folium.Icon(color='black',prefix='fa',icon='fa fa-shopping-basket')))   
    
for item in geo_sample_reg.itertuples():
    
    cluster_2.add_child(folium.Marker(location=[item.latitude, item.longitude],
                                popup="<h5>"+str(item.text)+"</li> </h5>" + 
                                      "<li> Produto: " +str(item.prod)+
                                      "<li> Avalição: " +str(item.avaliation),
                                       icon=folium.Icon(color='pink',prefix='fa',icon='fa fa-shopping-basket')))   
    
for item in geo_sample_pos.itertuples():
    
    cluster_3.add_child(folium.Marker(location=[item.latitude, item.longitude],
                                popup="<h5>"+str(item.text)+"</li> </h5>" + 
                                      "<li> Produto: " +str(item.prod)+
                                      "<li> Avalição: " +str(item.avaliation),
                                       icon=folium.Icon(color='blue',prefix='fa',icon='fa fa-shopping-basket')))   

    
base.add_child(cluster_1)
base.add_child(cluster_2)
base.add_child(cluster_3)




#Controle de camadas
folium.LayerControl().add_to(base)

#Salvar
# nome_arquivo='SAMPLE.html'
# base.save(nome_arquivo)

base
