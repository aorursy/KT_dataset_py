import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', 400)
pd.set_option('display.min_rows', 100)
df = pd.read_csv('../input/imoveis-fortaleza/imoveis_junho.csv')
df.shape
df.head()
df = df[(df['vagas'].str.len() <=3 ) & (df['area_m2'].str.len() <= 3) & (df['bedrooms'].str.len() <= 3) & (df['price'] != 'Sob consulta')]
print(df.shape)
df.isnull().sum()
print(df.shape)

df = df.dropna(subset=['bathrooms'])
print(df.shape)
df.dtypes
df[['price','area_m2','bedrooms','bathrooms','vagas']] = df[['price','area_m2','bedrooms','bathrooms','vagas']].astype(int)
df.describe()
df = df[(df.bedrooms <= 7) & (df.bathrooms <= 7) & (df.vagas <= 7)]
print(df.shape)
df['price/m2'] = df['price'] / df['area_m2']
df['price/m2'] = df['price/m2'].round(2)
df.sort_values(by='price/m2', ascending=False).tail(10)
print(df.shape)
#df = df[(df['price/m2'] > 1000)]
print('Quantidade de dados duplicados: ', df.duplicated().sum())
df.drop_duplicates(inplace=True)
print('Quantidade de dados duplicados após remoção: ', df.duplicated().sum())
print(df.shape)
df['Bairro'] = df['endereco'].apply(lambda x: x.split(',')[1] if any(s in x.split()[0] for s in ("Rua","Avenida","Travessa","Alameda")) else x.split(',')[0])    
df2 = df.drop(['description','endereco'], axis=1)
df2.head()
print(df2.shape)
df2['Bairro'] = df2['Bairro'].apply(lambda x: x.strip())
#df2['Bairro'] = df2['Bairro'].str.title()
df2 = df2[df2['Bairro'] != 'Fortaleza']
from unidecode import unidecode
df2["Bairro"]= df2["Bairro"].apply(lambda x: unidecode(x))
df2["Bairro"] = df2["Bairro"].str.upper()
df2["Bairro"] = df2["Bairro"].map(lambda x: x.replace("PREFEITO JOSE WALTER", "PREFEITO JOSE VALTER").replace("MANOEL DIAS BRANCO", "MANUEL DIAS BRANCO").replace("ENGENHEIRO LUCIANO CAVALCANTE","ENG LUCIANO CAVALCANTE").replace("SAPIRANGA","SAPIRANGA COITE").replace("GUARARAPES", "PATRIOLINO RIBEIRO"))
print(df2.shape)
a = len(df2['Bairro'].value_counts())
print(f'Numero de bairros presentes no dataset: {a}.') 
b = 5
bairros6 = df2['Bairro'].value_counts().loc[lambda x : x>=b]
print(f'Numero de bairros com mais de {b} anuncios: {len(bairros6)}')
import plotly.offline as py
import plotly.graph_objs as go


x=bairros6.index
y=bairros6.values

#print(f'Numero de bairros com mais de {b} anuncios: {len(x)}')
data = [go.Bar(x=x, y=y
            )]

layout = go.Layout(title = 'Quantidade de Aptos por Bairro',
                   #xaxis = {'title': 'Bairro'},
                   yaxis = {'title': 'Quantidade'},
                   height =500,
                   xaxis_tickangle=45
                   )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')
c = len(x)
df_47 = df2[(df2['Bairro'].isin(list(bairros6.index)))]
print(df2.shape)
print(df_47.shape)
print(f'Numero de aptos cortados: {len(df2) - len(df_47)}')
lista = bairros6.index.tolist()

trace1 = go.Box(
    y=df_47.price,
    x=df_47.Bairro,
    #name='kale',
    marker=dict(
        color='blue'
    )
)
data = [trace1]
layout = go.Layout(
       #title = 'Quantidade de Aptos por Bairro',
       yaxis = {'title': 'Preço'},
       barmode='stack',
       height = 700,
       xaxis=dict(
       categoryorder='array',
       categoryarray=lista,
       titlefont=dict(
         size=6,
         color='black'),
       showticklabels=True,
       tickfont=dict(
        size=12,
        color='black',
        ),
    tickangle=45,
   
    ),
)   

fig = go.Figure(data=data, layout=layout)
fig.show()
df_47 = df_47[(df_47['price'] < 1000000)]
df_47.shape
lista = bairros6.index.tolist()

trace1 = go.Box(
    y=df_47.price,
    x=df_47.Bairro,
    #name='kale',
    marker=dict(
        color='blue'
    )
)
data = [trace1]
layout = go.Layout(
       yaxis = {'title': 'Preço'},
       barmode='stack',
       height = 700,
       xaxis=dict(
       categoryorder='array',
       categoryarray=lista,
       titlefont=dict(
         size=6,
         color='black'),
       showticklabels=True,
       tickfont=dict(
        size=12,
        color='black',
        ),
    tickangle=45,
   
    ),
)   
fig = go.Figure(data=data, layout=layout)
fig.show()
trace1 = go.Histogram(x=df_47['price'], nbinsx=20, name= 'Hist. Original')

fig = go.Figure()

fig.add_trace(trace1)

fig.update_layout(
    title_text='Distribuiçao de preços', # title of plot
    xaxis_title_text='Price', # xaxis label
    yaxis_title_text='Quantidade Aptos', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
)
fig.show()
print("Skewness: %f" % df_47['price'].skew())
print("Kurtosis: %f" % df_47['price'].kurt())
import plotly.offline as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Plotando Boxplot
y1 = df_47.bedrooms
y2 = df_47.bathrooms
y3 = df_47.vagas
y4 = df_47.area_m2

fig = make_subplots(rows=2, cols=2,) #subplot_titles=("Scores", "Magnitudes"))

fig.add_trace(go.Histogram(y=y1, name='Bedrooms'), row=1, col=1)
fig.add_trace(go.Histogram(y=y2, name='Bathrooms'), row=1, col=2)
fig.add_trace(go.Histogram(y=y3, name='Vagas'), row=2, col=1)
fig.add_trace(go.Histogram(y=y4, name='Area_m2', nbinsy=20), row=2, col=2)

fig.update_layout(height=700,  title_text="Histogramas", title_x =0.05)
fig.update_layout(
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
)
fig.show()
bairros = df_47.groupby("Bairro").mean()
bairros['Number Aptos'] = bairros.index.map(bairros6)
bairros.dropna(subset=['Number Aptos'], inplace=True)
bairros['NOME']=bairros.index
bairros = bairros.round(2)
bairros = bairros.sort_values(by='price/m2', ascending=False)
bairros.head()
t = bairros['Number Aptos']
y = bairros['price/m2']
x = bairros.index

trace1 = go.Bar(y = y,
               x = x,
               marker={'color': y,'colorscale': 'Portland', 'reversescale': False}, 
               text= t, 
               textposition='auto',
               name='media')
              
data = [trace1]
layout = go.Layout(title = 'Media Preço/ M2 Bairros de Fortaleza',
                   xaxis = {'title': 'Bairro'},
                   yaxis = {'title': 'Price'},
                   height = 700,
                   uniformtext_minsize=9,
                   uniformtext_mode='hide',
                   xaxis_tickangle=45                 
                   )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
import seaborn as sns
import matplotlib.pyplot  as plt

# Gerar dataframe de correlações
z = df_47.corr()

# Plotar heatmap
plt.figure(figsize=(18,7))
g = sns.heatmap(z, cmap='coolwarm', linewidths=0.5, annot=True,annot_kws={"size": 15})
g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment='right', fontsize='x-large')
g.set_xticklabels(g.get_yticklabels(), rotation=45, horizontalalignment='right', fontsize='x-large')
import json
import geopandas as gpd

data = json.load(open('../input/bairros-fortaleza-coord/FortalezaBairros.geojson'))

mapa_bairros = gpd.GeoDataFrame.from_features(data,crs="epsg:4326")
mapa_bairros= mapa_bairros[['GID', 'NOME','geometry']]
unidos = pd.merge(mapa_bairros, bairros, on='NOME', how='inner')
unidos = unidos[unidos['Number Aptos'] >=3]
unidos= unidos[['GID', 'NOME','price','area_m2', 'bedrooms', 'bathrooms', 'vagas', 'price/m2', 'Number Aptos','geometry']]
unidos.sort_values(by='price/m2', ascending=False).head(3)
mapa_bairros2 = pd.merge(mapa_bairros, bairros, on='NOME', how='outer')
dropar = set(bairros.NOME).symmetric_difference(unidos.NOME)
print(f'Bairros fora da lista da prefeitura: {len(dropar)} -> {dropar}')
mapa_bairros2 = mapa_bairros2[(mapa_bairros2['NOME'].isin(dropar) == False)]
mapa_bairros2.fillna({'price/m2': 'No Info'}, inplace=True)
mapa_bairros2.fillna({'Number Aptos': 'No Info'}, inplace=True)
import folium
centro = [-3.7657875,-38.5078477]

# creating the map object
basemap = folium.Map(
    location=centro,
    tiles= 'cartodbpositron',
    zoom_start=12.,
    min_zoom=10,
    max_zoom=14,
    width='90%', height='90%',
    
)
# plotting the choropleth
legends = 'Casos confirmados'
x = folium.Choropleth(
    geo_data=mapa_bairros2,
    data=unidos,
    #name='teste',
    columns=['NOME','price/m2'],
    key_on='feature.properties.NOME',
    fill_color= 'YlOrRd',
    fill_opacity=0.4,
    line_opacity=0.3,
    nan_fill_color = 'gray',
    nan_fill_opacity = 0.1,
    legend_name='Price/m2',
    highlight=True,
    #tooltip = tooltip
).add_to(basemap)

x.geojson.add_child(
    folium.features.GeoJsonTooltip(
    fields=['NOME','price/m2','Number Aptos'],
    aliases=['Bairro:','Price/m2:', 'Num Aptos à Venda:'],          
 )
)

basemap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score
df3 = df_47.join(pd.get_dummies(df_47.Bairro))
df3 = df3.drop(['Bairro','price/m2'], axis=1)
df3.head()
X_train1, X_test1, y_train1, y_test1 = train_test_split(df3.drop('price', axis=1), df3.price, random_state=60)
model = RandomForestRegressor(n_estimators=100, random_state=22)
model.fit(X_train1, y_train1)
y_pred = model.predict(X_test1)
r2 = r2_score(y_test1,y_pred).round(4)
print(f'r2_score : {r2}')
print(model.score(X_test1, y_test1))
mean_absolute_error(y_test1,y_pred).round(2)

