datas = ['2020-02-25','2020-02-26','2020-02-27','2020-02-28','2020-02-29','2020-03-01','2020-03-02','2020-03-03','2020-03-04','2020-03-05','2020-03-06','2020-03-07','2020-03-08','2020-03-09','2020-03-10','2020-03-11','2020-03-12','2020-03-13','2020-03-14','2020-03-15','2020-03-16','2020-03-17','2020-03-18','2020-03-19','2020-03-20','2020-03-21','2020-03-22','2020-03-23','2020-03-24','2020-03-25','2020-03-26','2020-03-27','2020-03-28','2020-03-29','2020-03-30','2020-03-31','2020-04-01','2020-04-02','2020-04-03','2020-04-04','2020-04-05','2020-04-06','2020-04-07','2020-04-08','2020-04-09','2020-04-10','2020-04-11','2020-04-12','2020-04-13']
import pandas as pd
import numpy as np
import folium
import plotly.express as px
all_dataframes = {}

for data in datas:
    csv_filepath = '../input/casos-confirmados-e-mortes-'+data+'.csv'
    all_dataframes[data] = pd.read_csv(csv_filepath, sep=',')
#all_dataframes
def highlight_col(x):
    red = 'background-color: red'
    green = 'background-color: green'
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    df1.iloc[:, 4] = green
    df1.iloc[:, 5] = red
    return df1
def show_latest_cases(df, n):
    n = int(n)
    return df.sort_values('confirmed', ascending= False).head(n).style.apply(highlight_col, axis=None)
data_mais_recente = '2020-04-13'
casos_brasil_recentes_df = all_dataframes[data_mais_recente]
casos_brasil_recentes_ord_df = casos_brasil_recentes_df.sort_values('confirmed', ascending= False)

show_latest_cases(casos_brasil_recentes_ord_df, 10)
def bubble_chart(df, n, data):
    fig = px.scatter(df.head(n), x="nome", y="confirmed", size="confirmed", color="nome",
               hover_name="nome", size_max=60)
    fig.update_layout(
    title=str(n) +" Municipios mais atingidos em " + data,
    xaxis_title="Municipios",
    yaxis_title="Casos confirmados",
    width = 700
    )
    fig.show()
bubble_chart(casos_brasil_recentes_ord_df, 10, data_mais_recente)
px.bar(
    casos_brasil_recentes_ord_df.head(10),
    x = "nome",
    y = "deaths",
    title= "Os 10 municípios com mais mortes na data: " + data_mais_recente, # the axis names
    color_discrete_sequence=["pink"], 
    height=500,
    width=800
)

data_mapa = data_mais_recente
df_mapa = all_dataframes[data_mapa]

# creating world map using Map class
world_map = folium.Map(location=[-16.1237611, -59.9219642], tiles="cartodbpositron", zoom_start=4, max_zoom = 10, min_zoom = 2)
#title_html = '''<h3 align="center" style="font-size:20px"><b>COVID-19: Casos confirmados em: ''' + data_mapa + '''</b></h3>'''world_map.get_root().html.add_child(folium.Element(title_html))
#world_map.get_root().html.add_child(folium.Element(title_html))
# iterate over all the rows of confirmed_df to get the lat/long
for i in range(0,len(df_mapa)):
    folium.Circle(
        location=[df_mapa.iloc[i]['latitude'], df_mapa.iloc[i]['longitude']],
        fill=True,
        radius=(int((np.log(df_mapa.iloc[i,-1]+1.00001)))+0.2)*50000,
        color='red',
        fill_color='indigo',
    ).add_to(world_map)
    
world_map
# world_map.save('world_map.html')
from IPython.display import YouTubeVideo
YouTubeVideo('aMvVeQgTCV4', width=1020, height=800)