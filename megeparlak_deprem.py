# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import plotly.graph_objs as go
from   plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 
from wordcloud import WordCloud, STOPWORDS

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
deprem = pd.read_csv("../input/earthquake/earthquake.csv")
deprem.info()
deprem.head()
deprem.dropna(subset=["area"], inplace=True)
deprem.dropna(subset=["city"], inplace=True)
deprem.dropna(subset=["direction"], inplace=True)
deprem.info()
deprem_filtered = deprem[deprem["country"]=="turkey"]
deprem['area']= deprem['area'].astype(str)
counter = Counter(deprem_filtered.area +"\n"+ deprem_filtered.city)
most_eq = counter.most_common(10)
x,y= zip(*most_eq)
x,y= list(x), list(y)

plt.figure(figsize=(20,15))
ax= sns.barplot(x=x, y=y, palette = sns.cubehelix_palette(len(x)))
plt.xlabel("En çok depreme uğrayan 10 bölge (1910-2017)")
plt.ylabel("Deprem gerçekleşme sayısı")
plt.title("Depremsel Bölge Grafiği (En Çok)")
plt.figure(figsize=(20,10))
sns.countplot(deprem.direction)
all_text = " ".join(i for i in deprem.city)
plt.figure(figsize=(30,30))
wordcloud = WordCloud(background_color="white",height=1000,width=2000).generate(all_text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
deprem["year"] = [int(i.split(".")[0]) for i in deprem.iloc[:,1]]
years = [str(i) for i in list(deprem.year.unique())]
directions= ['west', 'south_west', 'south_east', 'south', 'north_west','north_east', 'east', 'north']
custom_color= {'west' : "rgb(0,0,0)",
               'south_west' : "rgb(131,0,31)" ,
               'south_east' : "rgb(255,242,0)",
               'south' : "rgb(255,0,60)" ,
               'north_west' :"rgb(0,239,255)",
               'north_east' : "rgb(179,0,255)",
               'east' : "rgb(255,0,0)" ,
               'north' : "rgb(26,0,255)"}

figure={"data":[], "layout":{}, "frames":[]}
figure["layout"]["geo"]=dict(showframe=False,showland=True,showcoastlines=True,showcountries=True,countrywidth=1,showlakes=True)
figure["layout"]["hovermode"]="closest"
figure["layout"]["sliders"]={"args":["transition",{"duration":400,"easing":"cubic-in-out"}],"initialValue":"1912","plotlycommand":"animate","values":years,"visible":True}
figure['layout']['updatemenus'] = [{'buttons': [{'args': [None, {'frame': {'duration': 500, 'redraw': False},'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],'label': 'Play','method': 'animate'},{'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate','transition': {'duration': 0}}],'label': 'Pause','method': 'animate'}],'direction': 'left','pad': {'r': 10, 't': 87},'showactive': False,'type': 'buttons','x': 0.1,'xanchor': 'right','y': 0,'yanchor': 'top'}]
sliders_dict = {'active': 0,'yanchor': 'top','xanchor': 'left','currentvalue': {'font': {'size': 20},'prefix': 'Year:','visible': True,'xanchor': 'right'},'transition': {'duration': 300, 'easing': 'cubic-in-out'},'pad': {'b': 10, 't': 50},'len': 0.9,'x': 0.1,'y': 0,'steps': []}
yil = 1912
for f in directions:
    by_year= deprem[deprem["year"]==yil]
    by_year_and_cont= by_year[by_year["direction"]==f]
    d1_dict = dict(type='scattergeo',lon = deprem['long'],lat = deprem['lat'],hoverinfo = 'text',text = f,mode = 'markers',marker=dict(sizemode = 'area',sizeref = 1,size= 10 ,line = dict(width=1,color = "white"),opacity = 0.7))
    figure["data"].append(d1_dict)
    
for i in years:
    frame = {'data': [], 'name': str(i)}
    for ty in directions:
        dataset_by_year = deprem[deprem['year'] == int(i)]
        dataset_by_year_and_cont = dataset_by_year[dataset_by_year['direction'] == f]
        d2_dict = dict(type='scattergeo',lon = by_year_and_cont['long'],lat = by_year_and_cont['lat'],hoverinfo = 'text',text = f,mode = 'markers', marker=dict(sizemode = 'area',sizeref = 1,size= 10 ,line = dict(width=1,color = "white"),opacity = 0.7),name = f)
        frame['data'].append(d2_dict)
        
figure['frames'].append(frame)
slider_step = {'args': [[yil],{'frame': {'duration': 300, 'redraw': False},'mode': 'immediate','transition': {'duration': 300}}],'label': yil,'method': 'animate'}
sliders_dict['steps'].append(slider_step)


figure["layout"]["autosize"]= True
figure['layout']['sliders'] = [sliders_dict]
fig = go.Figure(figure)
fig.show()