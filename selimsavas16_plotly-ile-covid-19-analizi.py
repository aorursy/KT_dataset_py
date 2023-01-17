import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go


from wordcloud import WordCloud

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
covid_19 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
covid_19.info()
covid_19.head()
covid_19.corr() # 0.881778 -> Confirmed ve Deaths arası ilişki
covid_19.describe()
covid_19.describe().T
covid_19_columns_v2 = ['sNo','ObservationDate','Province_State','Country_Region','Last Update',"Confirmed","Deaths","Recovered"]
covid_19.columns = covid_19_columns_v2
covid_19.head()
covid_19_plotly = pd.DataFrame({"Province_State":covid_19.Province_State,
                                "Confirmed": covid_19.Confirmed,
                                "Deaths": covid_19.Deaths
    })                                              
covid_19_plotly.head()
len(covid_19["Province_State"].unique())
plotly_data = covid_19_plotly.groupby("Province_State").sum() 
plotly_data = plotly_data.reset_index() 
plotly_data["Confirmed_Deaths_rate"] = plotly_data["Deaths"] / plotly_data["Confirmed"]
plotly_data[0:10]


# asceding : azalan sıralama
new_index = (plotly_data["Confirmed_Deaths_rate"].sort_values(ascending=False)).index.values

# indexleri değiştir
sorted_data = plotly_data.reindex(new_index)
# sorted_data
# go.Scatter ile tracelerimizin oluşturulması

trace1 = go.Scatter(
    x=sorted_data[:40].Province_State,
    y=sorted_data[:40].Confirmed_Deaths_rate,
    # Domain oluşturmamıza gerek yok ilki için
    name = "Confirmed_Deaths_rate"
)
trace2 = go.Scatter(
    y=sorted_data[:40].Confirmed,
    xaxis='x2', # x domain, farklı bir alanda grafik oluşucağı için
    yaxis='y2', # y domain
    name = "Confirmed"
)
trace3 = go.Scatter(
    y=sorted_data[:40].Deaths,
    xaxis='x3',
    yaxis='y3',
    name = "Deaths"
)
data = [trace1, trace2, trace3] # tracelerimiz data listetesinde birleşti

# Görselleştirme stringleri

layout = go.Layout(
    xaxis=dict(
        domain=[0, 1] # Hangisi nereye gelicek konum
    ),
    yaxis=dict(
        domain=[0, 1]
    ),
    xaxis2=dict(
        domain=[0, 1]
    ),
    xaxis3=dict(
        domain=[0, 1],
        
    ),
    yaxis1=dict(
        domain=[0, 0.33],
        
    ),
    yaxis2=dict(
        domain=[0.33, 0.66],
        
    ),
    yaxis3=dict(
        domain=[0.66, 1]
    ),
    title = 'Plotly ile COVID-19 Analizi'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# asceding : azalan sıralama
new_index = (plotly_data["Confirmed"].sort_values(ascending=False)).index.values

# indexleri değiştir
sorted_data2 = plotly_data.reindex(new_index)

sorted_data2.head()
sorted_data2 =sorted_data2[:40]
sorted_data2.head()
# asceding : azalan sıralama
new_index = (sorted_data2["Confirmed_Deaths_rate"].sort_values(ascending=False)).index.values

# indexleri değiştir
sorted_data3 = sorted_data2.reindex(new_index)

sorted_data3.head()
# go.Scatter ile tracelerimizin oluşturulması

trace1 = go.Scatter(
    x=sorted_data3[:20].Province_State,
    y=sorted_data3[:20].Confirmed_Deaths_rate,
    # Domain oluşturmamıza gerek yok ilki için
    name = "Confirmed_Deaths_rate"
)
trace2 = go.Scatter(
    y=sorted_data3[:40].Confirmed,
    xaxis='x2', # x domain, farklı bir alanda grafik oluşucağı için
    yaxis='y2', # y domain
    name = "Confirmed"
)
trace3 = go.Scatter(
    y=sorted_data3[:40].Deaths,
    xaxis='x3',
    yaxis='y3',
    name = "Deaths"
)
data = [trace1, trace2, trace3] # tracelerimiz data listetesinde birleşti

# Görselleştirme stringleri

layout = go.Layout(
    xaxis=dict(
        domain=[0, 1] # Hangisi nereye gelicek konum
    ),
    yaxis=dict(
        domain=[0, 1]
    ),
    xaxis2=dict(
        domain=[0, 1]
    ),
    xaxis3=dict(
        domain=[0, 1],
        
    ),
    yaxis1=dict(
        domain=[0, 0.33],
        
    ),
    yaxis2=dict(
        domain=[0.33, 0.66],
        
    ),
    yaxis3=dict(
        domain=[0.66, 1]
    ),
    title = 'Plotly ile COVID-19 Analizi'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
sorted_data3.info() 
sorted_data3.Confirmed_Deaths_rate = sorted_data3.Confirmed_Deaths_rate.astype("float")
sorted_data3.Deaths = sorted_data3.Deaths.astype("float")

sorted_data3.info()

from wordcloud import WordCloud
plt.subplots(figsize=(8,8))

wordcloud = WordCloud(
    
    background_color = "white", # arkaplan rengi
    width = 512, #genişlik
    height = 384, #yükseklik
).generate(" ".join(sorted_data3[:40].Province_State))   # kullanılan kelimeleri ayır ve ona göre çok olanları daha büyük şekilde bastır

plt.imshow(wordcloud)
plt.axis("off") # x ve y eksenlerini kapa

import plotly.figure_factory as ff 
# Data

dataframe = sorted_data3[:30] # veriyi filitreliyoruz 
data = dataframe.loc[:,["Confirmed","Deaths", "Confirmed_Deaths_rate"]] # karşılaştırmak istediğimiz featuresler
data["index"] = np.arange(1,len(data)+1) # datafremin indexine eşitliyor ve ekliyoruz.
data.head()
# Scatter Matrix

fig = ff.create_scatterplotmatrix(data, # verimiz
                                  diag='box', #orta (-x) ekseni
                                  index='index', # index olarak yeni belirlediğim index
                                  colormap='Portland', # renk paleti
                                  
                                  text = sorted_data3.Province_State,
                                  colormap_type='cat', 
                                  height=700, width=700) # boyutlandırma
iplot(fig)
sorted_data.head()
# 3 Boyutun her birini trace 1 de gösterebiliriz.

trace1 = go.Scatter3d( # go kütüphanesinden scatter3d yi çağırdık
    
    x=sorted_data[:40].Confirmed_Deaths_rate, # x ekseni değerlerimiz
    y=sorted_data[:40].Confirmed, # y ekseni değerlerimiz
    z=sorted_data[:40].Deaths, # z ekseni değerlerimiz
    
    text =  sorted_data.Province_State,
    mode='markers', # noktalı yapıda olsun
    marker=dict( # sözlük yapısında boyu ve rengimizi ayarladık
        size=10,
        color='rgb(255,0,0)',                 
    )
)

data = [trace1] # tek bir tracemizi listeledik


layout = go.Layout( # go kütüphanesinden Layout methodu ile
    margin=dict( # kenarlardan bırakılan paylar 0
        l=0,
        r=0,
        b=0,
        t=0  
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
labels = sorted_data3.Province_State

# figure

fig = {
    
  "data": [
    { # Trace 
      "values": sorted_data3[:15].Confirmed_Deaths_rate, # değerler
      "labels": labels,    
      "domain": {"x": [0, .5]}, 
      "name": "Şehir/eyalet",
      "hoverinfo":"label+percent+name",
      "hole": .3, # orta yuvarlak çapı
      "type": "pie" #tip
    },], 
    
  
  # Görselleştirmenin düzeni
  "layout": {
        "title":"En çok hastası olan şehirlerin ölüm oranı",
        "annotations": [
            { "font": { "size": 20}, #boyutlar
              "showarrow": False,
              "text": "Ölüm Oranı", 
                "x": 0,
                "y": 1
            },
        ]
    }
}
iplot(fig)