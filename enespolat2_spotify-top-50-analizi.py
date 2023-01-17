import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.offline import init_notebook_mode, iplot, plot
import seaborn as sns
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/top50spotify2019/top50.csv',encoding='ISO-8859-1')
data.rename(
            {'Unnamed: 0' : 'Sıralama',
            'Track.Name' : 'MüzikAdı',
            'Artist.Name' : 'ŞarkıcıAdı',
            'Genre' : 'Tür',
            'Beats.Per.Minute' : 'TempoSüresi',
            'Energy' : 'Enerji',
            'Danceability' : 'DansSeviyesi',
             'Loudness..dB..': 'ŞarkıDeğeri',
            'Liveness' : 'Canlılık',
            'Valence.' : 'Olumluluk',
            'Length.': 'Süre',
            'Acousticness..' : 'Akustik',
            'Speechiness.' : 'SözMiktarı',
            'Popularity' : 'Popülerlik'},axis=1,inplace=True)
data.head()
data.shape
data.columns
data.info()
print(data.describe())
data.head()
Şarkıcılar = data.ŞarkıcıAdı.unique()
data.Popülerlik = data.Popülerlik.astype(float)
popülerlik_oran_toplam=[]

for i in Şarkıcılar:
    x=data[data['ŞarkıcıAdı']==i]
    popülerlik_oran_bireysel = sum(x.Popülerlik)/len(x)
    popülerlik_oran_toplam.append(popülerlik_oran_bireysel)


popülerlik_data = pd.DataFrame({'Şarkıcı Adı' : Şarkıcılar, 'Popülerlik' : popülerlik_oran_toplam})
popülerlik_index=popülerlik_data['Popülerlik'].sort_values(ascending =False).index.values
popülerlik_sorted_data = popülerlik_data.reindex(popülerlik_index).head(15)

plt.Figure(figsize=(80,45))
sns.barplot(y=popülerlik_sorted_data['Popülerlik'] ,x=popülerlik_sorted_data['Şarkıcı Adı'])
plt.xticks(rotation=90)
plt.show()

x= data.ŞarkıcıAdı
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(x))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
tür_sayı = data.Tür.value_counts()
türler = data.Tür.unique()
tür_data = [türler,tür_sayı]


labels = data.Tür.value_counts().index
colors = ['grey','blue','red','yellow','green','brown']
explode = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
sizes =  data.Tür.value_counts().values

# visual
plt.figure(figsize = (10,10))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Tür Dağılımları',color = 'blue',fontsize = 15)
plt.show()
tür_ort = data.groupby('Tür').mean()
tür_ort
def plot_tür_analiz(feat):    
    plt.figure(figsize=(8,6))
    sns.barplot(data=data,y=tür_ort.sort_values(by=feat,ascending=False).index,x=tür_ort.sort_values(by=feat,ascending=False)[feat])

for i in data.select_dtypes(exclude='O').columns:
    plot_tür_analiz(i)
Tür_Data =[]
Türler = data.Tür.unique()
tempo=[]
dans=[]
canlılık=[]
popülerlik=[]

for i in Türler:
    x= data[data['Tür'] == i]
    Tür_data_oran_tempo = (sum(x.TempoSüresi)/len(x))
    Tür_data_oran_dans = (sum(x.DansSeviyesi)/len(x))
    Tür_data_oran_canlılık = (sum(x.Canlılık)/len(x))
    Tür_data_oran_popülerlik = (sum(x.Popülerlik)/len(x))
    tempo.append(Tür_data_oran_tempo)
    dans.append(Tür_data_oran_dans)
    canlılık.append(Tür_data_oran_canlılık)
    popülerlik.append(Tür_data_oran_popülerlik)

Tür_data=pd.DataFrame({'Türler' : Türler , 'Tempo Oranları' : tempo ,'Dans Edebilme Oranları' : dans ,'Canlılık Oranları' : canlılık,
                       'Popülerlik Oranları' : popülerlik})

trace1 = go.Scatter(
                    x=Tür_data['Türler'],
                    y=Tür_data['Tempo Oranları'],
                    name = 'Tür - Tempo İlişkisi',
                    )
trace2 = go.Scatter(
                    x=Tür_data['Türler'],
                    y=Tür_data['Dans Edebilme Oranları'],
                    name = 'Tür - Dans Edebilme İlişkisi',
                    )
trace3 = go.Scatter(
                    x=Tür_data['Türler'],
                    y=Tür_data['Canlılık Oranları'],
                    name = 'Tür - Canlılık İlişkisi',
                    )
trace4 = go.Scatter(
                    x=Tür_data['Türler'],
                    y=Tür_data['Popülerlik Oranları'],
                    name = 'Tür - Popülerlik İlişkisi',
                    )
İlişki_data = [trace1,trace2,trace3,trace4]


layout = go. Layout(
    dict(title = 'Şarkı Türleri İle Bazı Özelliklerin Değişimi',
              xaxis= dict(title= 'Türler',ticklen= 5,zeroline= False)))


fig =go.Figure(data=İlişki_data , layout=layout)
iplot(fig)

dans_oranları=[]
canlılık_oranları=[]
for i in türler:
    x=data[data['Tür']==i]
    dans_oranları_bir = (sum(x.DansSeviyesi)/len(x))/max(x.DansSeviyesi)
    canlılık_oranları_bir = (sum(x.Canlılık)/len(x))/max(x.Canlılık)
    canlılık_oranları.append(canlılık_oranları_bir)
    dans_oranları.append(dans_oranları_bir)
enerji_canlılık_data1= pd.DataFrame({'türler' : türler,'dans_oranları' : dans_oranları , 'canlılık_oranları' : canlılık_oranları})

trace1 = go.Scatter(
                    x=enerji_canlılık_data1.türler,
                    y=enerji_canlılık_data1.dans_oranları,
                    mode= 'lines',
                    name = 'Dans Oranları',
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text=enerji_canlılık_data1.dans_oranları)

trace2 = go.Scatter(
                    x=enerji_canlılık_data1.türler,
                    y=enerji_canlılık_data1.canlılık_oranları,
                    mode= 'lines+markers',
                    name = 'Canlılık Oranları',
                    marker = dict(color = 'rgba(250,12, 2, 0.6)'),
                    text=enerji_canlılık_data1.canlılık_oranları)

data1 =[trace1 , trace2]

layout = dict(title = 'Türlere Göre Dans Ve Canlılık Oranları',
              xaxis= dict(title= 'Türler',ticklen= 5,zeroline= False))


fig=dict(data=data1 , layout=layout)
iplot(fig)


f,ax=plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(),annot=True,linewidths = 0.5,linecolor = 'white' ,fmt= '.3f',ax=ax )
plt.show()
plt.Figure(figsize=(15,15))
sns.jointplot(data.Olumluluk,data.Enerji,color='cyan', kind='kde')
plt.show()
Enerji_seviye  = data.Enerji
enerji_color = [float(each) for each in data.Enerji]
data1 = [
    {
        'y': data.Enerji,
        'x': data.Sıralama,
        'mode': 'markers',
        'marker': {
            'color': enerji_color,
            'size': data.Sıralama,
            'showscale': True
        },
        "text" :  data.ŞarkıcıAdı
        
    }
]
iplot(data1)