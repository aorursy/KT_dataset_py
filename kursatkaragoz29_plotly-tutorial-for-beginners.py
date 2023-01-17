



#import chart_studio.plotly as py



#matplotlib

import matplotlib.pyplot as plt



#word cloud library

from wordcloud import WordCloud



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

# import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# cwurData = pd.read_csv("../input/world-university-rankings/cwurData.csv")

# education_expenditure_supplementary_data = pd.read_csv("../input/world-university-rankings/education_expenditure_supplementary_data.csv")

# educational_attainment_supplementary_data = pd.read_csv("../input/world-university-rankings/educational_attainment_supplementary_data.csv")

# school_and_country_table = pd.read_csv("../input/world-university-rankings/school_and_country_table.csv")

# shanghaiData = pd.read_csv("../input/world-university-rankings/shanghaiData.csv")

timesData = pd.read_csv("../input/world-university-rankings/timesData.csv")
timesData.info()
timesData.head()

# Data incelendiğinde: None değerler ve "-" değerler oldğu görülmektedir.
# import graph objects as "go"

import plotly.graph_objs as go



# Problem : 2011(ilk 100 üniversite) yılındaki üniversitelerin 

# alıntı yapma(citation) ve öğretim (teaching) skorlarını

#------------------------- Kaba kod.-----------------------------------

# ilk 100 sample'ı tüm featureler ile birlikte elde et.

# Trace 1 oluştur, Alıntı yapma featuresini ait olduğu üniversite ve puan ile birlikte işle.

# Trace 2 oluştur  Öğretim Featuresini ait olduğu üniversite ve puan ile birlikte işle

# trace1 ve trace2'i birleştir

# layout oluştur

# figüre oluştur, figüre traceleri ve layout'u göm.

# figüre iplot ile çiz.



# prepare data frame

df = timesData.iloc[:100,:]  #timesData nın ilk 100 sample'si (bütün featureleri ile birlikte)



# Creating trace1

# Trace1 oluşturuldu, x ekseninde dünya puanı, y eksenindede alıntı yapma feature'sine ait değerler,

# moduna lines, isim olarak citations, etiket olarakta ilk 100 sampledaki university_name featuresi verildi.



trace1 = go.Scatter(

                    x = df.world_rank,

                    y = df.citations,

                    mode = "lines",   #plotun şeklini belirtir (markers,lines or lines + markers)

                    name = "citations",   #

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),  #RGB + opacity(alpha)

                    text= df.university_name)

# Creating trace2

trace2 = go.Scatter(

                    x = df.world_rank,

                    y = df.teaching,

                    mode = "lines+markers",

                    name = "teaching",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'), #blue

                    text= df.university_name)

data = [trace1, trace2]

layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',   #plot dışında kalan herşey, y ekseni,başlaık bölgesi...

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout) #figure oluşturuldu

iplot(fig)
# prepare data frames

# 2014,2015,2016 yıllarına ait ilk 100 üniversite kayıtları dataframe'lere aktarıldı.

df2014 = timesData[timesData.year == 2014].iloc[:100,:] #2014'ün ilk 100 sample'si

df2015 = timesData[timesData.year == 2015].iloc[:100,:] #2015'in ilk 100 sample'si

df2016 = timesData[timesData.year == 2016].iloc[:100,:] #2016'nın ilk 100 sample'si

# import graph objects as "go"

#import plotly.graph_objs as go   #bir önceki örnekte zaten import ettik.



# ilk önce graph_objs kütüphanesinden yararlanarak scatterlarımızın yapısını oluşturuyoruz:

# creating trace1

trace1 =go.Scatter(

                    x = df2014.world_rank,   #üniversitelerin sıralaması

                    y = df2014.citations,    #üniversitelerin alıntıları

                    mode = "markers",        #plot tipi

                    name = "2014",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),   # (RGB,Opacity)

                    text= df2014.university_name)

# creating trace2

trace2 =go.Scatter(

                    x = df2015.world_rank,

                    y = df2015.citations,

                    mode = "markers",

                    name = "2015",

                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),

                    text= df2015.university_name)

# creating trace3

trace3 =go.Scatter(

                    x = df2016.world_rank,

                    y = df2016.citations,

                    mode = "markers",

                    name = "2016",

                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),

                    text= df2016.university_name)

# traceleri, data isimli list'de topladık.

data = [trace1, trace2, trace3]

# layoutumuzu oluşturduk

layout = dict(title = 'Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years',

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Citation',ticklen= 5,zeroline= False)

             )

# figüremizi oluşturp datayı ve layoutu içine gömdük. Tipini ise dictionary yaptık

fig = dict(data = data, layout = layout)

# çizdirdik.

iplot(fig)
timesData.head()
# prepare data frames

# timesData set içinde yılı 2014'e eşit olan ilk 3 kaydın tüm featurelerini timesData

# içine filtreleyerek bir dataframe elde ettik ve bu dataframe'i df2014 değişkenine atadık.

# Not: timesData nın içine  "timesData.year==2014" filtresini gömerek 2014 yılına ait kayıtları gördük

# filtre= timesData.year == 2014]

# ve  "illoc[:3,:]" ile (0-2) indexleri arasındaki kayıtları ve ":" ile tüm featurelere eriştik.

df2014 = timesData[timesData.year==2014].iloc[:3,:]

# filter = timesData.year==2014

# df2014 = timesData[filter].iloc[:3,:]

df2014
# create trace1

trace1 = go.Bar(

                x = df2014.university_name,

                y = df2014.citations,

                name = "citations",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df2014.country)

# create trace2 

trace2 = go.Bar(

                x = df2014.university_name,

                y = df2014.teaching,

                name = "teaching",

                marker = dict(color = 'rgba(255, 255, 128, 0.5)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df2014.country)

data = [trace1, trace2]

# barmod ile gruplayarak layout'u oluşturduk.

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# Traceler bu sefer go kütüphanesi ile bir method(bar,scatter vs..) çağırmadan yapılıyor.

# bu methodu dışarı çağırmak yerine içeride type'sine verilerek oluşturuluyor

# Burada traceler bir methoda gömülmek yerine Dictionary'e gömülerek, 

# tip (type) parametresine verilen input ile hangi plot olacağı söyleniyor.



# x parametresi ortak olduğu için dışarıda belirlenerek x değişkenine aktarılıyor

# trace dictionary içinde iki tracedeki x parametresinde x değişkenimizi veriyoruz.

x = df2014.university_name 



trace1 = {

  'x': x,   #dışarıda oluşturduğumuz x değişkeni parametreye input olarak veriliyor

  'y': df2014.citations,

  'name': 'citation',

  'type': 'bar'   # plotun hangi tip plot olduğu söyleniyor.

};

trace2 = {

  'x': x,  #dışarıda oluşturduğumuz x değişkeni parametreye input olarak veriliyor

  'y': df2014.teaching,

  'name': 'teaching',

  'type': 'bar'   # plotun hangi tip plot olduğu söyleniyor.

};

data = [trace1, trace2];

layout = {

  'xaxis': {'title': 'Top 3 universities'},

  'barmode': 'relative', # relative ile yan yana değilde üst üste barlar olmasını sağladık.

  'title': 'citations and teaching of top 3 universities in 2014'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)
df2016

# import graph objects as "go" and import tools

import plotly.graph_objs as go

from plotly import tools

import matplotlib.pyplot as plt

# prepare data frames

df2016 = timesData[timesData.year == 2016].iloc[:7,:] # 2016 yılının ilk 7 kaydına ulaştık



y_saving = [each for each in df2016.research] # 2016 yılındaki araştırmalar

y_net_worth  = [float(each) for each in df2016.income]  #2016 yılındaki gelir bilgileri float tipine çevrilerek y_net_wort'e atandı

x_saving = [each for each in df2016.university_name]  #2016 yılındaki ilk 7 üniversitenin isimleri

x_net_worth  = [each for each in df2016.university_name]

trace0 = go.Bar(

                x=y_saving,

                y=x_saving,

                marker=dict(color='rgba(171, 50, 96, 0.6)',line=dict(color='rgba(171, 50, 96, 1.0)',width=1)),

                name='research',

                orientation='h',

)

trace1 = go.Scatter(

                x=y_net_worth,

                y=x_net_worth,

                mode='lines+markers',

                line=dict(color='rgb(63, 72, 204)'),

                name='income',

)

layout = dict(

                title='Citations and income',

                yaxis=dict(showticklabels=True,domain=[0, 0.85]),

                yaxis2=dict(showline=True,showticklabels=False,linecolor='rgba(102, 102, 102, 0.8)',linewidth=2,domain=[0, 0.85]),

                xaxis=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0, 0.42]),

                xaxis2=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0.47, 1],side='top',dtick=25),

                legend=dict(x=0.029,y=1.038,font=dict(size=10) ),

                margin=dict(l=200, r=20,t=70,b=70),

                paper_bgcolor='rgb(248, 248, 255)',

                plot_bgcolor='rgb(248, 248, 255)',

)

annotations = []

y_s = np.round(y_saving, decimals=2)

y_nw = np.rint(y_net_worth)

# Adding labels

for ydn, yd, xd in zip(y_nw, y_s, x_saving):

    # labeling the scatter savings

    annotations.append(dict(xref='x2', yref='y2', y=xd, x=ydn - 4,text='{:,}'.format(ydn),font=dict(family='Arial', size=12,color='rgb(63, 72, 204)'),showarrow=False))

    # labeling the bar net worth

    annotations.append(dict(xref='x1', yref='y1', y=xd, x=yd + 3,text=str(yd),font=dict(family='Arial', size=12,color='rgb(171, 50, 96)'),showarrow=False))



layout['annotations'] = annotations



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,

                          shared_yaxes=False, vertical_spacing=0.001)



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)



fig['layout'].update(layout)

iplot(fig)
timesData[timesData.year==2016].iloc[:7,:]
# data preparation

df2016 = timesData[timesData.year == 2016].iloc[:7,:] #2016 yılına göre filtrelendi ve ilk 7 üniversite alındı

pie1 = df2016.num_students #öğrenci sayıları pie1'e atandı

# num_students içerisindeki değerlerin ondalık kısımları "," ile belirtilmiş bu türkçe yanlış gösterimdir ve veriler object türündedir bunlar float olmalıdır.

# Bu ifadeleri "." lı ondalık haline getirmeliyiz örneğin==> 2,243 = 2.243

# Aşağıda replace ile (,) kısımlar (.)'ya çevrildi ve her bir num_students değeri object tipinden float tipine çevrildi.'

pie1_list = [float(each.replace(',', '.')) for each in df2016.num_students]  # str(2,4) => str(2.4) = > float(2.4) = 2.4

#etiketlere 2016 yılındaki ilk 7 üniversitenin isimleri atandı.

labels = df2016.university_name

# Bu yöntemde figüre içine data ve layout gömülü olarak oluşturulur

# values gösterilecek değerler

# figure

fig = {

  "data": [

    { #trace oluşturuluyor

      "values": pie1_list,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "Number Of Students Rates",

      "hoverinfo":"label+percent+name", #oranı,yüzdesi,adı

      "hole": .2,  # oluşacak pie çhartı ortasındaki deliğin büyüklüğü

      "type": "pie" # trace'in tipi

    },],

  "layout": {  #layout oluşturuluyor

        "title":"Universities Number of Students rates",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "Number of Students",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

iplot(fig)



# Not: Oranlama işlemini Pie Chart'ın kendisi yapar. Mantığı.

# İlk 7 üniversitedeki öğrencilerin sayıları toplanır ==> 95.724....

# Örneğin Harvard Üni'nin  100delik dilimde oranı için yapılacak işlem :

# Harvard Öğrenci sayısı = 20.152  == > 20.152*100 / 95.724 = 21.05' gibi bir orana denk gelir

# Renklerin açıklamasının olduğu kısımda üniversitelerin üzerine basarak, dahil edilmediği taktirde olabilecek oranlamayı gösterir.
df2016.head(20)
# Bubble Charts Example: University world rank (first 20) vs 

# teaching score with number of students(size) and international score (color) in 2016



# data preparation

df2016 = timesData[timesData.year == 2016].iloc[:20,:]

#num_students ',' ondalıkları '.' ya çevrildi ve float tipine dönüştürüldü ve bir listeye aktarıldı.

num_students_size  = [float(each.replace(',', '.')) for each in df2016.num_students]

#uluslararası puanları renk değeri olarak bir listeye atadık.

international_color = [float(each) for each in df2016.international]

# Datayı oluşturuyoruz ve içinde trace gömüyoruz.

data = [

    {

        'y': df2016.teaching,

        'x': df2016.world_rank,

        'mode': 'markers',

        'marker': { #color ve size'a anlam katıyoruz.

            'color': international_color, # uluslararası skora göre renk yoğunlaşır

            'size': num_students_size, # üniversiteki öğrenci sayısına göre size büyür.

            'showscale': True

        },

        "text" :  df2016.university_name    

    }

]

iplot(data)



#Bublelerin (yuvarlak,kabarcık) boyutu büyüdükçe ilgili üniversitenin öğrenci sayısınında arttığını gösterir.

#Bublelerin rengi açıldıkça öğretimininde(öğretim gücü) yükseldiğini anlayabiliriz.

# En açık renk öğretim gücü en düşük üniversite, en koyu renk öğretim gücü en yüksek üniversite.
# prepare data

x2011 = timesData.student_staff_ratio[timesData.year == 2011] #2011 yılı filtrelendi ve student_staff aktarıldı

x2012 = timesData.student_staff_ratio[timesData.year == 2012] #2012 yılı filtrelendi ve student_staff aktarıldı

# Histogram Data içerisindeki x değerinden kaç adet olduğunu yani data yoğunluğunu count eder.

trace1 = go.Histogram(

    x=x2011,  #saydamlık

    opacity=0.75,

    name = "2011",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))

trace2 = go.Histogram(

    x=x2012,

    opacity=0.75,

    name = "2012",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



data = [trace1, trace2]

layout = go.Layout(barmode='overlay',  # barmode modu overlay: içiçe gelecek şekilde birleştir.

                   title=' students-staff ratio in 2011 and 2012',

                   xaxis=dict(title='students-staff ratio'),

                   yaxis=dict( title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
# data prepararion

# En sık geçen kelimeleri büyük yazdırır.

# En seyrek geçen kelimeleri küçük yazdırır.

x2011 = timesData.country[timesData.year == 2011]

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(x2011)) #kullanılan kelimeleri ayır ve en çok kullanılanları oranlayarak büyüt.

plt.imshow(wordcloud)

plt.axis('off')  # x ve y eksenlerini kapat





plt.show()
# data preparation

# 2015 yılındaki üniversitelerin toplam skoru ve araştırma puanını detaylı olarak görselleştirmek

x2015 = timesData[timesData.year == 2015]



trace0 = go.Box(

    y=x2015.total_score,

    name = 'total score of universities in 2015',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

trace1 = go.Box(

    y=x2015.research,

    name = 'research of universities in 2015',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

data = [trace0, trace1]

iplot(data)

# plotly kütüphanesi ile boxplotu biraz daha canlı hale getirmek bu şekilde mümküdür.

# Hangi nokta hangi değerlere sahip, lower,higher quartile gibi outlierlarıda görmek mümkündür.
# import figure factory

import plotly.figure_factory as ff

# prepare data

dataframe = timesData[timesData.year == 2015]

data2015 = dataframe.loc[:,["research","international", "total_score"]]

data2015["index"] = np.arange(1,len(data2015)+1)

#scatter matrix

#diag (column and row) = boxplot, portland="koyusu kırmızı, açığı mavi olan renk paleti"

fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',colormap_type='cat',height=700, width=700)

iplot(fig)
dataframe.head()
# Üniversitelerin Teachin(Öğretme) ve income(Gelirler) featureleri karşılaştırılacak.

# dataframe = 2015 yılına ait üniversiteler

# first line plot

trace1 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.teaching,

    name = "teaching",  #label

    marker = dict(color = 'rgba(16, 112, 2, 0.8)'), # RGB + Opacity(alpha)

)

# second line plot

# 2. plotumuz yardımcı plotumuz x ve y axisleri x2,y2 olarak veriliyor

trace2 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.income,

    xaxis='x2',  

    yaxis='y2',

    name = "income", #label

    marker = dict(color = 'rgba(160, 112, 20, 0.8)'), # RGB + Opacity(alpha)

)

data = [trace1, trace2]

layout = go.Layout( # iki plot iç içe olduğu için domain ve anchor kullanılıyor. domain yer,kordinat belirtir. anchor ikinci plotu çizdirmek için kullandığımız bir parametre.

    xaxis2=dict(

        domain=[0.6, 0.95],# iki değerin farkı plotun genişliğidir, 0.6 plotun başlayacağı nokta(1.plotun 6.çizgisi), 0.95 bitiş noktası(ilk plotun 9. ile 10. çizgisinin arası)

        anchor='y2', #ikinci plotumuzun x ekseni, ilk plotumuzun y ekseninde yer aldığı için       

    ),

    yaxis2=dict(

        domain=[0.6, 0.95],

        anchor='x2', #ikinci plotumuzun y ekseni, ilk plotumuzun x ekseninde sabit yer aldığı için alacağı konum.

    ),

    title = 'Income and Teaching vs World Rank of Universities'



)



fig = go.Figure(data=data, layout=layout)

iplot(fig)

dataframe.head()
# create trace 1 that is 3d scatter

trace1 = go.Scatter3d(

    x=dataframe.world_rank, # 1d Dünya sıralaması

    y=dataframe.research,   # 2d Araştırma Puanı

    z=dataframe.citations,  # 3d Alıntı Puanı

    mode='markers', #Mode marker = ..... şeklinde gösterim.

    marker=dict(

        size=10, 

        color='rgb(255,0,0)'         

    )

)



data = [trace1]

layout = go.Layout(

    margin=dict(   #margin(pay)  kenarlardan bırakılan boşluklar(paylar)

        l=0,   #left    (soldan)

        r=0,   #right   (sağdan)

        b=0,   #below   (alttan)

        t=0    #top     (üstten)

    )

    

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace1 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.research,

    name = "research"

)

trace2 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.citations,

    #domain1

    xaxis='x2',

    yaxis='y2',

    name = "citations"

)

trace3 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.income,

    #domain2

    xaxis='x3',

    yaxis='y3',

    name = "income"

)

trace4 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.total_score,

    #domain3

    xaxis='x4', 

    yaxis='y4',

    name = "total_score"

)

data = [trace1, trace2, trace3, trace4]

layout = go.Layout(

    xaxis=dict(

        domain=[0, 0.45]

    ),

    xaxis2=dict(

        domain=[0.55, 1]

    ),

    xaxis3=dict(

        domain=[0, 0.45],

        anchor='y3'

    ),

    xaxis4=dict(

        #domain=[0.55, 1],

        domain=[0.55, 1],

        

    ),

    yaxis=dict(

        domain=[0, 0.45]

    ),

    yaxis2=dict(

        domain=[0.55, 1],

        anchor='x2'

    ),

    yaxis3=dict(

        domain=[0.55, 1]

    ),

    yaxis4=dict(

            

        domain=[0, 0.45],

        anchor='x4'

    ),

    title = 'Research, citation, income and total score VS World Rank of Universities'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)