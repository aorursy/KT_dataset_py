# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import plotly as py

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly import tools

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
global_=pd.read_csv("../input/world-religions/global.csv")

national=pd.read_csv("../input/world-religions/national.csv")

regional=pd.read_csv("../input/world-religions/regional.csv")
values={"UKG":"GBR","BHM":"BHS","HAI":"HTI","TRI":"TTO","BAR":"BRB","GRN":"GRD","SLU":"LCA","SVG":"VCT","AAB":"ATG","SKN":"KNA",

       "GUA":"GTM","HON":"HND","SAL":"SLV","COS":"CRI","PAR":"PRY","URU":"URY","IRE":"IRL","NTH":"NLD","FRN":"FRA","MNC":"MCO",

       "SWZ":"CHE","SPN":"ESP","PAR":"PRT","GMY":"DEU","GDR":"DDR","AUS":"AUT","CZE":"CSK","CZR":"CZE","SLO":"SVK","SNM":"SMR",

       "MNG":"MNE","MAC":"MKD","CRO":"HRV","BOS":"BIH","SLV":"SVN","BUL":"BGR","MLD":"MDA","ROM":"ROU","LAT":"LVA","LIT":"LTU",

       "GRG":"GEO","SWD":"SWE","DEN":"DNK","ICE":"ISL","CAP":"CPV","EQG":"GNQ","GAM":"GMB","MAA":"MRT","NIR":"NER","CDI":"CIV",

       "GUI":"GIN","BFO":"BFA","SIE":"SLE","TOG":"TGO","CAO":"CMR","NIG":"NGA","CEN":"CAF","CHA":"TCD","CON":"COG","DRC":"COD",

       "TAZ":"TZA","BUI":"BDI","ANG":"AGO","MZM":"MOZ","ZAM":"ZMB","ZIM":"ZWE","MAW":"MWI","SAF":"ZAF","LES":"LSO","BOT":"BWA",

       "SWA":"SWZ","MAG":"MDG","MAS":"MUS","SEY":"SYC","MOR":"MAR","ALG":"DZA","LIB":"LBY","SUD":"SDN","LEB":"LBN",

        "YAR":"YEM","YPR":"YEM","KUW":"KWT","BAH":"BHR","UAE":"ARE","OMA":"OMN","TAJ":"TJK","KYR":"KGZ","KZK":"KAZ","MON":"MNG",

        "TAW":"TWN","ROK":"KOR","BHU":"BTN","BNG":"BGD","MYA":"MMR","SRI":"LKA","MAD":"MDV","NEP":"NPL","THI":"THA","CAM":"KHM",

       "DRV":"VNM","RWN":"VDR","MAL":"MYS","SIN":"SGP","BRU":"BRN","PHI":"PHL","INS":"IDN","ETM":"TMP","AUL":"AUS","NEW":"NZL",

       "VAN":"VUT","SOL":"SLB","FIJ":"FJI","NAU":"NRU","MSI":"MHL","PAL":"PLW"}

national["code"]=national["code"].replace(values)
fig = go.Figure()

fig.add_trace(go.Scatter(

                x=global_.year,

                y=global_.islam_sunni,

                name="Sunni",

                line = dict(color='royalblue', width=4, dash='dash'),

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=global_.year,

                y=global_["islam_shi’a"] ,

                name="Shi",

               line = dict(color='dimgray', width=4, dash='dot'),

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=global_.year,

                y=global_["islam_ibadhi"] ,

                name="Ibadhi",

               line = dict(color='orange', width=4, dash='dot'),

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=global_.year,

                y=global_["islam_alawite" ] ,

                name="Alawite",

                line = dict(color='red', width=4, dash='dash'),

                opacity=0.8))

fig.add_trace(go.Scatter(

                x=global_.year,

                y=global_["islam_ahmadiyya" ] ,

                name="Ahmadiyya",

                line = dict(color='purple', width=4, dash='dot'),

                opacity=0.8))

fig.add_trace(go.Scatter(

                x=global_.year,

                y=global_["islam_other" ] ,

                name="Other",

                line = dict(color='black', width=4, dash='dot'),

                opacity=0.8))

fig.update_layout(title_text="NUMBER OF MUSLIMS BY YEAR",   paper_bgcolor='lavenderblush',font=dict(family="sans", size=15,color="black"),

              plot_bgcolor='ivory', xaxis= dict(title= 'Year'),yaxis= dict(title= 'Number'))

fig.show()
import plotly.express as px



fig = px.choropleth(national, locations="code",

                    color="islam_sunni",

                    hover_name="state", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Greens,

                     animation_frame="year",animation_group="code")

fig.update_layout(

    title_text = 'SUNNI IN COUNTRIES BETWEEN 1945-2010')



fig.show()
import plotly.express as px



fig = px.choropleth(national, locations="code",

                    color="islam_shi’a",

                    hover_name="code", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Greens,

                     animation_frame="year",animation_group="code")



fig.update_layout(

    title_text = 'SHIA IN COUNTRIES BETWEEN 1945-2010')





fig.show()
fig = px.choropleth(national, locations="code",

                    color="islam_all",

                    hover_name="state", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Greens,

                     animation_frame="year",animation_group="code")



fig.update_layout(

    title_text = 'MUSLIMS IN COUNTRIES BETWEEN 1945-2010')

fig.show()
trace1 = go.Bar(

                    x = global_.year,

                    y = global_.christianity_protestant,

                  

                    name = "Protestans",

                    marker = dict(color = 'indianred'))



trace2 = go.Bar(

                    x = global_.year,

                    y = global_.christianity_romancatholic,

                    

                    name = "Roman Catholics",

                    marker = dict(color = 'lightsalmon'))

trace3 = go.Bar(

                    x = global_.year,

                    y = global_.christianity_easternorthodox,

                    marker = dict(color = 'rgb(158,202,225)'),

                    name = "Eastern North Orthodoxes")

trace5 = go.Bar(

                    x = global_.year,

                    y = global_.christianity_anglican,

                    marker = dict(color = 'darkblue'),

                    name = "Anglicans")

trace4 = go.Bar(

                    x = global_.year,

                    y = global_.christianity_other,

                     marker = dict(color = 'royalblue'),

                    name = "Other Christians"

                    )                 



data = [trace1, trace2,trace3,trace4,trace5]

layout = dict(title = 'NUMBER OF CHRISTIANS BY YEARS', barmode='stack',

              xaxis= dict(title= 'Years',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="lightsteelblue"),

              yaxis= dict(title= 'Count',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="lightsteelblue",),

              paper_bgcolor='whitesmoke',

              plot_bgcolor='snow' ,

font=dict(family="DejaVu Sans", size=15,color="black"))

fig = dict(data = data, layout = layout)

iplot(fig)




fig = px.choropleth(national, locations="code",

                    color="christianity_romancatholic",

                    hover_name="state", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Reds,

                     animation_frame="year",animation_group="code")

fig.update_layout(

    title_text = 'ROMAN CATHOLIC CHRISTIANS IN COUNTRIES BETWEEN 1945-2010')



fig.show()
fig = px.choropleth(national, locations="code",

                    color="christianity_protestant",

                    hover_name="state", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Reds,

                     animation_frame="year",animation_group="code")



fig.update_layout(

    title_text = 'PROTESTAN CHRISTIANS IN COUNTRIES BETWEEN 1945-2010')

fig.show()
fig = px.choropleth(national, locations="code",

                    color="christianity_easternorthodox",

                    hover_name="state", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Reds,

                     animation_frame="year",animation_group="code")

fig.update_layout(

    title_text = 'EASTERN ORTHODOX IN COUNTRIES BETWEEN 1945-2010')



fig.show()

fig = px.choropleth(national, locations="code",

                    color="christianity_anglican",

                    hover_name="state", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Reds,

                     animation_frame="year",animation_group="code")



fig.update_layout(

    title_text = 'ANGLICANS IN COUNTRIES BETWEEN 1945-2010')



fig.show()
fig = px.choropleth(national, locations="code",

                    color="christianity_all",

                    hover_name="state", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Reds,

                     animation_frame="year",animation_group="code")

fig.update_layout(

    title_text = 'ALL CHRISTIANS IN COUNTRIES BETWEEN 1945-2010')



fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(

    y=global_.year,

    x=global_.judaism_orthodox,

    name='Judaism Orthodox ',

    orientation='h',

    marker=dict(

        color='rgba(246, 78, 139, 0.6)',

        line=dict(color='rgba(246, 78, 139, 1.0)', width=3)

    )

))

fig.add_trace(go.Bar(

     y=global_.year,

     x=global_.judaism_conservative,

     name='Judaism Conservative ',

    orientation='h',

    marker=dict(

        color='rgba(58, 71, 80, 0.6)',

        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)

    )

))

fig.add_trace(go.Bar(

     y=global_.year,

     x=global_.judaism_reform,

     name='Judaism Reform ',

    orientation='h',

    marker=dict(

        color='rgb(158,202,225)',

        line=dict(color="dodgerblue", width=3))))



fig.add_trace(go.Bar(

     y=global_.year,

     x=global_.judaism_other,

     name='Judaism Other ',

    orientation='h',

    marker=dict(

        color='palegreen',

        line=dict(color="green", width=3))

        

    ))



fig.update_layout(barmode='stack',title="Number of Jewish People by Years",paper_bgcolor='lightgoldenrodyellow',

              plot_bgcolor='whitesmoke',font=dict(family="DejaVu Sans", size=15,color="black"),

                xaxis= dict(title= 'Count',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="tan"),

              yaxis= dict(title= 'Years',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="tan"))

fig.show()
fig = px.choropleth(national, locations="code",

                    color="judaism_orthodox",

                    hover_name="state", 

                    color_continuous_scale=px.colors.sequential.Blues,

                     animation_frame="year",animation_group="code")



fig.update_layout(

    title_text = 'ORTHODOX JEWISH IN COUNTRIES BETWEEN 1945-2010')



fig.show()
fig = px.choropleth(national, locations="code",

                    color="judaism_conservative",

                    hover_name="state", 

                    color_continuous_scale=px.colors.sequential.Blues,

                     animation_frame="year",animation_group="code")

fig.update_layout(

    title_text = 'CONSERVAIVE JEWISH IN COUNTRIES BETWEEN 1945-2010')



fig.show()
fig = px.choropleth(national, locations="code",

                    color="judaism_reform",

                    hover_name="state", 

                    color_continuous_scale=px.colors.sequential.Blues,

                     animation_frame="year",animation_group="code")



fig.update_layout(

    title_text = 'REFORMIST JEWISH IN COUNTRIES BETWEEN 1945-2010')



fig.show()
fig = px.choropleth(national, locations="code",

                    color="judaism_all",

                    hover_name="state", 

                    color_continuous_scale=px.colors.sequential.Blues,

                     animation_frame="year",animation_group="code")

fig.update_layout(

    title_text = 'ALL JEWISH PEOPLE IN COUNTRIES BETWEEN 1945-2010')



fig.show()
trace1 = go.Scatter(

                    y = global_.year,

                    x = global_.buddhism_all ,

                    mode = "markers",

                    name = "Buddhism",

                    marker = dict(color = 'rgba(28, 149, 249, 0.8)',symbol=220,size=12))



trace2 = go.Scatter(

                    y = global_.year,

                    x = global_.hinduism_all  ,

                    mode = "markers",

                    name = "Hinduism",

                    marker = dict(color = 'rgba(150, 26, 80, 0.8)',symbol=13,size=12))

trace3 = go.Scatter(

                    y = global_.year,

                    x = global_.sikhism_all   ,

                    mode = "markers",

                    name = "Sikhism",

                    marker = dict(color = 'red',symbol=4,size=12))

trace4 = go.Scatter(

                      y = global_.year,

                    x = global_.shinto_all ,  

                    mode = "markers",

                    name = "Shintoism",

                    marker = dict(color = 'grey',symbol="square",size=12))           



trace5 = go.Scatter(

                      y = global_.year,

                    x = global_.taoism_all ,  

                    mode = "markers",

                    name = "Taoism",

                    marker = dict(color = 'aqua',symbol=2,size=12)) 





trace6 = go.Scatter(

                      y = global_.year,

                    x = global_.confucianism_all ,  

                    mode = "markers",

                    name = "Confucianism",

                    marker = dict(color = "green",symbol='bowtie',size=12)) 



data = [trace1, trace2,trace3,trace4,trace5,trace6]

layout = dict(title = 'Buddhism, Hinduism, Sikhism, Shintoism, Taoism and Confucianism Believers by Years',

              xaxis= dict(title= 'Count',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="tan"),

              yaxis= dict(title= 'Years',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="tan",),

              paper_bgcolor='tan',

              plot_bgcolor='whitesmoke',font=dict(family="DejaVu Sans", size=15,color="black" ))

fig = dict(data = data, layout = layout)

iplot(fig)
fig = px.choropleth(national, locations="code",

                    color="buddhism_all",

                    hover_name="state", 

                    color_continuous_scale='Magma',

                     animation_frame="year",animation_group="code")



fig.update_layout(

    title_text = 'ALL BUDDHISTS IN COUNTRIES BETWEEN 1945-2010')



fig.show()
fig = px.choropleth(national, locations="code",

                    color="hinduism_all",

                    hover_name="state", 

                     color_continuous_scale='Magma',

                     animation_frame="year",animation_group="code")

fig.update_layout(

    title_text = 'ALL HINDUISTS IN COUNTRIES BETWEEN 1945-2010')



fig.show()
trace1 = go.Scatter(

                      x = global_.year,

                    y = global_.jainism_all  ,  

                    mode = "markers",

                    name = "Jainism",

                    marker = dict(color = 'orange',symbol=2,size=8)) 





trace2 = go.Scatter(

                      x = global_.year,

                    y = global_["baha’i_all"]  ,  

                    mode = "markers",

                    name = "Bahaism",

                    marker = dict(color = 'indianred',symbol=207,size=8)) 



trace3 = go.Scatter(

                    x = global_.year,

                    y = global_.zoroastrianism_all  ,

                    mode = "markers",

                    name = "Zoroastrianism",

                      marker = dict(color = 'rgba(249, 94, 28, 0.8)',symbol=306,size=8)) 

                  

trace4 = go.Scatter(

                      x = global_.year,

                    y = global_.syncretism_all ,  

                    mode = "markers",

                    name = "Syncretism",

                    marker = dict(color = 'cadetblue',symbol=11,size=8)) 

trace5 = go.Scatter(

                      x = global_.year,

                    y = global_.animism_all ,  

                    mode = "markers",

                    name = "Animism",

                    marker = dict(color = 'teal',symbol=119,size=8)) 

trace6 = go.Scatter(

                      x = global_.year,

                    y = global_.noreligion_all ,  

                    mode = "markers",

                    name = "No Religion",

                    marker = dict(color = 'firebrick',symbol=203,size=8)) 

trace7 = go.Scatter(

                      x = global_.year,

                    y = global_.otherreligion_all ,  

                    mode = "markers",

                    name = "Other Religion",

                    marker = dict(color = 'olive',symbol=214,size=8))  

data = [trace1, trace2,trace3,trace4,trace5,trace6,trace7]

layout = dict(title = 'Jainsm, Bahaism, Zoroastrianism, Syncretism, Animism and Other Religion Believers and Nonbelievers by Years',

              xaxis= dict(title= 'Years',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="lightgrey"),

              yaxis= dict(title= 'Count',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="lightgrey",),

              paper_bgcolor='lightgrey',

              plot_bgcolor='whitesmoke',font=dict(family="DejaVu Sans", size=15,color="black" ))

fig = dict(data = data, layout = layout)

iplot(fig)
fig = px.choropleth(national, locations="code",

                    color="noreligion_all",

                    hover_name="state", 

                    color_continuous_scale='Magma',

                     animation_frame="year",animation_group="code")



fig.update_layout(

    title_text = 'ALL NON RELIGIONISTS IN COUNTRIES BETWEEN 1945-2010')



fig.show()
global_["otherreligion"]=global_["otherreligion_all"]+global_["baha’i_all"]+global_["sikhism_all"]+global_["zoroastrianism_all"]+global_["jainism_all"]+global_["syncretism_all"]+global_["animism_all"]
data=global_[["year", "christianity_all" , "judaism_other" , "islam_all", 

"buddhism_all", "hinduism_all", "shinto_all", "taoism_all", "confucianism_all",                             

"noreligion_all", "otherreligion" ]]

data1=data.set_index("year")
fig = go.Figure(data=go.Heatmap(

                    z=data.values,

                   y=data.year,

                   x=data.columns,colorscale="YlOrRd", showscale=True))

fig.update_layout(

    title_text = 'NUMBER OF ALL RELIGIONS BETWEEN 1945-2010')



fig.show()
national["otherreligion"]=national["otherreligion_all"]+national["baha’i_all"]+national["sikhism_all"]+national["zoroastrianism_all"]+national["jainism_all"]+national["syncretism_all"]+national["animism_all"]



national=national[["year","state","code","christianity_all" , "judaism_all" , "islam_all", 

"buddhism_all", "hinduism_all", "shinto_all", "taoism_all", "confucianism_all",                             

"noreligion_all", "otherreligion","population"]]
region=413*["West Ham"]+463*["Europe"]+525*["Africa"]+205*["Middle East"]+389*["Asia"]
national["region"]=region
fig, axs = plt.subplots(3, 5,figsize=(40,25))

fig.patch.set_facecolor('seashell')



axs[0,2].text(-1.7, 2, "Religion Percentage by Years", fontsize=30)

axs[0, 0].pie( data1.iloc[0].values, shadow=True, startangle=90,autopct='%1.1f%%',textprops={'size': 20})

axs[0, 0].set_title("1945",size=20)





axs[0, 1].pie( data1.iloc[1].values, shadow=True, startangle=90,autopct='%1.1f%%',textprops={'size': 20})

axs[0, 1].set_title("1950",size=20)





axs[0, 2].pie( data1.iloc[2].values, shadow=True, startangle=90,autopct='%1.1f%%',textprops={'size': 20})

axs[0, 2].set_title("1955",size=20)





axs[0, 3].pie( data1.iloc[3].values, shadow=True, startangle=90,autopct='%1.1f%%',textprops={'size': 20})

axs[0, 3].set_title("1960",size=20)





axs[0, 4].pie( data1.iloc[4].values, shadow=True, startangle=90,autopct='%1.1f%%',textprops={'size': 20})

axs[0, 4].set_title("1965",size=20)





axs[1, 0].pie( data1.iloc[5].values, shadow=True, startangle=90,autopct='%1.1f%%',textprops={'size': 20})



axs[1, 0].set_title("1970",size=20)





axs[1, 1].pie( data1.iloc[6].values, shadow=True, startangle=90,autopct='%1.1f%%',textprops={'size': 20})



axs[1, 1].set_title("1975",size=20)





axs[1, 2].pie( data1.iloc[7].values, shadow=True, startangle=90,autopct='%1.1f%%',textprops={'size': 20})



axs[1, 2].set_title("1980",size=20)





axs[1, 3].pie( data1.iloc[8].values, shadow=True, startangle=90,autopct='%1.1f%%',textprops={'size': 20})

axs[1, 3].set_title("1985",size=20)





axs[1, 4].pie( data1.iloc[9].values, shadow=True, startangle=90,autopct='%1.1f%%',textprops={'size': 20})

axs[1, 4].set_title("1990",size=20)





axs[2, 0].pie( data1.iloc[10].values, shadow=True, startangle=90,autopct='%1.1f%%',textprops={'size': 20})

axs[2, 0].set_title("1995",size=20)





axs[2, 1].pie( data1.iloc[11].values, shadow=True, startangle=90,autopct='%1.1f%%',textprops={'size': 20})

axs[2, 1].set_title("2000",size=20)





axs[2, 2].pie( data1.iloc[12].values, shadow=True, startangle=90,autopct='%1.1f%%',textprops={'size': 20})

axs[2, 2].set_title("2005",size=20)





axs[2, 3].pie( data1.iloc[13].values, shadow=True, startangle=90,autopct='%1.1f%%' ,textprops={'size': 20})

axs[2, 3].set_title("2010",size=20)





axs[2,4].pie([2,2,2,2,2,2,2,2,2,2])



plt.legend(data1.iloc[0].index,title="Religions",loc="best",fontsize=20)





plt.show()
import plotly.express as px

px.scatter(national, x="islam_all", y="christianity_all",animation_frame="year", animation_group="state",size="population",

           color="region", hover_name="state",symbol="region")
px.scatter(national, x="buddhism_all", y="hinduism_all",size="population",animation_frame="year", animation_group="state",

           color="region", hover_name="state",symbol='region')
national2=pd.read_csv("../input/world-religions/national.csv")
values={"UKG":"GBR","BHM":"BHS","HAI":"HTI","TRI":"TTO","BAR":"BRB","GRN":"GRD","SLU":"LCA","SVG":"VCT","AAB":"ATG","SKN":"KNA",

       "GUA":"GTM","HON":"HND","SAL":"SLV","COS":"CRI","PAR":"PRY","URU":"URY","IRE":"IRL","NTH":"NLD","FRN":"FRA","MNC":"MCO",

       "SWZ":"CHE","SPN":"ESP","PAR":"PRT","GMY":"DEU","GDR":"DDR","AUS":"AUT","CZE":"CSK","CZR":"CZE","SLO":"SVK","SNM":"SMR",

       "MNG":"MNE","MAC":"MKD","CRO":"HRV","BOS":"BIH","SLV":"SVN","BUL":"BGR","MLD":"MDA","ROM":"ROU","LAT":"LVA","LIT":"LTU",

       "GRG":"GEO","SWD":"SWE","DEN":"DNK","ICE":"ISL","CAP":"CPV","EQG":"GNQ","GAM":"GMB","MAA":"MRT","NIR":"NER","CDI":"CIV",

       "GUI":"GIN","BFO":"BFA","SIE":"SLE","TOG":"TGO","CAO":"CMR","NIG":"NGA","CEN":"CAF","CHA":"TCD","CON":"COG","DRC":"COD",

       "TAZ":"TZA","BUI":"BDI","ANG":"AGO","MZM":"MOZ","ZAM":"ZMB","ZIM":"ZWE","MAW":"MWI","SAF":"ZAF","LES":"LSO","BOT":"BWA",

       "SWA":"SWZ","MAG":"MDG","MAS":"MUS","SEY":"SYC","MOR":"MAR","ALG":"DZA","LIB":"LBY","SUD":"SDN","LEB":"LBN",

        "YAR":"YEM","YPR":"YEM","KUW":"KWT","BAH":"BHR","UAE":"ARE","OMA":"OMN","TAJ":"TJK","KYR":"KGZ","KZK":"KAZ","MON":"MNG",

        "TAW":"TWN","ROK":"KOR","BHU":"BTN","BNG":"BGD","MYA":"MMR","SRI":"LKA","MAD":"MDV","NEP":"NPL","THI":"THA","CAM":"KHM",

       "DRV":"VNM","RWN":"VDR","MAL":"MYS","SIN":"SGP","BRU":"BRN","PHI":"PHL","INS":"IDN","ETM":"TMP","AUL":"AUS","NEW":"NZL",

       "VAN":"VUT","SOL":"SLB","FIJ":"FJI","NAU":"NRU","MSI":"MHL","PAL":"PLW"}

national2["code"]=national2["code"].replace(values)
national2["otherreligion"]=national2["otherreligion_all"]+national2["baha’i_all"]+national2["sikhism_all"]+national2["zoroastrianism_all"]+national2["jainism_all"]+national2["syncretism_all"]+national2["animism_all"]+national2["confucianism_all"]+national2["taoism_all"]+national2["shinto_all"]



national2=national2[["year","state","code","christianity_all" , "judaism_all" , "islam_all", 

"buddhism_all", "hinduism_all","noreligion_all", "otherreligion","population"]]
n=national2.drop(["state","year","code","population"],axis=1)
n=n.assign(max_value=n.values.max(1))
national2["max_value"]=n.max_value
b=1995*[0]

national2["religion"]=b

c=1995*[0]

national2["religion2"]=c

chris=national2[national2.christianity_all==national2.max_value]

chris["religion"]=chris["religion"].replace({0:"Christianism"})

chris["religion2"]=chris["religion2"].replace({0:0})

jewish=national2[national2.judaism_all==national2.max_value]

jewish["religion"]=jewish["religion"].replace({0:"Judaism"})

jewish["religion2"]=jewish["religion2"].replace({0:1})

islam=national2[national2.islam_all==national2.max_value]

islam["religion"]=islam["religion"].replace({0:"Islam"})

islam["religion2"]=islam["religion2"].replace({0:2})

bud=national2[national2.buddhism_all==national2.max_value]

bud["religion"]=bud["religion"].replace({0:"Buddhism"})

bud["religion2"]=bud["religion2"].replace({0:3})

hindu=national2[national2.hinduism_all==national2.max_value]

hindu["religion"]=hindu["religion"].replace({0:"Hinduism"})

hindu["religion2"]=hindu["religion2"].replace({0:4})

norel=national2[national2.noreligion_all==national2.max_value]

norel["religion"]=norel["religion"].replace({0:"No religion"})

norel["religion2"]=norel["religion2"].replace({0:5})

other=national2[national2.otherreligion==national2.max_value]

other["religion"]=other["religion"].replace({0:"Other Religion"})

other["religion2"]=other["religion2"].replace({0:6})
frames=[chris,jewish,islam,bud,hindu,norel,other]

result = pd.concat(frames)
result["percent"]=(result["max_value"]/result["population"])*100
fig = px.choropleth(result, locations="code",

                    color="religion2",

                    hover_data=["percent","state"],

                    hover_name="religion",

                     color_continuous_scale='Magma',

                     animation_frame="year",animation_group="state")

fig.update_layout(

    title_text = 'THE MOST BELIEVED RELIGIONS IN COUNTRIES BETWEEN 1945-2010')



fig.show()