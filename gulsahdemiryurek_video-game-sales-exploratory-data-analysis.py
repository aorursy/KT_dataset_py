import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.plotly as py

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly.tools import FigureFactory as ff

import matplotlib.pyplot as plt

from wordcloud import WordCloud,STOPWORDS

from PIL import Image

import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.
vgsales=pd.read_csv("../input/videogamesales/vgsales.csv")
vgsales.info()
d=vgsales.head(4)

colorscale = "YlOrRd"

table = ff.create_table(d,colorscale=colorscale)

for i in range(len(table.layout.annotations)):

    table.layout.annotations[i].font.size = 9

iplot(table)


df=vgsales.head(100)
trace1 = go.Scatter(

                    x = df.Rank,

                    y = df.NA_Sales,

                    mode = "markers",

                    name = "North America",

                    marker = dict(color = 'rgba(28, 149, 249, 0.8)',size=8),

                    text= df.Name)



trace2 = go.Scatter(

                    x = df.Rank,

                    y = df.EU_Sales,

                    mode = "markers",

                    name = "Europe",

                    marker = dict(color = 'rgba(249, 94, 28, 0.8)',size=8),

                    text= df.Name)

trace3 = go.Scatter(

                    x = df.Rank,

                    y = df.JP_Sales,

                    mode = "markers",

                    name = "Japan",

                    marker = dict(color = 'rgba(150, 26, 80, 0.8)',size=8),

                    text= df.Name)

trace4 = go.Scatter(

                    x = df.Rank,

                    y = df.Other_Sales,

                    mode = "markers",

                    name = "Other",

                    marker = dict(color = 'lime',size=8),

                    text= df.Name)

                    



data = [trace1, trace2,trace3,trace4]

layout = dict(title = 'North America, Europe, Japan and Other Sales of Top 100 Video Games',

              xaxis= dict(title= 'Rank',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="white"),

              yaxis= dict(title= 'Sales(In Millions)',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="white",),

              paper_bgcolor='rgb(243, 243, 243)',

              plot_bgcolor='rgb(243, 243, 243)' )

fig = dict(data = data, layout = layout)

iplot(fig)
fig={

    "data" : [

    {

        'x': df.Rank,

        'y': df.Year,

        'mode': 'markers',

        'marker': {

            "color":df.Global_Sales,

            'size': df.Global_Sales,

            'showscale': True,

            "colorscale":'Blackbody'

        },

        "text" :  "Name:"+ df.Name +","+" Publisher:" + df.Publisher

        

    },

],

"layout":

    {

    "title":"Release Years of Top 100 Video Games According to Global Sales",

    "xaxis":{

        "title":"Rank",

        "gridcolor":'rgb(255, 255, 255)',

        "zerolinewidth":1,

        "ticklen":5,

        "gridwidth":2,

    },

    "yaxis":{

        "title":'Years',

        "gridcolor":'rgb(255, 255, 255)',

        "zerolinewidth":1,

        "ticklen":5,

        "gridwidth":2,

    },

    

    "paper_bgcolor":'rgb(243, 243, 243)',

    "plot_bgcolor":'rgb(243, 243, 243)'

    }}



iplot(fig)
trace = go.Histogram(x=df.Publisher,marker=dict(color="crimson",line=dict(color='black', width=2)),opacity=0.75)

layout = go.Layout(

    title='Numbers of Top 100 Video Games Publishers',

    xaxis=dict(

        title='Publishers'

    ),

    yaxis=dict(

        title='Count'

    ),

    bargap=0.2,

    bargroupgap=0.1, paper_bgcolor='rgb(243, 243, 243)',

    plot_bgcolor="rgb(243, 243, 243)")

fig = go.Figure(data=[trace], layout=layout)

iplot(fig)
df2 = df.loc[:,["Year","Platform","NA_Sales","EU_Sales" ]]



df2["index"] = np.arange(1,len(df)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(df2, diag='box', index='index',colormap='YlOrRd',

                                  colormap_type='seq',

                                  height=1000, width=1200)

iplot(fig)
xaction=vgsales[vgsales.Genre=="Action"]

xsports=vgsales[vgsales.Genre=="Sports"]

xmisc=vgsales[vgsales.Genre=="Misc"]

xrole=vgsales[vgsales.Genre=="Role-Playing"]

xshooter=vgsales[vgsales.Genre=="Shooter"]

xadventure=vgsales[vgsales.Genre=="Adventure"]

xrace=vgsales[vgsales.Genre=="Racing"]

xplatform=vgsales[vgsales.Genre=="Platform"]

xsimulation=vgsales[vgsales.Genre=="Simulation"]

xfight=vgsales[vgsales.Genre=="Fighting"]

xstrategy=vgsales[vgsales.Genre=="Strategy"]

xpuzzle=vgsales[vgsales.Genre=="Puzzle"]
trace1 = go.Histogram(

    x=xaction.Platform,

    opacity=0.75,

    name = "Action",

    marker=dict(color='rgb(165,0,38)'))

trace2 = go.Histogram(

    x=xsports.Platform,

    opacity=0.75,

    name = "Sports",

    marker=dict(color='rgb(215,48,39)'))

trace3 = go.Histogram(

    x=xmisc.Platform,

    opacity=0.75,

    name = "Misc",

    marker=dict(color='rgb(244,109,67)'))

trace4 = go.Histogram(

    x=xrole.Platform,

    opacity=0.75,

    name = "Role Playing",

    marker=dict(color='rgb(253,174,97)'))

trace5 = go.Histogram(

    x=xshooter.Platform,

    opacity=0.75,

    name = "Shooter",

    marker=dict(color='rgb(254,224,144)'))

trace6 = go.Histogram(

    x=xadventure.Platform,

    opacity=0.75,

    name = "Adventure",

    marker=dict(color='rgb(170,253,87)'))

trace7 = go.Histogram(

    x=xrace.Platform,

    opacity=0.75,

    name = "Racing",

    marker=dict(color='rgb(171,217,233)'))

trace8 = go.Histogram(

    x=xplatform.Platform,

    opacity=0.75,

    name = "Platform",

    marker=dict(color='rgb(116,173,209)'))

trace9 = go.Histogram(

    x=xsimulation.Platform,

    opacity=0.75,

    name = "Simulation",

    marker=dict(color='rgb(69,117,180)'))

trace10 = go.Histogram(

    x=xfight.Platform,

    opacity=0.75,

    name = "Fighting",

    marker=dict(color='rgb(49,54,149)'))

trace11 = go.Histogram(

    x=xstrategy.Platform,

    opacity=0.75,

    name = "Strategy",

    marker=dict(color="rgb(10,77,131)"))

trace12 = go.Histogram(

    x=xpuzzle.Platform,

    opacity=0.75,

    name = "Puzzle",

    marker=dict(color='rgb(1,15,139)'))



data = [trace1, trace2,trace3,trace4,trace5,trace6,trace7,trace8,trace9,trace10,trace11,trace12]

layout = go.Layout(barmode='stack',

                   title='Genre Counts According to Platform',

                   xaxis=dict(title='Platform'),

                   yaxis=dict( title='Count'),

                   paper_bgcolor='beige',

                   plot_bgcolor='beige'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace1 = go.Bar(

    x=xaction.groupby("Platform")["Global_Sales"].sum().index,

    y=xaction.groupby("Platform")["Global_Sales"].sum().values,

    opacity=0.75,

    name = "Action",

    marker=dict(color="rgb(119,172,238)"))

trace2 = go.Bar(

    x=xsports.groupby("Platform")["Global_Sales"].sum().index,

    y=xsports.groupby("Platform")["Global_Sales"].sum().values,

    opacity=0.75,

    name = "Sports",

    marker=dict(color='rgb(21,90,174)'))

trace3 = go.Bar(

    x=xrace.groupby("Platform")["Global_Sales"].sum().index,

    y=xrace.groupby("Platform")["Global_Sales"].sum().values,

    opacity=0.75,

    name = "Racing",

    marker=dict(color="rgb(156,245,163)"))

trace4 = go.Bar(

    x=xshooter.groupby("Platform")["Global_Sales"].sum().index,

    y=xshooter.groupby("Platform")["Global_Sales"].sum().values,

    opacity=0.75,

    name = "Shooter",

    marker=dict(color="rgb(14,135,23)"))

trace5 = go.Bar(

    x=xmisc.groupby("Platform")["Global_Sales"].sum().index,

    y=xmisc.groupby("Platform")["Global_Sales"].sum().values,

    opacity=0.75,

    name = "Misc",

    marker=dict(color='rgb(252,118,103)'))

trace6 = go.Bar(

    x=xrole.groupby("Platform")["Global_Sales"].sum().index,

    y=xrole.groupby("Platform")["Global_Sales"].sum().values,

    opacity=0.75,

    name = "Role Playing",

    marker=dict(color="rgb(226,28,5)"))

trace7 = go.Bar(

    x=xfight.groupby("Platform")["Global_Sales"].sum().index,

    y=xfight.groupby("Platform")["Global_Sales"].sum().values,

    opacity=0.75,

    name = "Fighting",

    marker=dict(color="rgb(247,173,13)"))

trace8 = go.Bar(

    x=xplatform.groupby("Platform")["Global_Sales"].sum().index,

    y=xplatform.groupby("Platform")["Global_Sales"].sum().values,

    opacity=0.75,

    name = "Platform",

    marker=dict(color="rgb(242,122,13)"))

trace9 = go.Bar(

    x=xsimulation.groupby("Platform")["Global_Sales"].sum().index,

    y=xsimulation.groupby("Platform")["Global_Sales"].sum().values,

    opacity=0.75,

    name = "Simulation",

    marker=dict(color="rgb(188,145,202)"))

trace10 = go.Bar(

    x=xadventure.groupby("Platform")["Global_Sales"].sum().index,

    y=xadventure.groupby("Platform")["Global_Sales"].sum().values,

    opacity=0.75,

    name = "Adventure",

    marker=dict(color='rgb(104,57,119)'))

trace11 = go.Bar(

    x=xstrategy.groupby("Platform")["Global_Sales"].sum().index,

    y=xstrategy.groupby("Platform")["Global_Sales"].sum().values,

    opacity=0.75,

    name = "Strategy",

    marker=dict(color='rgb(245,253,104)'))

trace12 = go.Bar(

    x=xpuzzle.groupby("Platform")["Global_Sales"].sum().index,

    y=xpuzzle.groupby("Platform")["Global_Sales"].sum().values,

    opacity=0.75,

    name = "Puzzle",

    marker=dict(color='rgb(138,72,40)'))



data = [trace1, trace2,trace3,trace4,trace5,trace6,trace7,trace8,trace9,trace10,trace11,trace12]

layout = go.Layout(barmode='stack',

                   title='Total Global Sales According to Platform and Genre',

                   xaxis=dict(title='Platform'),

                   yaxis=dict( title='Global Sales(In Millions)'),

                   paper_bgcolor='beige',

                   plot_bgcolor='beige'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
genre=pd.DataFrame(vgsales.groupby("Genre")[["NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales"]].sum())

genre.reset_index(level=0, inplace=True)

genrecount=pd.DataFrame(vgsales["Genre"].value_counts())

genrecount.reset_index(level=0, inplace=True)

genrecount.rename(columns={"Genre": "Counts","index":"Genre"}, inplace=True)



genre=pd.merge(genre,genrecount,on="Genre")
table_data=genre[["Genre","NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales"]]

table_data = table_data.rename(columns = {"NA_Sales": "North America", 

                                  "EU_Sales":"Europe", 

                                  "JP_Sales": "Japan","Other_Sales":"Other","Global_Sales":"Total"})
x=genre.Genre

NA_Perce=list(genre["NA_Sales"]/genre["Global_Sales"]*100)

EU_Perce=list(genre["EU_Sales"]/genre["Global_Sales"]*100)

JP_Perce=list(genre["JP_Sales"]/genre["Global_Sales"]*100)

Other_Perce=list(genre["Other_Sales"]/genre["Global_Sales"]*100)



trace1 = go.Bar(

    x=x,

    y=NA_Perce,

    name="North America" ,

    xaxis='x2', yaxis='y2',

    marker=dict(

        color='rgb(158,202,225)',

        line=dict(

            color='rgb(8,48,107)',

            width=3),

        ),

    opacity=0.75)

trace2 = go.Bar(

    x=x,

    y=EU_Perce,

    xaxis='x2', yaxis='y2',

    marker=dict(

        color='red',

        line=dict(

            color='rgb(8,48,107)',

            width=3),

        ),

    opacity=0.75,

    name = "Europe",

    )

trace3 = go.Bar(

    x=x,

    y=JP_Perce,

    xaxis='x2', yaxis='y2',

  

    marker=dict(

        color='orange',

        line=dict(

            color='rgb(8,48,107)',

            width=3),

        ),

    opacity=0.75,

    name = "Japan",

    )

trace4 = go.Bar(

    x=x,

    y=Other_Perce,

    xaxis='x2', yaxis='y2',

    

    marker=dict(

        color='purple',

        line=dict(

            color='rgb(8,48,107)',

            width=3),

        ),

    opacity=0.75,

    name = "Other",)

trace5=go.Table(

  header = dict(

    values = table_data.columns,

    line = dict(color = 'rgb(8,48,107)',width=3),

    fill = dict(color = ["darkslateblue","blue","red", "orange","purple","green"]),

    align = ['left','center'],

    font = dict(color = 'white', size = 12),

     height=30,

  ),

  cells = dict(

    values = [table_data.Genre,round(table_data["North America"]),round(table_data["Europe"]), round(table_data["Japan"]), round(table_data["Other"]),round(table_data["Total"])],

    height=30,

    line = dict(color = 'rgb(8,48,107)',width=3),

    fill = dict(color = ["silver","rgb(158,202,225)","darksalmon", "gold","mediumorchid","yellowgreen"]),

    align = ['left', 'center'],

    font = dict(color = '#506784', size = 12)),

    domain=dict(x=[0.60,1],y=[0,0.95]),

)



data = [trace1, trace2,trace3,trace4,trace5]

layout = go.Layout(barmode='stack',autosize=False,width=1200,height=650,

                legend=dict(x=.58, y=0,orientation="h",font=dict(family='Courier New, monospace',size=11,color='#000'),

                           bgcolor='beige', bordercolor='beige', borderwidth=1),

                title='North America, Europe, Japan and Other Sales Percentage and Amounts According to Genre',

                titlefont=dict(family='Courier New, monospace',size=17,color='black'),

                xaxis2=dict(domain=[0, 0.50],anchor="y2", title='Genre',titlefont=dict(family='Courier New, monospace'),tickfont=dict(family='Courier New, monospace')), yaxis2=dict( domain=[0, 1],anchor='x2',title="Total Percentage",titlefont=dict(family='Courier New, monospace'),tickfont=dict(family='Courier New, monospace')),

                paper_bgcolor='beige',plot_bgcolor='beige',

                annotations=[ dict( text='Sales Percentage According to Region',x=0.08,y=1.02,xref="paper",yref="paper",showarrow=False,font=dict(size=15,family="Courier New, monospace"),bgcolor="lightyellow",borderwidth=5),dict( text='Total Sales(In Millions)',x=0.9,y=1.02,xref="paper",yref="paper",showarrow=False,font=dict(size=15,family='Courier New, monospace'),bgcolor="lightyellow",borderwidth=5)],

              

                  )

fig = go.Figure(data=data, layout=layout)

iplot(fig)
wave_mask= np.array(Image.open("../input/contoller/controller.png"))

stopwords = set(STOPWORDS)

stopwords.update(["II", "III"])

plt.subplots(figsize=(15,15))

wordcloud = WordCloud(mask=wave_mask,background_color="lavenderblush",colormap="hsv" ,contour_width=2, contour_color="black",

                      width=950,stopwords=stopwords,

                          height=950

                         ).generate(" ".join(vgsales.Name))



plt.imshow(wordcloud ,interpolation='bilinear')

plt.axis('off')

plt.savefig('graph.png')



plt.show()

df1000=vgsales.iloc[:1000,:]
df1000["normsales"] = (df1000["Global_Sales"] - np.min(df1000["Global_Sales"]))/(np.max(df1000["Global_Sales"])-np.min(df1000["Global_Sales"]))
df1000.Rank=df1000.Rank.astype("str")

df1000.Global_Sales=df1000.Global_Sales.astype("str")

trace1 = go.Scatter3d(

    y=df1000["Publisher"],

    x=df1000["Year"],

    z=df1000["normsales"],

    text="Name:"+ df1000.Name +","+" Rank:" + df1000.Rank + " Global Sales: " + df1000["Global_Sales"] +" millions",

    mode='markers',

    marker=dict(

        size=df1000['NA_Sales'],

        color = df1000['normsales'],

        colorscale = "Rainbow",

        colorbar = dict(title = 'Global Sales'),

        line=dict(color='rgb(140, 140, 170)'),

       

    )

)



data=[trace1]



layout=go.Layout(height=800, width=800, title='Top 1000 Video Games, Release Years, Publishers and Sales',

            titlefont=dict(color='rgb(20, 24, 54)'),

            scene = dict(xaxis=dict(title='Year',

                                    titlefont=dict(color='rgb(20, 24, 54)')),

                            yaxis=dict(title='Publisher',

                                       titlefont=dict(color='rgb(20, 24, 54)')),

                            zaxis=dict(title='Global Sales',

                                       titlefont=dict(color='rgb(20, 24, 54)')),

                            bgcolor = 'whitesmoke'

                           ))

 

fig=go.Figure(data=data, layout=layout)

iplot(fig)