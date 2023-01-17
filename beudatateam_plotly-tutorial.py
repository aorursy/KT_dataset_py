import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
# plotly
# import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# word cloud library
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt
# Input data files are available in the "
import seaborn as sns
import missingno as msno
from collections import Counter
import plotly.figure_factory as ff
from plotly import tools
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# import warnings library
import warnings        
# ignore filters
warnings.filterwarnings("ignore") # if there is a warning after some codes, this will avoid us to see them.
plt.style.use('ggplot') # style of plots. ggplot is one of the most used style, I also like it.
# Any results you write to the current directory are saved as output.
dataset=pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")
dataset.head()
dataset.info()
dataset.dropna(axis=0,inplace=True)
# Drop nan values
dataset.drop(dataset[dataset["Type"] == "0"].index,inplace = True)
dataset.drop(dataset[dataset["Type"].isna()].index,inplace = True)
dataset.drop(dataset[dataset["Category"] == "1.9"].index,inplace = True)
dataset["Content Rating"].dropna(inplace = True)
# Clean Price column
dataset["Price"] = dataset["Price"].apply(lambda x: float(x.replace("$",'')))
# Convert reviews to int
dataset["Reviews"] = dataset["Reviews"].apply(lambda x: int(x))

# Clean Installs column
dataset['Installs'] = dataset['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
dataset['Installs'] = dataset['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
dataset['Installs'] = dataset['Installs'].apply(lambda x: int(x))
# Make another dataset for Size analysis as dropping rows will reduce the amount of dataset
def kb_to_mb(row):
    
    if "k" in str(row):
        row = row.replace('k','')
        size = float(row)/1000
    else:
        row = row.replace("M",'').replace(",",'').replace("+",'')
        size = float(row)
    return size
ds_clear_size = dataset[dataset["Size"] != 'Varies with device']
ds_clear_size["Size"] = ds_clear_size["Size"].apply(kb_to_mb)
ds_clear_size.head()
df=ds_clear_size.iloc[:100].sort_values("Size").reset_index()
trace1=go.Scatter (
       x=df.index,
       y=df.Rating,
       mode="lines",
       name="Rating",
       marker=dict(color='rgba(16,112,2,0.8)'),
       text=df.App
     
    )
trace2=go.Scatter (
       x=df.index,
       y=df.Size,
       mode="lines+markers",
       name="Size",
       marker=dict(color='rgba(80,26,80,0.8)'),
       text=df.App) 
data=[trace1,trace2]
layout=dict(title={
        'text': "Size vs Rating",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},xaxis=dict(ticklen=5,zeroline=False))    
fig=dict(layout=layout,data=data)
iplot(fig)    

df_art=ds_clear_size[ds_clear_size["Category"]=="ART_AND_DESIGN"].sort_values("Size",ascending=False).reset_index().iloc[:50]
df_family=ds_clear_size[ds_clear_size["Category"]=="FAMILY"].sort_values("Size",ascending=False).reset_index().iloc[:50]
df_ls=ds_clear_size[ds_clear_size["Category"]=="LIFESTYLE"].sort_values("Size",ascending=False).reset_index().iloc[:50]
df_art.drop(columns="index",inplace=True)
df_family.drop(columns="index",inplace=True)
df_ls.drop(columns="index",inplace=True)
trace1=go.Scatter(x=df_art.index,
                  y=df_art.Size,
                  mode="markers",
                  name="ART_DESIGN",
                  marker=dict(color='rgba(255,128,255,0.8)'),
                  text=df_art.App)

trace2=go.Scatter(x=df_family.index,
                  y=df_family.Size,
                  mode="markers",
                  name="FAMILY",
                  marker=dict(color='rgba(255,128,2,0.8)'),
                  text=df_family.App)
trace3=go.Scatter(x=df_ls.index,
                  y=df_ls.Size,
                  mode="markers",
                  name="LIFESTYLE",
                  marker=dict(color='rgba(0,255,2000,0.8)'),
                  text=df_ls.App)

data=[trace1,trace2,trace3]
layout=dict(title={
        'text': "Top 50 Apps Sizes in FAMILY and ART Category",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},xaxis=dict(ticklen=5,zeroline=False))
fig=dict(layout=layout,data=data) 
iplot(fig)



df_art=ds_clear_size[ds_clear_size["Category"]=="ART_AND_DESIGN"].sort_values("Size",ascending=False).reset_index().iloc[:3]
df_art.drop(columns="index",inplace=True)
trace1=go.Bar(x=df_art.App,
              y=df_art.Size,
              name="Size",
              marker=dict(color = 'rgba(255, 174, 255, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
              text=df_art.App)

trace2=go.Bar(x=df_art.App,
              y=df_art.Rating,
              name="Rating",
              marker=dict(color = 'rgba(255, 200, 2, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
              text=df_art.App)
data=[trace1,trace2]

layout=dict(title={
        'text': "Top 3 App with Size vs Rating",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},xaxis=dict(ticklen=5,zeroline=False))

fig=dict(layout=layout,data=data) 
iplot(fig) 
df_art=ds_clear_size[ds_clear_size["Category"]=="ART_AND_DESIGN"].sort_values("Reviews",ascending=True).reset_index().iloc[2:9]
df_art.drop(columns="index",inplace=True)
y_size=[each for each in df_art.Size]
y_reviews=[each for each in df_art.Reviews]
x_app=[each for each in df_art.App]

trace1=go.Bar(x=y_reviews,
              y=x_app,
              name="Reviews",
              marker=dict(color='rgba(171, 50, 96, 0.6)',line=dict(color='rgba(171, 50, 96, 1.0)',width=1)),
              orientation='h',
              )
trace2=go.Scatter(x=y_size,
              y=x_app,
              name="Size",
              mode="lines+markers",
              line=dict(color='rgb(63, 72, 204)'),
           
              )
layout=dict(title="Reviews and Size",
            yaxis1=dict(showticklabels=True,domain=[0,0.85]),
            yaxis2=dict(showticklabels=False,showline=True,linecolor='rgba(102, 102, 102, 0.8)',linewidth=2,domain=[0, 0.85]),
            xaxis1=dict(showline=True,zeroline=False,showticklabels=True,showgrid=True,domain=[0,0.45]),
            xaxis2=dict(showline=True,zeroline=False,showticklabels=True,domain=[0.47,1],showgrid=True,side="top",dtick=25),
            legend=dict(x=0.029,y=1.038, font=dict(size=10)),
            margin=dict(l=200,r=30,b=70,t=70),
            paper_bgcolor='rgba(248,255,248)',
            plot_bgcolor='rgba(248,255,248)'
    )
annotations=[]

y_s = np.round(y_reviews, decimals=2)
y_in = np.rint(y_size)

for ydn,yd,xd in zip(y_s,y_in,x_app):
    annotations.append(dict(xref='x2',yref='y2',x=yd-4,y=xd,text=str(yd),font=dict(family='Arial', size=12,color='rgb(63, 72, 204)'),showarrow=False))
    annotations.append(dict(xref='x1', yref='y1', y=xd, x=ydn + 3,text=str(ydn),font=dict(family='Arial', size=12,color='rgb(171, 50, 96)'),showarrow=False))
layout['annotations'] = annotations

fig=tools.make_subplots(rows=1,cols=2,specs=[[{},{}]],shared_xaxes=False,shared_yaxes=False)

fig.append_trace(trace1,1,1)
fig.append_trace(trace2, 1, 2)

fig["layout"].update(layout)
iplot(fig)
a=ds_clear_size["Genres"].value_counts().iloc[:7]
pie_list=a.values
labels=a.index
fig={
     "data":[
         {
         "values":pie_list,
         "labels":labels,
         "domain":{"x":[0,0.5]},
         "name":"Number of Genres",
         "hoverinfo":"label+percent+name",
         "hole":.3,
         "type":"pie"},],
     
     "layout":{
          "title":"Genres of Apps",
          "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Number of Genres",
                "x": 0.20,
                "y": 1.08
             },]
     }
    }
iplot(fig)       

df_ls=ds_clear_size[ds_clear_size["Category"]=="LIFESTYLE"].sort_values("Installs",ascending=True).reset_index().iloc[20:50]
df_ls.drop(columns="index",inplace=True)
Installs=df_ls.Installs
review=df_ls.Reviews
data=[
      {
       "x":df_ls.index,
       "y":review,
       "mode":"markers",
       "marker":
           {
               "color":Installs,
               "size":df_ls.Size,
               "showscale":True  
           },
          "text":df_ls.App      
       
       }]
    
iplot(data)    

df_family=ds_clear_size[ds_clear_size["Category"]=="FAMILY"].sort_values("Size",ascending=False).reset_index()
df_family.drop(columns="index",inplace=True)
trace2=go.Histogram(x=df_family.Size,
                    name="Family",
                    opacity=0.8,
                    marker=dict(color='rgba(71,50,196,0.8)'))
data=[trace2]

layout=go.Layout(xaxis=dict(title="App Size"),
                 title="App Size in  Family Category",
                 yaxis=dict(title="Count"),
                 )

fig=dict(data=data,layout=layout)
iplot(fig)

xapp=ds_clear_size[ds_clear_size.Category=="FAMILY"].App
wordcloud=WordCloud(
               background_color="white",
               width=512,
               height=312 ).generate(" ".join(xapp))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
ulimit = np.percentile(ds_clear_size.Installs.values, 99)
llimit = np.percentile(ds_clear_size.Installs.values, 1)
ds_clear_size['Installs tt'] = ds_clear_size['Installs'].copy()
ds_clear_size['Installs tt'].loc[ds_clear_size['Installs']>ulimit] = ulimit
ds_clear_size['Installs tt'].loc[ds_clear_size['Installs']<llimit] = llimit

trace=[]
for name,group in ds_clear_size.groupby("Category"):
    trace.append(
        go.Box(
            x=group["Installs tt"].values,
            name=name
            )
        )
layout=go.Layout(
             title="Installs Distirbution",
             width=800,
             height=2000
    )
fig=go.Figure(data=trace,layout=layout)
iplot(fig)
df_family=ds_clear_size[ds_clear_size.Category=="FAMILY"].loc[:,["Reviews","Size","Installs"]]
df_family["index"]=np.arange(1,len(df_family)+1)

fig=ff.create_scatterplotmatrix(df_family,diag="box",height=700,width=700,index="index",colormap="Portland",colormap_type="cat")
iplot(fig)
df_game=ds_clear_size[ds_clear_size.Category=="GAME"]

trace1=go.Scatter3d(x=df_game.Size,
                    y=df_game.Rating,
                    z=df_game.Reviews,
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=df_game.Installs,
                        ))
data=[trace1]
layout=go.Layout(
          margin=dict(
              t=0,
              b=0,
              r=0,
              l=0
              
              )
    )    
fig=go.Figure(layout=layout,data=data)
iplot(fig)
df_game=ds_clear_size[ds_clear_size["Category"]=="GAME"].sort_values("Size",ascending=True).reset_index().iloc[:50]
df_game.drop(columns="index",inplace=True)
trace1=go.Scatter(x=df_game.index,
                  y=df_game.Reviews,
                  name="Reviews",
                  marker=dict(color='rgba(12,12,140,0.8)'))

trace2=go.Scatter(x=df_game.index,
                  y=df_game.Installs,
                  xaxis="x2",
                  yaxis="y2",
                  name="Installs",
                  marker=dict(color='rgba(12,128,127,0.8)')
                  )
data=[trace1,trace2]
layout=go.Layout(xaxis2=dict(domain=[0.6,0.95],
                            anchor="y2",),
                 yaxis2=dict(domain=[0.6,0.95],
                            anchor="x2",),
                 title="Reviews and Installs in Game Category"
                 )
fig=dict(data=data,layout=layout)
iplot(fig)