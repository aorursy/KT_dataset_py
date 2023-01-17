# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.graph_objects as go

from plotly.subplots import make_subplots

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Comment this if the data visualisations doesn't work on your side

%matplotlib inline



plt.style.use('bmh')

df = pd.read_csv("../input/ramen-ratings.csv")

print(df.shape)

df.head(5)
df.info()

df.describe(include="all")
columns = ['Style','Country','Stars','Top Ten']

for col in columns : 

    if df[col].dtypes == "object" : print(col,":" ,df[col].unique().tolist(),"\n")
#df = pd.read_csv("../input/ramen-ratings.csv")

# Store preprocess dataframe to df 1

df1=df.copy()
# Fix spaces in column name

df1.columns = [c.replace(' ', '_') for c in df1.columns]



# Stars - Replace 'Unrated'

df1.Stars = df1['Stars'].replace(to_replace='Unrated',value='-1')



# Stars - Convert data type from object to float

df1.Stars = df1.Stars.astype(float)



# Top Ten - Replace '\n'

df1.Top_Ten = df1.Top_Ten.replace(to_replace='\n',value=np.nan)



# Top Ten - Slice column into 2 and drop the column 

df1[['Topten_Year','Topten_Rank']] = df1['Top_Ten'].str.split('#', expand=True)

df1 = df1.drop('Top_Ten', axis=1)



# Top Ten - Set rank as float 

df1.Topten_Rank = df1.Topten_Rank.astype(float)
df1[df1.Topten_Rank.notnull()].head(5)

df1.info()
df1.head(10)
# Preparing data 

df1_brand_histograms= df1.groupby(['Brand']).agg({'Variety':pd.Series.nunique,'Stars':'mean'})



# Generate plot and styling 1 out of 2

fig = go.Figure(data=go.Histogram(x=df1_brand_histograms["Variety"],

                 marker=dict(color="rgba(20,189,204,0.2)",line=dict(color='#14BDCC', width=1)),

                 hoverinfo="x+y+z+text",

                 name = "Variety",

                 hoverlabel = dict(bgcolor="rgba(20,189,204,1)",bordercolor="rgba(20,189,204,0.5)",

                       font=dict(family="Arial",size=10,color='rgba(255,255,255,1)'))

                ))



# Update xaxis properties

fig.update_xaxes(title=dict(font=dict(size=12, color='#1A1817')),

                 ticks="outside", tickwidth=0.5, tickcolor='#F2E9E1', ticklen=10,hoverformat=",.1f",

                 showline=True, linewidth=2, linecolor='#F2E9E1',

                 showgrid=False, gridwidth=1, gridcolor='#F2E9E1',

                 zeroline=True, zerolinewidth=1, zerolinecolor='crimson')

fig.update_xaxes(title_text="<b>VARIETY</b> (total variety)")



# Update yaxis properties

fig.update_yaxes(title= dict(font=dict(size=12, color='#1A1817'),),ticks="outside", tickwidth=0.5, tickcolor='#F2E9E1', ticklen=10, hoverformat=",.1f",

                 showline=True, linewidth=2, linecolor='#F2E9E1',

                 showgrid=True, gridwidth=1, gridcolor='#F2E9E1',

                 zeroline=True, zerolinewidth=1, zerolinecolor='crimson',)

fig.update_yaxes(title_text="<b>BRAND</b> (total)")



fig.update_layout(    

    font=dict(family="Raleway, sans-serif", size=12, color='#98928E'),

    plot_bgcolor="#fffaf7",

    showlegend=False,

    paper_bgcolor = "#fffaf7",

    height = 500,

    annotations=[go.layout.Annotation(text="<b>Most brand has at most 4 variety</b>",x=-0.05,y=1.18,xref="paper",yref="paper",showarrow=False,

                                      xanchor="left",yanchor="top",align="left",font=dict(family=" Raleway,sans-serif", size=16, color='#1A1817')),

                 go.layout.Annotation(text="Total brands by total variety of ramen",x=-0.05,y=1.12,xref="paper",yref="paper",showarrow=False,

                                      xanchor="left",yanchor="top",align="left",font=dict(family='Raleway,sans-serif', size=14, color='#98928E'))]

)

fig.show()
# Generate plot and styling 2 out of 2

fig = go.Figure(data=go.Histogram(x=df1_brand_histograms["Stars"],

                 cumulative_enabled=True,

                marker=dict(color="rgba(20,189,204,0.2)",line=dict(color='#14BDCC', width=1)),

                 showlegend=False,

                 hoverinfo="x+y+z+text",

                 hoverlabel = dict(bgcolor="rgba(20,189,204,1)",bordercolor="rgba(20,189,204,0.5)",

                       font=dict(family="Arial",size=10,color='rgba(255,255,255,1)'))

                ))



# Update xaxis properties

fig.update_xaxes(title=dict(font=dict(size=12, color='#1A1817')),

                 ticks="outside", tickwidth=0.5, tickcolor='#F2E9E1', ticklen=10,hoverformat=",.1f",

                 showline=True, linewidth=2, linecolor='#F2E9E1',

                 showgrid=False, gridwidth=1, gridcolor='#F2E9E1',

                 zeroline=True, zerolinewidth=1, zerolinecolor='crimson')

fig.update_xaxes(title_text="<b>STARS</b> (average ratings)")



# Update yaxis properties

fig.update_yaxes(title= dict(font=dict(size=12, color='#1A1817'),),ticks="outside", tickwidth=0.5, tickcolor='#F2E9E1', ticklen=10, hoverformat=",.1f",

                 showline=True, linewidth=2, linecolor='#F2E9E1',

                 showgrid=True, gridwidth=1, gridcolor='#F2E9E1',

                 zeroline=True, zerolinewidth=1, zerolinecolor='crimson',)

fig.update_yaxes(title_text="<b>BRAND</b> (total cummulative)")



fig.update_layout(    

    font=dict(family="Raleway, sans-serif", size=12, color='#98928E'),

    plot_bgcolor="#fffaf7",

    showlegend=False,

    paper_bgcolor = "#fffaf7",

    height = 500,

    annotations=[go.layout.Annotation(text="<b>Most brand rated less than 4</b>",x=-0.05,y=1.18,xref="paper",yref="paper",showarrow=False,

                                      xanchor="left",yanchor="top",align="left",font=dict(family=" Raleway,sans-serif", size=16, color='#1A1817')),

                 go.layout.Annotation(text="Total brands (cummulative) by average ratings",x=-0.05,y=1.12,xref="paper",yref="paper",showarrow=False,

                                      xanchor="left",yanchor="top",align="left",font=dict(family='Raleway,sans-serif', size=14, color='#98928E')),]

)

fig.show()
# Preparing data

df2 = df1.groupby('Country').agg({'Variety':pd.Series.nunique,

                                  'Brand':pd.Series.nunique, 

                                  'Stars':'mean',

                                  'Review_#':['mean','sum'],

                                  

                                 })

df2.columns =df2.columns.get_level_values(0)+"_"+df2.columns.get_level_values(1)

df3 = df1.groupby(['Country','Brand']).agg({

                                  'Stars':'mean',

                                  'Review_#':'mean'

                                 })



# Generate plot and styling

colorscale1 = [

[0,'rgba(0,204,204,.5)'],

[0.5,'rgba(0,102,102,.5)'], 

[1,'rgba(0,25,51,.5)']]



colorscale1_line = [

[0,'rgba(0,204,204,1)'],

[0.5,'rgba(0,102,102,1)'], 

[1,'rgba(0,25,51,1)']]



fig = go.Figure(data=  go.Scatter(

        x=df2["Variety_nunique"],

        y=df2["Brand_nunique"],

        name ="Country",

        mode="markers",

        text = df2.index,

        marker=dict(

            color="rgba(20,189,204,0.2)",

            size=15,

            line=dict(color='#14BDCC', width=1)),

        hoverinfo="x+y+z+text",

        hoverlabel = dict(

                       bgcolor="rgba(20,189,204,1)",

                       bordercolor="rgba(20,189,204,0.5)",

                       font=dict(

                           family="Arial", 

                           size=10, 

                           color='rgba(255,255,255,1)'))))





# Update xaxis properties

fig.update_xaxes(title=dict(font=dict(size=12, color='#1A1817')),

                 ticks="outside", tickwidth=0.5, tickcolor='#F2E9E1', ticklen=10,hoverformat=",.1f",

                 showline=True, linewidth=2, linecolor='#F2E9E1',

                 showgrid=True, gridwidth=1, gridcolor='#F2E9E1',

                 zeroline=True, zerolinewidth=1, zerolinecolor='crimson')



fig.update_xaxes(title_text="<b>VARIETY</b> (total)")



# Update yaxis properties

fig.update_yaxes(title= dict(font=dict(size=12, color='#1A1817'),),ticks="outside", tickwidth=0.5, tickcolor='#F2E9E1', ticklen=10, hoverformat=",.1f",

                 showline=True, linewidth=2, linecolor='#F2E9E1',

                 showgrid=True, gridwidth=1, gridcolor='#F2E9E1',

                 zeroline=True, zerolinewidth=1, zerolinecolor='crimson',)



fig.update_yaxes(title_text="<b>BRAND</b> (total)")



fig.update_layout(    

    font=dict(family="Raleway, sans-serif", size=12, color='#98928E'),

    plot_bgcolor="#fffaf7",

    showlegend=True,

    paper_bgcolor = "#fffaf7",

    height = 500,

    annotations=[go.layout.Annotation(text="<b>Brand and variety are proportional with 1:5 ratio</b>",x=-0.05,y=1.18,xref="paper",yref="paper",showarrow=False,

                                      xanchor="left",yanchor="top",align="left",font=dict(family=" Raleway,sans-serif", size=16, color='#1A1817')),

                 go.layout.Annotation(text="Total brands and ramen varieties by country",x=-0.05,y=1.12,xref="paper",yref="paper",showarrow=False,

                                      xanchor="left",yanchor="top",align="left",font=dict(family='Raleway,sans-serif', size=14, color='#98928E')),]

)

fig.show()
# Preparing data

df2 = df1.groupby('Country').agg({'Variety':pd.Series.nunique,

                                  'Brand':pd.Series.nunique, 

                                  'Stars':'mean',

                                  'Review_#':['mean','sum'],

                                  

                                 })

df2.columns =df2.columns.get_level_values(0)+"_"+df2.columns.get_level_values(1)

df3 = df1.groupby(['Country','Brand']).agg({

                                  'Stars':'mean',

                                  'Review_#':'mean'

                                 })



# Generate plot and styling

colorscale1 = [

[0,'rgba(0,204,204,.5)'],

[0.5,'rgba(0,102,102,.5)'], 

[1,'rgba(0,25,51,.5)']]



colorscale1_line = [

[0,'rgba(0,204,204,1)'],

[0.5,'rgba(0,102,102,1)'], 

[1,'rgba(0,25,51,1)']]



fig = go.Figure(data=  go.Scatter(x=df2["Review_#_mean"],

               y=df2["Stars_mean"],

               mode="markers",

               name="Country",

               hoverinfo="x+y+z+text",

               text = df2.index,

               marker=dict(color=(df2["Review_#_sum"]), size=15,

                           colorbar=dict(title= dict(text="<b>TOTAL<br>REVIEWS</b>",font=dict(size=12, color='#1A1817'),),x=1.02, y=0.95,yanchor="top", len=1, ),colorscale=colorscale1,

                           line=dict(color=df2["Review_#_sum"],width=1,colorscale=colorscale1_line)), 

               

              ))







# Update xaxis properties

fig.update_xaxes(title=dict(font=dict(size=12, color='#1A1817')),

                 ticks="outside", tickwidth=0.5, tickcolor='#F2E9E1', ticklen=10,hoverformat=",.1f",

                 showline=True, linewidth=2, linecolor='#F2E9E1',

                 showgrid=True, gridwidth=1, gridcolor='#F2E9E1',

                 zeroline=True, zerolinewidth=1, zerolinecolor='crimson')

fig.update_xaxes(title_text="<b>REVIEWS</b> (average per ramen)")



# Update yaxis properties

fig.update_yaxes(title= dict(font=dict(size=12, color='#1A1817'),),ticks="outside", tickwidth=0.5, tickcolor='#F2E9E1', ticklen=10, hoverformat=",.1f",

                 showline=True, linewidth=2, linecolor='#F2E9E1',

                 showgrid=True, gridwidth=1, gridcolor='#F2E9E1',

                 zeroline=True, zerolinewidth=1, zerolinecolor='crimson',)

fig.update_yaxes(title_text="<b>STARS</b> (rating)")



fig.update_layout(    

    font=dict(family="Raleway, sans-serif", size=12, color='#98928E'),

    plot_bgcolor="#fffaf7",

    showlegend=True,

    paper_bgcolor = "#fffaf7",

    height = 500,

    annotations=[go.layout.Annotation(text="<b>Most countries with high total reviews, generously rate ramens</b>",x=-0.05,y=1.18,xref="paper",yref="paper",showarrow=False,

                                      xanchor="left",yanchor="top",align="left",font=dict(family=" Raleway,sans-serif", size=16, color='#1A1817')),

                 go.layout.Annotation(text="Countries average stars and reviews color coded by total reviews",x=-0.05,y=1.12,xref="paper",yref="paper",showarrow=False,

                                      xanchor="left",yanchor="top",align="left",font=dict(family='Raleway,sans-serif', size=14, color='#98928E')),]

)

fig.show()
# Preparing data 

top_brand = df1.groupby('Brand').count()['Variety'][df1.groupby('Brand').count()['Variety']>20].index.values # Listing brands with 20+ variety

df1_top_brand =  df1.loc[df1['Brand'].isin(top_brand)] # Filtering the brand

df1_brand_heatmaps = df1_top_brand.groupby(['Country','Brand']).agg({'Stars':'mean','Review_#':'mean'})



# Generate plot and styling

fig = go.Figure(data=go.Heatmap(

        x=df1_brand_heatmaps.index.get_level_values(0),

        y=df1_brand_heatmaps.index.get_level_values(1),

        z=df1_brand_heatmaps['Stars'],

    colorbar=dict(title= dict(text='<b>RATINGS</b><br>(average)',font=dict(size=12, color='#1A1817'),) )



))

# Update xaxis properties

fig.update_xaxes(title=dict(text="<b>COUNTRY</b>",font=dict(size=12, color='#1A1817')),

                 ticks="outside", tickwidth=0.5, tickcolor='#F2E9E1', ticklen=10,hoverformat=",.1f",

                 showline=True, linewidth=2, linecolor='#F2E9E1',

                 showgrid=True, gridwidth=1, gridcolor='#F2E9E1',

                 zeroline=True, zerolinewidth=1, zerolinecolor='crimson')





# Update yaxis properties

fig.update_yaxes(title= dict(text = "<b>BRAND</b>",font=dict(size=12, color='#1A1817'),),ticks="outside", tickwidth=0.5, tickcolor='#F2E9E1', ticklen=10, hoverformat=",.1f",

                 showline=True, linewidth=2, linecolor='#F2E9E1',

                 showgrid=True, gridwidth=1, gridcolor='#F2E9E1',

                 zeroline=True, zerolinewidth=1, zerolinecolor='crimson',)



fig.update_layout(    

    xaxis_nticks=len(df1_brand_heatmaps.index.get_level_values(0)),

    yaxis_nticks=len(df1_brand_heatmaps.index.get_level_values(1)),

    font=dict(family="raleway, sans-serif", size=10, color='#98928E'),

    plot_bgcolor="#fffaf7",

    showlegend=True,

    paper_bgcolor = "#fffaf7",

    height = 500,

    annotations=[go.layout.Annotation(text="<b>Nissin has the most presence in contries</b>",x=-0.05,y=1.18,xref="paper",yref="paper",showarrow=False,

                                      xanchor="left",yanchor="top",align="left",font=dict(family=" Raleway,sans-serif", size=16, color='#1A1817')),

                 go.layout.Annotation(text="Average rating by brand and country",x=-0.05,y=1.12,xref="paper",yref="paper",showarrow=False,

                                      xanchor="left",yanchor="top",align="left",font=dict(family='Raleway,sans-serif', size=14, color='#98928E')),]

                 )



fig.show()