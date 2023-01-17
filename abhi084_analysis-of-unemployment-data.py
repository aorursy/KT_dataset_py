# Import the relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
df=pd.read_csv('../input/API_ILO_country_YU.csv')
df.head(5)
df.shape
df.columns
non_country_list=['Arab World','Central Europe and the Baltics','Caribbean small states','East Asia & Pacific (excluding high income)',
                 'Early-demographic dividend', 'East Asia & Pacific','Europe & Central Asia (excluding high income)',
                 'Europe & Central Asia','Euro area','European Union','Fragile and conflict affected situations','High income',
                 'Heavily indebted poor countries (HIPC)','IBRD only', 'IDA & IBRD total', 'IDA total','IDA blend','IDA only',
                 'Latin America & Caribbean (excluding high income)','Latin America & Caribbean','Least developed countries: UN classification', 
                 'Low income','Lower middle income', 'Low & middle income','Late-demographic dividend','Middle East & North Africa',
                 'Middle income','Middle East & North Africa (excluding high income)','North America','OECD members','Other small states',
                 'Pre-demographic dividend','Post-demographic dividend','South Asia','Sub-Saharan Africa (excluding high income)',
                 'Sub-Saharan Africa','Small states','East Asia & Pacific (IDA & IBRD countries)',
                 'Europe & Central Asia (IDA & IBRD countries)','Latin America & the Caribbean (IDA & IBRD countries)',
                 'Middle East & North Africa (IDA & IBRD countries)','South Asia (IDA & IBRD)',
                 'Sub-Saharan Africa (IDA & IBRD countries)','Upper middle income','World']
df_non_country=df[df['Country Name'].isin(non_country_list)]
df_non_country.head()
df_non_country.shape
index=df_non_country.index
df_country=df.drop(index)
df_country.head()
df_country.shape
x_data = ['2010', '2011','2012', '2013','2014']

y0 = df_country['2010']
y1 = df_country['2011']
y2 = df_country['2012']
y3 = df_country['2013']
y4 = df_country['2014']

y_data = [y0,y1,y2,y3,y4]

colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
          'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)']

traces = []

for xd, yd, color in zip(x_data, y_data, colors):
        traces.append(go.Box(
            y=yd,
            name=xd,
            boxpoints='all',
            whiskerwidth=0.2,
            fillcolor=color,
            marker=dict(
                size=2,
            ),
            boxmean=True,    
            line=dict(width=1),
        ))

layout = go.Layout(
    title='Distribution of Unemployment Data',
    xaxis=dict(
        title='Year'
    ),
    yaxis=dict(
        title='Unemployment Rate (%)',
        autorange=True,
        showgrid=True,
        zeroline=False,
        dtick=5,
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
#        zerolinecolor='rgb(255, 255, 255)',
#        zerolinewidth=2,
    ),
    margin=dict(
        l=40,
        r=30,
        b=80,
        t=100,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
    showlegend=False
)

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)
l=[]
trace0= go.Scatter(
        y= df_country['2010'],
        mode= 'markers',
        name='Unemployment (%)',
        marker= dict(size= df_country['2010'].values,
                    line= dict(width=1),
                    color= df_country['2010'].values,
                    opacity= 0.7,
                    colorscale='Portland',
                    showscale=True),
        text= df_country['Country Name'].values) # The hover text goes here... 
l.append(trace0);

layout= go.Layout(
    title= 'Scatter plot of unemployment rates in 2010',
    hovermode= 'closest',
    xaxis= dict(
#        title= 'Pop',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Unemployment Rate (%)',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False,
)
fig= go.Figure(data=l, layout=layout)
py.iplot(fig)

# Pre-defined color scales - 'pairs' | 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' | 'Jet' | 
# 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'

# Chose Portland because it seems to be the best colorscale
l1=[]
trace1= go.Scatter(
        y= df_country['2011'],
        mode= 'markers',
        name='Unemployment (%)',
        marker= dict(size= df_country['2011'].values,
                    line= dict(width=1),
                    color= df_country['2011'].values,
                    opacity= 0.7,
                    colorscale='Portland',
                    showscale=True),
        text= df_country['Country Name'].values) # The hover text goes here... 
l1.append(trace1);

layout= go.Layout(
    title= 'Scatter plot of unemployment rates in 2011',
    hovermode= 'closest',
    xaxis= dict(
#        title= 'Pop',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Unemployment Rate (%)',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig= go.Figure(data=l1, layout=layout)
py.iplot(fig,filename='scatter_plot2011')

# Pre-defined color scales - 'pairs' | 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' | 'Jet' | 
# 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'

# Chose Portland because it seems to be the best colorscale
l2=[]
trace2= go.Scatter(
        y= df_country['2012'],
        mode= 'markers',
        name='Unemployment (%)',
        marker= dict(size= df_country['2012'].values,
                    line= dict(width=1),
                    color= df_country['2012'].values,
                    opacity= 0.7,
                    colorscale='Portland',
                    showscale=True),
        text= df_country['Country Name'].values) # The hover text goes here... 
l2.append(trace2);

layout= go.Layout(
    title= 'Scatter plot of unemployment rates in 2012',
    hovermode= 'closest',
    xaxis= dict(
#        title= 'Pop',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Unemployment Rate (%)',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig= go.Figure(data=l2, layout=layout)
py.iplot(fig,filename='scatter_plot2012')

# Pre-defined color scales - 'pairs' | 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' | 'Jet' | 
# 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'

# Chose Portland because it seems to be the best colorscale
l3=[]
trace3= go.Scatter(
        y= df_country['2013'],
        mode= 'markers',
        name='Unemployment (%)',
        marker= dict(size= df_country['2013'].values,
                    line= dict(width=1),
                    color= df_country['2013'].values,
                    opacity= 0.7,
                    colorscale='Portland',
                    showscale=True),
        text= df_country['Country Name'].values) # The hover text goes here... 
l3.append(trace3);

layout= go.Layout(
    title= 'Scatter plot of unemployment rates in 2013',
    hovermode= 'closest',
    xaxis= dict(
#        title= 'Pop',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Unemployment Rate (%)',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig= go.Figure(data=l3, layout=layout)
py.iplot(fig,filename='scatter_plot2013')

# Pre-defined color scales - 'pairs' | 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' | 'Jet' | 
# 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'

# Chose Portland because it seems to be the best colorscale
l4=[]
trace4= go.Scatter(
        y= df_country['2014'],
        mode= 'markers',
        name='Unemployment (%)',
        marker= dict(size= df_country['2014'].values,
                    line= dict(width=1),
                    color= df_country['2014'].values,
                    opacity= 0.7,
                    colorscale='Portland',
                    showscale=True),
        text= df_country['Country Name'].values) # The hover text goes here... 
l4.append(trace4);

layout= go.Layout(
    title= 'Scatter plot of unemployment rates in 2014',
    hovermode= 'closest',
    xaxis= dict(
#        title= 'Pop',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Unemployment Rate (%)',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig= go.Figure(data=l4, layout=layout)
py.iplot(fig,filename='scatter_plot2014')

# Pre-defined color scales - 'pairs' | 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' | 'Jet' | 
# 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'

# Chose Portland because it seems to be the best colorscale

df_country['2014-2012 change']=df_country['2014']-df_country['2012']
df_country['2012-2010 change']=df_country['2012']-df_country['2010']
df_country.head()
# Tried with Plotly now going with seaborn
twoyearchange201412_bar, countries_bar1 = (list(x) for x in zip(*sorted(zip(df_country['2014-2012 change'], df_country['Country Name']), 
                                                             reverse = True)))

twoyearchange201210_bar, countries_bar2 = (list(x) for x in zip(*sorted(zip(df_country['2012-2010 change'], df_country['Country Name']), 
                                                             reverse = True)))

# Another direct way of sorting according to values is creating distinct sorted dataframes as in below commented ways and then
# passing their values directly as in below mentioned code to achieve the same effect as by above mentioned method.

# df_country_sorted=df_country.sort(columns='2014-2012 change',ascending=False)
# df_country_sorted.head()


sns.set(font_scale=1) 
fig, axes = plt.subplots(1,2,figsize=(20, 50))
colorspal = sns.color_palette('husl', len(df_country['2014']))
sns.barplot(twoyearchange201412_bar, countries_bar1, palette = colorspal,ax=axes[0])
sns.barplot(twoyearchange201210_bar, countries_bar2, palette = colorspal,ax=axes[1])
axes[0].set(xlabel='%age change in Youth Unemployment Rates', title='Net %age change in Youth Unemployment Rates between 2012-2014')
axes[1].set(xlabel='%age change in Youth Unemployment Rates', title='Net %age change in Youth Unemployment Rates between 2010-2012')
fig.savefig('output.png')
df_country['2014-2010 change']=df_country['2014']-df_country['2010']
def top_successful_1(df,n=10,column='2014-2010 change'):
    return df.sort_index(by=column,ascending=True).head(n)
def top_failure_1(df,n=10,column='2014-2010 change'):
    return df.sort_index(by=column,ascending=False).head(n)
top15=top_successful_1(df_country,n=15)
bottom15=top_failure_1(df_country,n=15)
sns.set(font_scale=1.4) 
fig, axes = plt.subplots(1,2,figsize=(25, 20))
colorspal = sns.color_palette('husl', len(top15['2014']))
sns.barplot(top15['2014-2010 change'], top15['Country Name'], palette = colorspal,ax=axes[0])
sns.barplot(bottom15['2014-2010 change'], bottom15['Country Name'], palette = colorspal,ax=axes[1])
axes[0].set(xlabel='%age change in Youth Unemployment Rates', title='Top 15 Performers in Controlling Unemployment between 2010-14')
axes[1].set(xlabel='%age change in Youth Unemployment Rates', title='Bottom 15 Performers in Controlling Unemployment between 2010-14')
fig.savefig('output1.png')
# Plotting 2010 World Unemployment Data Geographically
data = [ dict(
        type = 'choropleth',
        locations = df_country['Country Code'],
        z = df_country['2010'],
        text = df_country['Country Name'],
        colorscale = 'Reds',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Unemployment (%)'),
      ) ]

layout = dict(
    title = 'Unemployment around the globe in 2010',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        showocean = True,
        #oceancolor = 'rgb(0,255,255)',
        oceancolor = 'rgb(222,243,246)',
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False,filename='world2010')

# Colorscale Sets the colorscale and only has an effect if `marker.color` is set to a numerical array. 
# Alternatively, `colorscale` may be a palette name string of the following list: Greys, YlGnBu, Greens, YlOrRd, 
# Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis
# Plotting 2014 World Unemployment Data Geographically
data = [ dict(
        type = 'choropleth',
        locations = df_country['Country Code'],
        z = df_country['2014'],
        text = df_country['Country Name'],
        colorscale = 'Reds',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Unemployment (%)'),
      ) ]

layout = dict(
    title = 'Unemployment around the globe in 2014',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        showocean = True,
        #oceancolor = 'rgb(0,255,255)',
        oceancolor = 'rgb(222,243,246)',
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout)
py.iplot( fig, validate=False,filename='world2014')
# Colorscale Sets the colorscale and only has an effect if `marker.color` is set to a numerical array. 
# Alternatively, `colorscale` may be a palette name string of the following list: Greys, YlGnBu, Greens, YlOrRd, 
# Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis
# Plotting 2014 World Unemployment Data Geographically
data = [ dict(
        type = 'choropleth',
        locations = df_country['Country Code'],
        z = df_country['2014-2010 change'],
        text = df_country['Country Name'],
        colorscale = 'RdBu',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Unemployment (%)'),
      ) ]

layout = dict(
    title = 'Net Change in Unemployment around the globe over the 5 year period (2010-14)',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        showocean = True,
        #oceancolor = 'rgb(0,255,255)',
        oceancolor = 'rgb(222,243,246)',
        projection = dict(
            type = 'Mercator'
         )
    )    
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False,filename='WorldChange')

# Colorscale Sets the colorscale and only has an effect if `marker.color` is set to a numerical array. 
# Alternatively, `colorscale` may be a palette name string of the following list: Greys, YlGnBu, Greens, YlOrRd, 
# Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis
supranational_groups=['Arab World','Caribbean small states','East Asia & Pacific','European Union','Latin America & Caribbean',
                      'Middle East & North Africa','North America','OECD members','Other small states','South Asia',
                      'Sub-Saharan Africa','World']
df_supranational=df_non_country[df_non_country['Country Name'].isin(supranational_groups)]
df_supranational=df_supranational[['Country Name','2010','2011','2012','2013','2014']]
df_supranational.head()
df_supranational=df_supranational.set_index('Country Name')
df_supranational=(df_supranational.T).copy()
df_supranational.head()
income_groups=['High income','Heavily indebted poor countries (HIPC)','Least developed countries: UN classification', 
                 'Low income','Lower middle income', 'Low & middle income','Middle income','Upper middle income','World']
df_income=df_non_country[df_non_country['Country Name'].isin(income_groups)]
df_income=df_income[['Country Name','2010','2011','2012','2013','2014']]

df_income=df_income.set_index('Country Name')
df_income=(df_income.T).copy()
df_income
# Supranational Group Unemployment Comparison

supranational_groups=['Arab World','Caribbean small states','East Asia & Pacific','European Union','Latin America & Caribbean',
                      'Middle East & North Africa','North America','OECD members','Other small states','South Asia',
                      'Sub-Saharan Africa','World']

years=df_supranational.index

traces=[]

for i in range(len(supranational_groups)):
    traces.append(go.Scatter(
                  x=years,
                  y=df_supranational.iloc[:,i],
                  name=supranational_groups[i],
                  mode='lines+markers',
                  line = dict(
                              width = 3,
                              dash = 'dashdot')
        ))

layout = go.Layout(
    title='Unemployment Over the Years in different Regions of the World',
    yaxis=dict(title='Unemployment Rate (%)',
               zeroline=True,
               showline=True,
               showgrid=False,
               showticklabels=True,
               linecolor='rgb(0,0,0)',
               linewidth=2,
               tickmode='auto',
               tickwidth=2,
               ticklen=5,
               nticks=8,
               tickfont=dict(
                            family='Arial',
                            size=12,
                            color='rgb(82, 82, 82)',
                            ),
               ticks='outside'),
    
    xaxis=dict(title='Years',
               showline=True,
               showgrid=False,
               showticklabels=True,
               linecolor='rgb(0,0,0)',
               linewidth=2,
               autotick=False,
               tickwidth=2,
               ticklen=5,
               tickfont=dict(
                            family='Arial',
                            size=12,
                            color='rgb(82, 82, 82)',
                            ),
               ticks='outside',
               tickmode='array',
               tickvals=['2009','2010', '2011', '2012', '2013', '2014','2015'])
)

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)
# Income Group Unemployment Comparison

income_groups=['High income','Heavily indebted poor countries (HIPC)','Least developed countries: UN classification', 
                 'Low income','Lower middle income', 'Low & middle income','Middle income','Upper middle income','World']

years_income=df_income.index

traces=[]

for i in range(len(income_groups)):
    traces.append(go.Scatter(
                  x=years_income,
                  y=df_income.iloc[:,i],
                  name=income_groups[i],
                  mode='lines+markers',
                  line = dict(
                              width = 3,
                              dash = 'dashdot')
        ))

layout = go.Layout(
    title='Unemployment Over the Years among Various Income Groups',
    yaxis=dict(title='Unemployment Rate (%)',
               zeroline=True,
               showline=True,
               showgrid=False,
               showticklabels=True,
               linecolor='rgb(0,0,0)',
               linewidth=2,
               tickmode='auto',
               tickwidth=2,
               ticklen=5,
               nticks=8,
               tickfont=dict(
                            family='Arial',
                            size=12,
                            color='rgb(82, 82, 82)',
                            ),
               ticks='outside'),
    
    xaxis=dict(title='Years',
               showline=True,
               showgrid=False,
               showticklabels=True,
               linecolor='rgb(0,0,0)',
               linewidth=2,
               autotick=False,
               tickwidth=2,
               ticklen=5,
               tickfont=dict(
                            family='Arial',
                            size=12,
                            color='rgb(82, 82, 82)',
                            ),
               ticks='outside',
               tickmode='array',
               tickvals=['2009','2010', '2011', '2012', '2013', '2014','2015'])
)

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)