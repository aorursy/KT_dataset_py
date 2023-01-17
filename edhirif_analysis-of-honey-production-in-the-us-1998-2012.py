import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.offline as py
import seaborn as sns
%matplotlib inline

df = pd.read_csv('../input/honeyproduction.csv')
df.head()
df_copy = df.copy()
df_copy['total'] = df_copy['totalprod'].groupby(df_copy['year']).transform('sum')
df_copy['total_col'] = df_copy['numcol'].groupby(df_copy['year']).transform('sum')
drop_cols=['state','numcol','yieldpercol','totalprod','stocks','priceperlb','prodvalue']
df_by_year = df_copy.drop(drop_cols,1)
df_by_year = df_by_year.drop_duplicates(keep='first')
sns.regplot(df_by_year['total_col'],df_by_year['total'],scatter_kws={"color": "orange"}, line_kws={"color": "black"})
df_by_year.plot(x='year',y='total',color='orange',title='total production by year')
df_by_year.plot(x='year',y='total_col',color='orange', title='total colonies by year')
df_1998 = df[df.year==1998]
print('number of states in dataset for 1998: ' + str(len(df_1998)))
df_2012 = df[df.year==2012]
print('number of states in dataset for 2012: ' + str(len(df_2012)))
import plotly.offline as py
py.init_notebook_mode(connected=True)

scl = [[0.0, 'rgb(255, 204, 128)'],[0.2, 'rgb(255, 184, 77)'],[0.4, 'rgb(255, 153, 0)'],\
            [0.6, 'rgb(255, 153, 0)'],[0.8, 'rgb(204, 122, 0)'],[1.0, 'rgb(128, 77, 0)']]

labels = df_1998['state']
values = df_1998['yieldpercol']


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = labels,
        z = np.array(values).astype(float),
        locationmode = 'USA-states',
        text = labels,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "yield per colony (in pounds)")
        ) ]

layout = dict(
        title = 'honey yield per colony by state in the US in 1998',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='US_honey_yield_per_bee_colony_1998.html' )
py.init_notebook_mode(connected=True)

scl = [[0.0, 'rgb(255, 204, 128)'],[0.2, 'rgb(255, 184, 77)'],[0.4, 'rgb(255, 153, 0)'],\
            [0.6, 'rgb(255, 153, 0)'],[0.8, 'rgb(204, 122, 0)'],[1.0, 'rgb(128, 77, 0)']]

labels = df_2012['state']
values = df_2012['yieldpercol']


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = labels,
        z = np.array(values).astype(float),
        locationmode = 'USA-states',
        text = labels,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "yield per colony (in pounds)")
        ) ]

layout = dict(
        title = 'honey yield per colony by state in the US in 2012',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='US_honey_yield_per_bee_colony_2012.html' )
df_ca = df[df['state']=='CA']
df_ms = df[df['state']=='MS']
sns.regplot(df_ca['year'],df_ca['yieldpercol'],scatter_kws={"color": "orange"}, line_kws={"color": "black"}).set_title('California')
sns.regplot(df_ms['year'],df_ms['yieldpercol'],scatter_kws={"color": "orange"}, line_kws={"color": "black"}).set_title('Mississippi')
print(df_1998.priceperlb.mean())
print(df_2012.priceperlb.mean())