import pandas as pd
import plotly.graph_objects as go
cholera = pd.read_csv('../input/cholera-dataset/data.csv')
cholera.head()
# Making column names shorter
cholera.rename(columns = {'Number of reported cases of cholera': 'Cases', 
                          'Number of reported deaths from cholera': 'Deaths', 
                          'Cholera case fatality rate': 'Fatality rate', 
                          'WHO Region': 'Region'}, inplace = True)
cholera.info()
# Checking if 'Fatality rate' can be calculated based on 'Cases' and 'Deaths'
cholera[(cholera['Fatality rate'].isnull()) & (~cholera['Cases'].isnull()) & (~cholera['Deaths'].isnull())]
# Checking non-numerical values in Fatality rate column
cholera [(~cholera['Cases'].fillna('0').str.replace(' ','').str.isnumeric()) | (~cholera['Deaths'].fillna('0').str.replace('.','').str.isnumeric()) | (~cholera['Fatality rate'].fillna('0').str.replace('.','').str.isnumeric())]
# Fixing data and changing types to floats
cholera['Cases'] = cholera['Cases'].str.replace('3 5','3').str.replace(' ','').astype('float')
cholera['Deaths'] = cholera['Deaths'].str.replace('Unknown','0').str.replace('0 0','0').astype('float')
cholera['Fatality rate'] = cholera['Fatality rate'].str.replace('Unknown','0').str.replace('0.0 0.0','0').astype('float')
cholera.describe()
cholera[cholera['Fatality rate'] > 100]
cholera.loc[1094, 'Deaths'] = 0
cholera.loc[1094, 'Fatality rate'] = 0
# Total number of Cases
cholera['Cases'].sum()
# Countries with top 10 number of cases
cholera.groupby(['Country'])['Cases'].sum().sort_values(ascending = False).head(10)
# Total number of deaths
cholera['Deaths'].sum()
# Countries with top 10 number of deaths
cholera.groupby(['Country'])['Deaths'].sum().sort_values(ascending = False).head(10)
# 10 year with biggest outbreaks
cholera.groupby(['Year'])['Cases'].sum().sort_values(ascending = False).head(10)
# Statistics for last 5 years 
cholera[cholera['Year'] > 2010].groupby(['Year', 'Region'])['Cases'].sum().sort_index(ascending = [False, True])
import plotly.graph_objects as go

fig = go.Figure(data=go.Choropleth(    
    locations = cholera.groupby('Country')['Cases'].sum().index,
    locationmode = "country names",
    z = cholera.groupby('Country')['Cases'].sum(),
    #text = cholera.groupby('Country')['Cases'].sum().index,
    colorscale = 'Reds_r',
    autocolorscale=False,
    reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_title = 'Number of cases',
))

fig.update_layout(
    title_text='Cholera Cases',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='natural earth'
    )
)

fig.show()
import plotly.graph_objects as go

fig = go.Figure(data=go.Choropleth(    
    locations = cholera.groupby('Country')['Deaths'].sum().index,
    locationmode = "country names",
    z = cholera.groupby('Country')['Deaths'].sum(),
    colorscale = 'Hot',
    autocolorscale=False,
    reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_title = 'Number of cases',
))

fig.update_layout(
    title_text='Cholera Cases',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='natural earth'
    )
)

fig.show()
import plotly.express as px
fig = px.scatter_geo(cholera.dropna(), 
                     locations = "Country", locationmode = "country names", 
                     color="Cases", color_continuous_scale = "Reds",
                     hover_name="Country", 
                     size = cholera.dropna()["Cases"], size_max = 50,
                     animation_frame = "Year",
                     category_orders = {"Year": range(1949,2017)},
                     projection="natural earth"                    
                    )
fig.update_geos(
    showframe=False,
    showcoastlines=False,
    showcountries=True, 
    countrycolor="White")
fig.show()
fig = go.Figure(data=[
     go.Bar(name='Deaths', x = cholera.groupby(['Country'])['Fatality rate'].mean().sort_values(ascending = False).head(15).sort_values(), 
            y = cholera.groupby(['Country'])['Fatality rate'].mean().sort_values(ascending = False).head(15).sort_values().index,
            orientation='h', marker_color='indianred')
])

fig.update_layout(title_text='Cholera cases fatality rate', height = 500)
fig.show()
fig = go.Figure(data=[
    go.Bar(name='Deaths', x = cholera[cholera['Country'] == 'Ukraine']['Year'], y = cholera[cholera['Country'] == 'Ukraine']['Deaths'], 
    marker_color='lightslategray'), 
    go.Bar(name='Cases', x = cholera[cholera['Country'] == 'Ukraine']['Year'], y = cholera[cholera['Country'] == 'Ukraine']['Cases'], marker_color='indianred')
])
fig.update_layout(barmode='stack', title_text='Cholera in Ukraine', height = 400, width = 600)
fig.show()