import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


from datetime import date


sns.set(style='whitegrid')
pio.templates.default = "plotly_white"
%matplotlib inline
warnings.filterwarnings('ignore')


df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
world_coor = pd.read_csv('../input/world-coor/world_coordinates.csv')

df.head()
df.info()
df.isna().sum()
df.columns = df.columns.str.replace(' ','_').str.replace('/','_')
df.rename(columns={'Country_Region':'Country'}, inplace=True)
df['Date'] = df['ObservationDate'].apply(pd.to_datetime)
df['Last_Update'] = pd.to_datetime(df.Last_Update)
df.drop('SNo', axis=1, inplace=True)
df.Country.replace({'Mainland China': 'China'}, inplace=True)
df = df.astype({'Confirmed':'int','Deaths':'int','Recovered':'int'})

d = df['Date'][-1:].astype('str')
year = int(d.values[0].split('-')[0])
month = int(d.values[0].split('-')[1])
day = int(d.values[0].split('-')[2].split()[0])

df_1 = df[df['Date'] >= pd.Timestamp(date(year,month,day))]
df_1
df['Date'] = df['Date'].dt.date
glob_spread = df[df['Date'] > pd.Timestamp(date(2020,1,21))]
glob_spread = glob_spread.groupby('Date')["Confirmed", "Deaths", "Recovered"].sum().reset_index()

total_cases = df_1[['Confirmed','Deaths','Recovered']].sum()
total_cases = total_cases.to_frame().reset_index()
total_cases.rename(columns={'index':'types',0:'Total'}, inplace=True)
labels = total_cases.types
values = total_cases.Total

colors = ['STEELBLUE','crimson','MEDIUMSEAGREEN']

fig_1 = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', showlegend=False)])

fig_1.update_traces(marker=dict(colors=colors, line=dict(color='#000000', width=0.5)))

fig_1.update_layout( margin={"r":0,"t":100,"l":0,"b":0},
    title_text="<br>COVID-19 Global Cases<br>",
    font=dict(size=15, color='black', family="Arial, Balto, Courier New, Droid Sans"),

)
fig_1.show()

fig_2 = go.Figure()
fig_2.add_trace(go.Scatter(x=glob_spread['Date'], y=glob_spread['Confirmed'], name='Confirmed',
                         line=dict(color='STEELBLUE', width=4),  mode='lines+markers'))

fig_2.add_trace(go.Scatter(x=glob_spread['Date'], y=glob_spread['Recovered'], name = 'Recovered',
                         line=dict(color='MEDIUMSEAGREEN', width=4),mode='lines+markers'))
                
fig_2.add_trace(go.Scatter(x=glob_spread['Date'], y=glob_spread['Deaths'], name='Deaths',
                         line=dict(color='crimson', width=4),mode='lines+markers'
                          ))
                
fig_2.update_layout(yaxis_title='COVID-19 Total Cases',
    xaxis=dict(
        showline=False,
        showgrid=False,
        showticklabels=True,
        linecolor='black',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
    #plot_bgcolor='white'
)

fig_2.show()
count = df.Country.unique()
cases = df_1.groupby('Country')[['Confirmed','Deaths','Recovered']].sum()
most_count = cases.sort_values(by=['Confirmed','Deaths','Recovered'], ascending=False)
df_world = pd.merge(world_coor,cases,on='Country')

print(count)
print('-----'*20)
print('Number of Countries affected by NCov Virus: ',len(count))

fig_3 = go.Figure()
fig_3.add_trace(go.Scattergeo(
        lon = df_world['longitude'],
        lat = df_world['latitude'],
        mode='markers',
         marker=dict(
            size=10,
            color='rgb(255, 0, 0)',
            opacity=0.7
         ),
        text = df_world[['Country','Confirmed']],
        hoverinfo='text'))

fig_3.update_layout(mapbox_style="carto-darkmatter",
        margin={"r":0,"t":100,"l":0,"b":0},
       title=dict(text="<br>Countries with confirmed cases of COVID-19, 2020<br>"),
                             font=dict(size=15, color='black', family="Arial, Balto, Courier New, Droid Sans"),
        geo = dict(
            landcolor = 'rgb(217, 217, 217)',
            #projection_type="equirectangular",
            coastlinecolor = "black",
             bgcolor = "#9cd3db",
            showframe=False,
            showcountries=True,
        
        
            
            
        ),
    )

fig_3.show()

plt.figure(figsize=(10,5))
ax = sns.heatmap(most_count[:10], annot=True, fmt="d", linewidths=0.1, cmap='Reds', linecolor='white', cbar=False)
fig_4 = go.Figure(data=go.Choropleth(
    locations = df_world['Country'],
    z = df_world['Confirmed'],
    locationmode="country names",
    text = df_world['Country'],
    autocolorscale=False,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_title = 'Confirmed Cases<br>of COVID-19',
    colorscale = 'peach',
))

fig_4.update_layout(margin={"r":0,"t":40,"l":0,"b":0},
               
    geo=dict(
        showframe=False,
        showcoastlines=True,
        showcountries=True,
        projection_type='equirectangular',
         coastlinecolor = "black",
            countrycolor='black',
        scope='asia'
        
    ),

)

fig_4.show()
#burg, geyser, hsv
china_prov = df_1[df_1.Country =='China']
china_prov = china_prov.groupby('Province_State')[['Confirmed','Recovered']].sum().sort_values(by='Confirmed', ascending=False).reset_index()
china_prov = china_prov[:10].sort_values(by='Confirmed', ascending=True)

fig_5 = go.Figure()
fig_5.add_trace(go.Bar(
    y=china_prov['Province_State'],
    x=china_prov['Recovered'],
    text=china_prov['Recovered'],
    textposition='auto',
    name='Recovered',
    orientation='h',
    marker=dict(
        color='STEELBLUE')
    )
)
fig_5.add_trace(go.Bar(
    y=china_prov['Province_State'],
    x=china_prov['Confirmed'],
    text=china_prov['Confirmed'],
    textposition='auto',
    name='Confirmed',
    orientation='h',
    marker=dict(
        color='crimson',
    )
    )
)

fig_5.update_layout(
    yaxis=dict(
        title='States',
        titlefont_size=16,
        tickfont_size=14),
    barmode='group',
    margin=dict(l=0, r=0, t=20, b=0))
    
fig_5.show()
fig_6 = go.Figure(data=go.Choropleth(
    locations = df_world['Country'],
    z = df_world['Confirmed'],
    locationmode="country names",
    text = df_world['Country'],
    autocolorscale=False,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_title = 'Confirmed Cases<br>of COVID-19',
    colorscale = 'Reds',
))

fig_6.update_layout(margin={"r":0,"t":40,"l":0,"b":0},
               
    geo=dict(
        showframe=False,
        showcoastlines=True,
        showcountries=True,
        projection_type='equirectangular',
         coastlinecolor = "black",
            countrycolor='black',
        scope='europe'
        
    ),

)

fig_6.show()
#burg, geyser, hsv
fig_8 = go.Figure(data=go.Choropleth(
    locations = df_world['Country'],
    z = df_world['Confirmed'],
    locationmode="country names",
    text = df_world['Country'],
    autocolorscale=False,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_title = 'Confirmed Cases<br>of COVID-19',
    colorscale = 'Reds',
))

fig_8.update_layout(margin={"r":0,"t":40,"l":0,"b":0},
               
    geo=dict(
        showframe=False,
        showcoastlines=True,
        showcountries=True,
        projection_type='equirectangular',
         coastlinecolor = "black",
            countrycolor='black',
        scope='usa'
        
    ),

)

fig_8.show()
#burg, geyser, hsv
us_prov = df_1[df_1.Country =='US']
us_prov = us_prov.groupby('Province_State')[['Confirmed','Recovered']].sum().sort_values(by='Confirmed', ascending=False).reset_index()
us_prov = us_prov[:10].sort_values(by='Confirmed', ascending=True)

fig_9 = go.Figure()
fig_9.add_trace(go.Bar(
    y=us_prov['Province_State'],
    x=us_prov['Recovered'],
    text=us_prov['Recovered'],
    textposition='auto',
    name='Recovered',
    orientation='h',
    marker=dict(
        color='STEELBLUE')
    )
)
fig_9.add_trace(go.Bar(
    y=us_prov['Province_State'],
    x=us_prov['Confirmed'],
    text=china_prov['Confirmed'],
    textposition='auto',
    name='Confirmed',
    orientation='h',
    marker=dict(
        color='crimson',
    )
    )
)

fig_9.update_layout(
    yaxis=dict(
        title='States',
        titlefont_size=16,
        tickfont_size=14),
    barmode='group',
    margin=dict(l=0, r=0, t=20, b=0))
    
fig_9.show()