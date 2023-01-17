import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib.animation as animation

from IPython.display import HTML

import plotly.express as px

import plotly.graph_objects as go

import seaborn as sns

from plotly.subplots import make_subplots

%matplotlib inline





import plotly.tools as tls

import cufflinks as cf

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



init_notebook_mode(connected=True)



print(__version__) # requires version >= 1.9.0

cf.go_offline()
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df.head()
df.drop(['SNo'],1,inplace=True)
df_reg=df.groupby(['Country/Region']).agg({'Confirmed':'sum','Recovered':'sum','Deaths':'sum'}).sort_values(["Confirmed"],ascending=False).reset_index()

df_reg['Survival Rate'] = round(df_reg['Recovered']/df_reg['Confirmed']*100,2)

df_reg['Mortality Rate'] = round(df_reg['Deaths']/df_reg['Confirmed']*100,2)

df_reg.head(10)
fig = go.Figure(data=[go.Table(

    columnwidth = [50],

    header=dict(values=('Country/Region', 'Confirmed', 'Recovered', 

                       'Deaths', 'Survival Rate', 'Mortality Rate'),

                fill_color='#104E8B',

                align='center',

                font_size=14,

                font_color='white',

                height=40),

    cells=dict(values=[df_reg['Country/Region'].head(10), df_reg['Confirmed'].head(10), df_reg['Recovered'].head(10), 

                       df_reg['Deaths'].head(10), df_reg['Survival Rate'].head(10), df_reg['Mortality Rate'].head(10)],

               fill=dict(color=['#509EEA', '#A4CEF8',]),

               align='right',

               font_size=12,

               height=30))

])



fig.show()

df_reg.iplot(kind='box')
fig = px.pie(df_reg.head(10),

             values="Survival Rate",

             names="Country/Region",

             title="Survival Rate",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo='value+label')

fig.show()
fig = px.pie(df_reg.head(10),

             values="Mortality Rate",

             names="Country/Region",

             title="Mortality Rate",

             template="seaborn",

             )

fig.update_traces(rotation=90, pull=0.05, textinfo='value+label')

fig.show()
df_country=df.groupby(['ObservationDate','Country/Region']).agg({'Confirmed':'sum','Recovered':'sum','Deaths':'sum'}).sort_values(["Confirmed"],ascending=False)

df_country.head(10)
df_country = df_country.reset_index(col_level = 1)

df_country['Dates'] = pd.to_datetime(df_country['ObservationDate'], format = '%m/%d/%Y')

df_country['Mortality Rate'] = df_country['Deaths']/df_country['Confirmed']*100
fig, ax = plt.subplots(figsize=(15, 8))



colors = ['#FFBCBC','#FFA9A9','#FF8888', '#FF6F6F', 

          '#FF4D4D', '#FF4141','#FF2B2B','#FE0000','#F70000','#E60000',

          '#C30000', '#CD0000','#8B0000', '#800000','#660000']



def draw_barchart(year):

    dff = df_country[df_country['Dates'].eq(year)].sort_values(["Confirmed"],ascending=False).head(15)

    dff = dff[::-1]

    dff['Dates'] = pd.to_datetime(dff['Dates'])

    dff['Dates'] = dff['Dates'].dt.strftime('%d/%m/%Y')

    ax.clear()

    ax.barh(dff['Country/Region'], dff['Confirmed'], color=colors)

    for i, (value, name) in enumerate(zip(dff['Confirmed'], dff['Country/Region'])):

        ax.text(value, i, name, size=14, weight=600, ha='left', va='bottom')

        ax.text(value, i-.25, f'{value:,.0f}',  size=14, ha='left',  va='center')

        

    ax.text(1, 0.2, year, transform=ax.transAxes, color='#777777', size=36, ha='right', weight=800)

    ax.text(0, 1.06, 'Confirmed Cases', transform=ax.transAxes, size=12, color='#777777')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    ax.xaxis.set_ticks_position('top')

    ax.tick_params(axis='x', colors='#777777', labelsize=12)

    ax.set_yticks([])

    ax.margins(0, 0.01)

    ax.grid(which='major', axis='x', linestyle='-')

    ax.set_axisbelow(True)

    ax.text(0, 1.12, 'Confirmed Cases by Country',

            transform=ax.transAxes, size=24, weight=600, ha='left', backgroundcolor = 'whitesmoke')

    ax.text(1, 0, 'by @AlenaVorushilova', transform=ax.transAxes, ha='right',

            color='#777777', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))

    plt.box(False)

    

draw_barchart('03-24-2020')
fig, ax = plt.subplots(figsize=(15, 8))

date1 =  df_country['Dates'].min()

date2 =  df_country['Dates'].max()

animator = animation.FuncAnimation(fig, draw_barchart, frames=pd.date_range(date1, date2).tolist(), interval=50)

HTML(animator.to_jshtml())
fig, ax = plt.subplots(figsize=(15, 8))



colors1 = ['#E8EAFA','#C8CDF0','#A6AEE7', '#8798DF', 

          '#9DA9E4', '#95A4DE','#8798DF','#6876C1','#4E64CE','#3E59C2',

          '#314BB6', '#3B4990','#283A90', '#22316C','#162252']



def draw_barmort(year):

    dff1 = df_country[df_country['Dates'].eq(year)].sort_values(['Deaths'],ascending=False).head(15)

    dff1 = dff1[::-1]

    ax.clear()

    ax.barh(dff1['Country/Region'], dff1['Deaths'], color=colors1)

    for i, (value, name) in enumerate(zip(dff1['Deaths'], dff1['Country/Region'])):

        ax.text(value, i, name, size=14, weight=600, ha='left', va='bottom')

        ax.text(value, i-.25, f'{value:,.0f}',  size=14, ha='left',  va='center')

        

    ax.text(1, 0.2, year, transform=ax.transAxes, color='#777777', size=36, ha='right', weight=800)

    ax.text(0, 1.06, 'Fatal Cases', transform=ax.transAxes, size=12, color='#777777')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    ax.xaxis.set_ticks_position('top')

    ax.tick_params(axis='x', colors='#777777', labelsize=12)

    ax.set_yticks([])

    ax.margins(0, 0.01)

    ax.grid(which='major', axis='x', linestyle='-')

    ax.set_axisbelow(True)

    ax.text(0, 1.12, 'Fatal Cases by Country - Top 15',

            transform=ax.transAxes, size=24, weight=600, ha='left', backgroundcolor = 'whitesmoke')

    ax.text(1, 0, 'by @AlenaVorushilova', transform=ax.transAxes, ha='right',

            color='#777777', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))

    plt.box(False)

    

#draw_barmort('03-23-2020')



date1 = df_country['Dates'].min()

date2 = df_country['Dates'].max()

animator = animation.FuncAnimation(fig, draw_barmort, frames=pd.date_range(date1, date2).tolist(), interval=50)

HTML(animator.to_jshtml())

#animator.save('Fatal.gif', writer='Alena', fps=30)
dfd = df_country.groupby('Dates').sum()

dfd.head()
dfd[['Confirmed', 'Recovered','Deaths']].iplot(title = 'World Situation Over Time')
dfd.reset_index(level = 0, inplace = True)
fig = make_subplots(rows=1, cols=3, subplot_titles=('Comfirmed Cases', 'Fatal Cases', 'Recovered'))



trace1 = go.Scatter(

                x=dfd['Dates'],

                y=dfd['Confirmed'],

                name='Comfirmed Cases',

                mode='lines',

                line=dict(color='rgb(115,115,115)', width=2),

                connectgaps=True)

trace2 = go.Scatter(

                x=dfd['Dates'],

                y=dfd['Deaths'],

                name='Fatal Cases',

                line_color='#9D1309',

                mode='lines',

                opacity=0.8)



trace3 = go.Scatter(

                x=dfd['Dates'],

                y=dfd['Recovered'],

                name='Recovered',

                mode='lines',

                line_color='#00C957',

                opacity=0.8)





fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)

fig.update_layout(title_text = '<b>Global Spread of the COVID-19 </b>',

                  font=dict(

                      family='Arial, Balto, Courier New, Droid Sans',

                      color='#35586C'),

                  xaxis=dict(ticks='outside',

                             showline=True,

                             showticklabels=True,

                             linewidth=2,

                             tickfont=dict(family='Arial',

                                           size=12,

                                           color='rgb(115,115,115)'

                                          ),

                            ),

                  yaxis=dict(showgrid=True,

                             zeroline=False,

                             showline=False,

                             showticklabels=True,

                            ),

                  autosize=False,

                  margin=dict(autoexpand=False,

                              l=100,

                              r=20,

                              t=110,

                             ),

                  showlegend=False,

                  plot_bgcolor='#F2F2F2'

                 )

fig.show()
dfd['Active'] = dfd['Confirmed']-dfd['Deaths']-dfd['Recovered']
dfd_comp = dfd.melt(id_vars='Dates', value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='case', value_name='Count')





fig = px.area(dfd_comp, x='Dates', y='Count', color='case',

             title='Cases over time: Area Plot', color_discrete_sequence = ['green', 'red', 'orange'])

fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=dfd['Dates'], y=dfd['Deaths'],

    mode='lines',

    name='Fatal Cases',

    line=dict(width=0.5, color='#BE2625'),

    stackgroup='one',

    groupnorm='percent' # sets the normalization for the sum of the stackgroup

))

fig.add_trace(go.Scatter(

    x=dfd['Dates'], y=dfd['Active'],

    mode='lines',

    name='Active Cases',

    line=dict(width=0.5, color='#FFA500'),

    stackgroup='one'

))

fig.add_trace(go.Scatter(

    x=dfd['Dates'], y=dfd['Recovered'],

    mode='lines',

    name='Recovered Cases',

    line=dict(width=0.5, color='#138F6A'),

    stackgroup='one'

))



fig.update_layout(

    showlegend=True,

    xaxis_tickformat = '%d %B (%a)<br>%Y',

    yaxis=dict(

        type='linear',

        range=[1, 100],

        ticksuffix='%'))



fig.show()
df_country.head()
df_glob = df_country.groupby(['Dates', 'Country/Region'])['Confirmed', 'Deaths'].max().reset_index()

df_glob['Dates'] = pd.to_datetime(df_glob['Dates'])

df_glob['Dates'] = df_glob['Dates'].dt.strftime('%m/%d/%Y')

df_glob['size'] = df_glob['Confirmed'].pow(0.3)



fig = px.scatter_geo(df_glob, locations='Country/Region', locationmode='country names', 

                     color="Confirmed", size='size', hover_name="Country/Region", 

                     range_color= [0, 1500], 

                     projection="mercator", animation_frame='Dates', 

                     title='COVID-19: World Spread', color_continuous_scale="portland")

fig.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})

fig.show()