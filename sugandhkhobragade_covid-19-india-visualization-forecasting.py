#importing Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib

from matplotlib import cm

import geopandas as gpd

import plotly.express as px

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 

#NOTEBOOK WHILE KERNEL IS RUNNING

import plotly.graph_objects as go



from IPython.display import HTML,display

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))


#importing main dataset 

df = pd.read_csv('../input/covid19-in-india/covid_19_india.csv', parse_dates=['Date'], dayfirst=True)



#importing tests dataset

tests = pd.read_csv('../input/covid19-in-india/StatewiseTestingDetails.csv' )

#keeping only required columns

df = df[['Date', 'State/UnionTerritory','Cured','Deaths','Confirmed']]

#renaming column names

df.columns = ['date', 'state','cured','deaths','confirmed']



df.tail()
#current date

today = df[df.date == '2020-09-24']
gdf = gpd.read_file('../input/india-states/Igismap/Indian_States.shp')

#renaming state names

gdf['st_nm'].replace({"Andaman & Nicobar Island": "Andaman and Nicobar Islands",

                      "Arunanchal Pradesh": "Arunachal Pradesh",

                      'Dadara & Nagar Havelli':'Dadra and Nagar Haveli and Daman and Diu',

                      'Jammu & Kashmir':'Jammu and Kashmir',

                      'NCT of Delhi':'Delhi',

                      'Telangana': 'Telengana'

                     }, inplace=True)

merged = gdf.merge(today ,  left_on='st_nm', right_on='state')

merged.state.replace({"Andaman and Nicobar Islands" : "A & N",

                      'Dadra and Nagar Haveli and Daman and Diu' : 'Daman & Diu',

                      'Jammu and Kashmir' : 'J & K'

                      }, inplace=True)

merged['coords'] = merged['geometry'].apply(lambda x: x.representative_point().coords[:])

merged['coords'] = [coords[0] for coords in merged['coords']]





sns.set_context("talk")

sns.set_style("dark")

#plt.style.use('fivethirtyeight')

cmap = 'YlGn'

figsize = (25, 20)

ax = merged.plot(column= 'confirmed', cmap=cmap, 

                          figsize=figsize, scheme='User_Defined',

                          classification_kwds=dict(bins=[100,1000,10000,50000,100000,200000,1000000]),

                          edgecolor='black', legend = True)



for idx, row in merged.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['confirmed'], 

           horizontalalignment='center', bbox={'facecolor': 'yellow', 

                                               'alpha':0.8, 'pad': 5, 'edgecolor':'blue'})



ax.get_legend().set_bbox_to_anchor((0.66,0.9))

ax.get_legend().set_title('Number of cases')

ax.set_title("Confirmed Cases of Covid - 19 ", size = 25)

leg = ax.get_legend()

for lbl in leg.get_texts():

    label_text = lbl.get_text()

    lower = label_text.split()[0]

    upper = label_text.split()[2]

    new_text = f'{float(lower):,.0f} - {float(upper):,.0f}'

    lbl.set_text(new_text)





#a = ax.text(65, 35, 'Total Cases \n 529,577',

           # fontsize=25,

           # )





ax.set_axis_off()

plt.axis('equal')

plt.show()


sns.set_context("talk")

sns.set_style("dark")

#plt.style.use('fivethirtyeight')

cmap = 'OrRd'

figsize = (25, 20)

ax = merged.dropna().plot(column= 'deaths', cmap=cmap, 

                          figsize=figsize, scheme='User_Defined',

                          classification_kwds=dict(bins=[10,50,100,500,1000,2000,4000,8000]),

                          edgecolor='black', legend = True)



for idx, row in merged.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['deaths'], 

           horizontalalignment='center', bbox={'facecolor': 'yellow', 

                                               'alpha':0.8, 'pad': 2, 'edgecolor':'black'})



ax.get_legend().set_bbox_to_anchor((0.66,0.9))

ax.get_legend().set_title('Number of deaths')

ax.set_title("Deaths due to Covid -19 ", size = 25)

leg = ax.get_legend()

for lbl in leg.get_texts():

    label_text = lbl.get_text()

    lower = label_text.split()[0]

    upper = label_text.split()[2]

    new_text = f'{float(lower):,.0f} - {float(upper):,.0f}'

    lbl.set_text(new_text)

#a = ax.text(65, 35, 'Total Deaths \n 16,103',

            #fontsize=25,

            #)

ax.set_axis_off()

plt.axis('equal')

plt.show()






sns.set_context("talk")

sns.set_style("dark")

#plt.style.use('fivethirtyeight')

cmap = 'Blues'

figsize = (25, 20)

ax = merged.dropna().plot(column= 'cured', cmap=cmap, 

                          figsize=figsize, scheme='User_Defined',

                          classification_kwds=dict(bins=[100,1000,10000,50000,100000]),

                          edgecolor='black', legend = True)



for idx, row in merged.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['cured'], 

           horizontalalignment='center', bbox={'facecolor': 'yellow', 

                                               'alpha':0.8, 'pad': 2, 'edgecolor':'black'})



ax.get_legend().set_bbox_to_anchor((0.66,0.9))

ax.get_legend().set_title('Number of Recovered')

ax.set_title("Recovered Cases of Covid-19", size = 25)

leg = ax.get_legend()

for lbl in leg.get_texts():

    label_text = lbl.get_text()

    lower = label_text.split()[0]

    upper = label_text.split()[2]

    new_text = f'{float(lower):,.0f} - {float(upper):,.0f}'

    lbl.set_text(new_text)



#a = ax.text(65, 35, 'Total Recovered \n 310,146',

           # fontsize=25,

            #)

ax.set_axis_off()

plt.axis('equal')

plt.show()
merged['dcratio'] = merged['deaths'] / merged['confirmed'] * 100

merged.dcratio = merged.dcratio.round(2)







sns.set_context("talk")

sns.set_style("dark")

#plt.style.use('fivethirtyeight')

cmap = 'OrRd'

figsize = (25, 20)

ax = merged.dropna().plot(column= 'dcratio', cmap=cmap, 

                          figsize=figsize, scheme='User_Defined',

                          classification_kwds=dict(bins=[0.1 , 0.2 , 0.4 , 0.8 , 1.6 , 3.2 , 6.4]),

                          

                          edgecolor='black', legend = True)



for idx, row in merged.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['dcratio'], 

           horizontalalignment='center', bbox={'facecolor': 'azure', 

                                               'alpha':0.8, 'pad': 2, 'edgecolor':'black'})



ax.get_legend().set_bbox_to_anchor((0.66,0.9))

ax.get_legend().set_title('Percentage')

ax.set_title("Fatality Rate Per State", size = 25)



#a = ax.text(61, 35, 'Total Death/Confirmed \n percentage \n 3.05',

           # fontsize=25,

           # )

ax.set_axis_off()

plt.axis('equal')

plt.show()
confirmed = merged.sort_values('confirmed', ascending = True)



lowc = confirmed.head(10)

highc = confirmed.tail(10)





fig = px.bar(confirmed, x="confirmed", y="state", orientation='h', text = 'confirmed')

fig.update_layout(

    title_text='<b>Confirmed cases of Covid-19 per State <b>',

    title_x=0.5,

    paper_bgcolor='aqua',

    plot_bgcolor = "aqua",

    autosize=False,

    width= 850,

    height=850)

fig.update_traces(marker_color='teal')

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.show()
deaths = merged.sort_values('deaths', ascending = True)

deaths = deaths[deaths.deaths > 0 ]

lowd = deaths.head(10)

highd = deaths.tail(10)

fig = px.bar(deaths, x="deaths", y="state", orientation='h', text = 'deaths')

fig.update_layout(

    title_text='<b>Deaths due to covid 19<b>',

    title_x=0.5,

    paper_bgcolor='rgb(255,223,0)',

    plot_bgcolor = "rgb(255,223,0)",

    autosize=False,

    width=850,

    height= 850)

fig.update_traces(marker_color='red')

fig.update_xaxes(tickfont=dict(family='Rockwell', color='darkblue', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='darkblue', size=14))

fig.show()
recovered = merged.sort_values('cured', ascending = True)

highr = recovered.head(10)

lowr = recovered.tail(10)



fig = px.bar(recovered, x="cured", y="state", orientation='h', text = 'cured')

fig.update_layout(

    title_text='<b> Recovered cases of covid 19 per State<b>',

    title_x=0.5,

    paper_bgcolor='aquamarine',

    plot_bgcolor = "aquamarine",

    autosize=False,

    width= 850,

    height= 850)

fig.update_traces(marker_color='blue')

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.show()
merged = merged[merged.dcratio > 0.0]

merged = merged.sort_values('dcratio', ascending = True)

fig = px.bar(merged, x="dcratio", y="state", orientation='h', text = 'dcratio')

fig.update_layout(

    title_text='<b>Fatality Rate per State<b>',

    title_x=0.5,

    paper_bgcolor='blanchedalmond',

    plot_bgcolor = "blanchedalmond",

    autosize=False,

    width= 850,

    height= 850)

fig.update_traces(marker_color='darkred')

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.show()
df2 = df.groupby(['date'])['confirmed', 'deaths','cured',].sum().reset_index()
#making columns for daily new cases

df2['new_confirmed'] = df2.confirmed.diff()

df2['new_deaths'] = df2.deaths.diff()

df2['new_cured'] = df2.cured.diff()





#taking dates from 15th March

df2 = df2.iloc[44:]
#dropping null 

#df2 = df2.dropna()



#changing float to int

#columns = ['new_confirmed' , 'new_deaths' , 'new_cured']



#for i , c in enumerate(columns):

    #df2[c] = df2[c].astype(int)






fig = go.Figure(go.Bar(x= df2.date, y= df2.cured, name='Recovered'))

fig.add_trace(go.Bar(x=df2.date, y= df2.deaths, name='Deaths'))

fig.add_trace(go.Bar(x=df2.date, y= df2.confirmed, name='Confirmed'))



fig.update_layout(barmode='stack',legend_orientation="h",legend=dict(x= 0.3, y=1.1),

                  xaxis={'categoryorder':'total descending'},

                 title_text='<b>Covid 19 Total cases in India (since 15 March)<b>',

                  title_x=0.5,

                 paper_bgcolor='whitesmoke',

                 plot_bgcolor = "whitesmoke",)

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.show()






fig = go.Figure(go.Bar(x= df2.date, y= df2.new_cured, name='Daily Recovered'))

fig.add_trace(go.Bar(x=df2.date, y= df2.new_deaths, name='Daily Deaths'))

fig.add_trace(go.Bar(x=df2.date, y= df2.new_confirmed, name='Daily Confirmed'))

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.add_annotation(

            x= "2020-04-03",

            y=23000,

           text="<b>Lockdown 1<b>",showarrow= False,font=dict(

            family="Courier New, monospace",

            size=12,

            color="teal"

            ),

        align="center",)

fig.add_annotation(

            x= "2020-04-23",

            y=23000,

            text="<b>Lockdown 2<b>",showarrow= False,font=dict(

            family="Courier New, monospace",

            size=12,

            color="teal"

            ),

        align="center",)

fig.add_annotation(

            x="2020-05-10",

            y=23000,

           text="<b>Lockdown 3<b>",showarrow= False,font=dict(

            family="Courier New, monospace",

            size=12,

            color="teal"

            ),

        align="center",)

fig.add_annotation(

            x= "2020-05-24",

            y=23000,

            text="<b>Lockdown 4<b>",showarrow= False,font=dict(

            family="Courier New, monospace",

            size=12,

            color="teal"

            ),

        align="center",)



fig.update_layout(barmode='stack', 

                  xaxis={'categoryorder':'total descending'},

                 title_text='<b>Daily Cases (since 15 March)<b>',

                  title_x=0.5,legend_orientation="h",legend=dict(x= 0.3, y=1.1),

                 paper_bgcolor='whitesmoke',

                 plot_bgcolor = "whitesmoke",

                  shapes=[

        

        dict(

            type="rect",

            # x-reference is assigned to the x-values

            xref="x",

            # y-reference is assigned to the plot paper [0,1]

            yref="paper",

            x0="2020-03-25",

            y0=0,

            x1="2020-04-14",

            y1=1,

            fillcolor="Lightgreen",

            opacity=0.3,

            layer="below",

            line_width=2,

        ),

        

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-04-15",

            y0=0,

            x1="2020-05-03",

            y1=1,

            fillcolor="Lightgreen",

            opacity=0.3,

            layer="below",

            line_width=2,

        ),

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-05-04",

            y0=0,

            x1="2020-05-17",

            y1=1,

            fillcolor="Lightgreen",

            opacity=0.3,

            layer="below",

            line_width=2,

        ),

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-05-18",

            y0=0,

            x1="2020-05-31",

            y1=1,

            fillcolor="Lightgreen",

            opacity=0.3,

            layer="below",

            line_width=2,

        )

        

        

        

    ]

)

fig.show()
import plotly.graph_objects as go





fig = go.Figure([go.Bar(x= df2.date, y= df2.new_confirmed,marker_color='teal')])

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))



fig.add_annotation(

            x= "2020-04-03",

            y=12500,

           text="<b>Lockdown 1<b>",showarrow= False,font=dict(

            family="Courier New, monospace",

            size=12,

            color="teal"

            ),

        align="center",)

fig.add_annotation(

            x= "2020-04-23",

            y=12500,

            text="<b>Lockdown 2<b>",showarrow= False,font=dict(

            family="Courier New, monospace",

            size=12,

            color="teal"

            ),

        align="center",)

fig.add_annotation(

            x="2020-05-10",

            y=12500,

           text="<b>Lockdown 3<b>",showarrow= False,font=dict(

            family="Courier New, monospace",

            size=12,

            color="teal"

            ),

        align="center",)

fig.add_annotation(

            x= "2020-05-24",

            y=12500,

            text="<b>Lockdown 4<b>",showarrow= False,font=dict(

            family="Courier New, monospace",

            size=12,

            color="teal"

            ),

        align="center",)



fig.update_layout(

    title_text='<b>Daily New Confirmed Cases <b>',

                  title_x=0.5,

                 paper_bgcolor='aqua',

                 plot_bgcolor = "aqua",

    shapes=[

        

        dict(

            type="rect",

            # x-reference is assigned to the x-values

            xref="x",

            # y-reference is assigned to the plot paper [0,1]

            yref="paper",

            x0="2020-03-25",

            y0=0,

            x1="2020-04-14",

            y1=1,

            fillcolor="lightskyblue",

            opacity=0.5,

            layer="below",

            line_width=2,

        ),

        

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-04-15",

            y0=0,

            x1="2020-05-03",

            y1=1,

            fillcolor="lightskyblue",

            opacity=0.5,

            layer="below",

            line_width=2,

        ),

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-05-04",

            y0=0,

            x1="2020-05-17",

            y1=1,

            fillcolor="lightskyblue",

            opacity=0.5,

            layer="below",

            line_width=2,

        ),

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-05-18",

            y0=0,

            x1="2020-05-31",

            y1=1,

            fillcolor="lightskyblue",

            opacity=0.5,

            layer="below",

            line_width=2,

        )

        

        

        

    ]

)



fig.show()

import plotly.graph_objects as go





fig = go.Figure([go.Bar(x= df2.date, y= df2.new_deaths,marker_color='red')])

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))





fig.add_annotation(

            x= "2020-06-17",

            y= 2003,

            text="<b>Highest (17 June,2003 Deaths)<b>",showarrow= True,font=dict(

            family="Courier New, monospace",

            size=15,

            color="blue"

            ),

        align="center",)



fig.update_layout(

    title_text='<b>Daily New Deaths (since 15 March)<b>',

                  title_x=0.5,

                 paper_bgcolor='rgb(255,223,0)',

                 plot_bgcolor = "rgb(255,223,0)",

    shapes=[

        

        dict(

            type="rect",

            # x-reference is assigned to the x-values

            xref="x",

            # y-reference is assigned to the plot paper [0,1]

            yref="paper",

            x0="2020-03-25",

            y0=0,

            x1="2020-04-14",

            y1=1,

            fillcolor="lightyellow",

            opacity=0.3,

            layer="below",

            line_width=2,

        ),

        

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-04-15",

            y0=0,

            x1="2020-05-03",

            y1=1,

            fillcolor="lightyellow",

            opacity=0.3,

            layer="below",

            line_width=2,

        ),

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-05-04",

            y0=0,

            x1="2020-05-17",

            y1=1,

            fillcolor="lightyellow",

            opacity=0.3,

            layer="below",

            line_width=2,

        ),

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-05-18",

            y0=0,

            x1="2020-05-31",

            y1=1,

            fillcolor="lightyellow",

            opacity=0.3,

            layer="below",

            line_width=2,

        )

        

        

        

    ]

)



fig.show()






fig = go.Figure([go.Bar(x= df2.date, y= df2.new_cured,marker_color='blue')])

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))



fig.update_layout(

    title_text='<b>Daily New Recovered Cases (since 15 March)<b>',

                  title_x=0.5,

                 paper_bgcolor='aquamarine',

                 plot_bgcolor = "aquamarine",

    shapes=[

        

        dict(

            type="rect",

            # x-reference is assigned to the x-values

            xref="x",

            # y-reference is assigned to the plot paper [0,1]

            yref="paper",

            x0="2020-03-25",

            y0=0,

            x1="2020-04-14",

            y1=1,

            fillcolor="Lightgreen",

            opacity=0.5,

            layer="below",

            line_width=2,

        ),

        

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-04-15",

            y0=0,

            x1="2020-05-03",

            y1=1,

            fillcolor="Lightgreen",

            opacity=0.5,

            layer="below",

            line_width=2,

        ),

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-05-04",

            y0=0,

            x1="2020-05-17",

            y1=1,

            fillcolor="Lightgreen",

            opacity=0.5,

            layer="below",

            line_width=2,

        ),

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-05-18",

            y0=0,

            x1="2020-05-31",

            y1=1,

            fillcolor="Lightgreen",

            opacity=0.5,

            layer="below",

            line_width=2,

        )

        

        

        

    ]

)



fig.show()


state = list(today.state)

cases = list(today.confirmed)



fig= go.Figure(go.Treemap(

    

    labels =  state,

    parents=[""]*len(state),

    values =  cases,

    textinfo = "label+percent entry"

))

fig.update_layout(

    title_text= '<b>Distribution of Confirmed Cases in India<b>',

    title_x=0.5,

    paper_bgcolor='lightgoldenrodyellow',

    plot_bgcolor = "lightgoldenrodyellow",

    

    )



fig.show()
#Statewise new confirmed

df['new_confirmed'] = df.groupby(['state'])['confirmed'].diff()

df['new_deaths'] = df.groupby(['state'])['deaths'].diff()

df['new_cured'] = df.groupby(['state'])['cured'].diff()
#df = df.dropna()

#columns = ['new_confirmed' , 'new_deaths' , 'new_cured']



#for i , c in enumerate(columns):

    #df[c] = df[c].astype(int)
import plotly.express as px





fig = px.line(df, x="date", y="confirmed", color='state',template= "plotly_white")

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_traces(mode='lines + markers')

fig.update_layout(legend_orientation="h",legend=dict(x= -.1, y=-.3),

                  autosize=False,

                  width= 750,

                  height= 850,

                  title_text='<b>Confirmed Cases of Covid-19 in India<b> ',

                  title_x=0.5,

                 paper_bgcolor='snow',

                 plot_bgcolor = "snow")

fig.show()


state = list(today.state)

cases = list(today.deaths)



fig= go.Figure(go.Treemap(

    

    labels =  state,

    parents=[""]*len(state),

    values =  cases,

    textinfo = "label+percent entry"

))

fig.update_layout(

    title_text= '<b>Distribution of Deaths in India<b>',

    title_x=0.5,

    paper_bgcolor='lightgreen',

    plot_bgcolor = "lightgreen",

    

    )



fig.show()
import plotly.express as px





fig = px.line(df, x="date", y="deaths", color='state',template= "plotly_white")

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_traces(mode='lines + markers')

fig.update_layout(legend_orientation="h",legend=dict(x= -.1, y=-.2),

                  autosize=False,

                  width= 750,

                  height= 850,

                  title_text='<b>Deaths due to Covid-19 in India<b>',

                  title_x=0.5,

                 paper_bgcolor='snow',

                 plot_bgcolor = "snow")

fig.show()


state = list(today.state)

cases = list(today.cured)



fig= go.Figure(go.Treemap(

    

    labels =  state,

    parents=[""]*len(state),

    values =  cases,

    textinfo = "label+percent entry"

))

fig.update_layout(

    title_text= '<b>Distribution of Recovered cases in India<b>',

    title_x=0.5,

    paper_bgcolor='lightgreen',

    plot_bgcolor = "lightgreen",

    

    )



fig.show()
import plotly.express as px





fig = px.line(df, x="date", y="cured", color='state',template= "plotly_white")

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_traces(mode='lines + markers')

fig.update_layout(legend_orientation="h",legend=dict(x= -.1, y=-.2),

                  autosize=False,

                  width= 750,

                  height= 850,

                  title_text='<b>Recovered cases of Covid-19 in India<b>',

                  title_x=0.5,

                 paper_bgcolor='snow',

                 plot_bgcolor = "snow")

fig.show()
highc = highc.sort_values('confirmed', ascending = False)

fig = px.bar(highc, x="state", y="confirmed", text = 'confirmed')

fig.update_layout(

    title_text='<b>Cases in Worst Affeced States <b>',

    title_x=0.5,

    paper_bgcolor='aquamarine',

    plot_bgcolor = "aquamarine",

    )

fig.update_traces(marker_color='blue')

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.show()
states = ['Maharashtra', 'Delhi', 'Tamil Nadu', 'Gujarat', 'Uttar Pradesh',

       'Rajasthan', 'West Bengal', 'Madhya Pradesh', 'Haryana',

       'Andhra Pradesh']







higha = df[df.state.isin(states)]
import plotly.express as px





fig = px.line(higha, x="date", y="confirmed", color='state')

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_traces(mode='lines + markers')

fig.update_layout(legend=dict(x= 0.07, y= 1),

    title_text='<b>Confirmed Cases in Worst affected states<b>',

                  title_x=0.5,

                 paper_bgcolor='whitesmoke',

                 plot_bgcolor = "whitesmoke")

fig.show()
highd = highd.sort_values('deaths', ascending = False)

fig = px.bar(highd, x="state", y="deaths", text = 'deaths')

fig.update_layout(

    title_text='<b> Deaths in  Worst Affeced States<b>',

    title_x=0.5,

    paper_bgcolor='aqua',

    plot_bgcolor = "aqua",

    )

fig.update_traces(marker_color='red')

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.show()






fig = px.line(higha, x="date", y="deaths", color='state')

fig.update_traces(mode='lines+markers')

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_layout(legend=dict(x= 0.07, y= 1),

    title_text='<b>Deaths in Highly affected states<b> ',

                  title_x=0.5,

                 paper_bgcolor='whitesmoke',

                 plot_bgcolor = "whitesmoke")

fig.show()
df = df[df.new_confirmed > 0]

import plotly.express as px





fig = px.line(higha, x="date", y="new_confirmed", color='state',template= "plotly_white")

fig.update_traces(mode='lines +markers')

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_layout(legend=dict(x= 0.05, y= 1),

    title_text='<b>Daily New Confirmed Cases of Covid-19<b> ',

                  title_x=0.5,

                 paper_bgcolor='whitesmoke',

                 plot_bgcolor = "whitesmoke",

    shapes=[

        # 1st highlight during Feb 4 - Feb 6

        dict(

            type="rect",

            # x-reference is assigned to the x-values

            xref="x",

            # y-reference is assigned to the plot paper [0,1]

            yref="paper",

            x0="2020-03-25",

            y0=0,

            x1="2020-04-14",

            y1=1,

            fillcolor="lightskyblue",

            opacity=0.5,

            layer="below",

            line_width=2,

        ),

        # 2nd highlight during Feb 20 - Feb 23

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-04-15",

            y0=0,

            x1="2020-05-03",

            y1=1,

            fillcolor="lightskyblue",

            opacity=0.5,

            layer="below",

            line_width=2,

        ),

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-05-04",

            y0=0,

            x1="2020-05-17",

            y1=1,

            fillcolor="lightskyblue",

            opacity=0.5,

            layer="below",

            line_width=2,

        ),

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-05-18",

            y0=0,

            x1="2020-05-31",

            y1=1,

            fillcolor="lightskyblue",

            opacity=0.5,

            layer="below",

            line_width=2,

        )

        

        

        

    ]

)

fig.show()
df = df[df.new_deaths > 0]

import plotly.express as px





fig = px.line(higha, x="date", y="new_deaths", color='state',template= "plotly_white")

fig.update_traces(mode='lines+markers')

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_layout(legend=dict(x= 0.05, y= 1),

    title_text='<b>Daily New Deaths due to Covid-19<b>',

                  title_x=0.5,

                 

    shapes=[

        # 1st highlight during Feb 4 - Feb 6

        dict(

            type="rect",

            # x-reference is assigned to the x-values

            xref="x",

            # y-reference is assigned to the plot paper [0,1]

            yref="paper",

            x0="2020-03-25",

            y0=0,

            x1="2020-04-14",

            y1=1,

            fillcolor="lightskyblue",

            opacity=0.5,

            layer="below",

            line_width=2,

        ),

        # 2nd highlight during Feb 20 - Feb 23

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-04-15",

            y0=0,

            x1="2020-05-03",

            y1=1,

            fillcolor="lightskyblue",

            opacity=0.5,

            layer="below",

            line_width=2,

        ),

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-05-04",

            y0=0,

            x1="2020-05-17",

            y1=1,

            fillcolor="lightskyblue",

            opacity=0.5,

            layer="below",

            line_width=2,

        ),

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-05-18",

            y0=0,

            x1="2020-05-31",

            y1=1,

            fillcolor="lightskyblue",

            opacity=0.5,

            layer="below",

            line_width=2,

        )

        

        

        

    ]

)

fig.show()
higha = higha[higha.new_confirmed > 0]
states1 = ['Maharashtra', 'Delhi']
#dont delete this again!!!

from plotly.subplots import make_subplots

import plotly.graph_objects as go

fig = make_subplots(rows = 1, cols = 2 , subplot_titles= ['<b>Maharashtra<b>' , '<b>Delhi<b>'] )

for i , state in enumerate(states1):

    highan = higha[higha.state == state]

    

    

    fig.add_trace(

        go.Bar(x= highan.date, y= highan.new_confirmed, marker_color='blue',name= state),

    row=(i // 2) + 1, col=((i % 2) + 1)

        

    )

    fig.add_annotation(

            xref='x1', 

            yref='y1',

            x= "2020-05-02",

            y= 12500,

            text="Lockdown",showarrow= False,font=dict(

            family="Courier New, monospace",

            size=14,

            color="blue"

            ),

        align="center",)

    fig.add_annotation(

            xref='x2', 

            yref='y2',

            x= "2020-05-02",

            y= 2200,

            text="Lockdown",showarrow= False,font=dict(

            family="Courier New, monospace",

            size=14,

            color="blue"

            ),

        align="center",)

    fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

    fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

    fig.update_layout(

                     showlegend=False,

                     title_text='<b>Daily New Confirmed Cases<b>',

                     title_x=0.5,

                     paper_bgcolor='gold',

                     plot_bgcolor = "gold",

                     shapes=[dict(type="rect",xref="x1",yref="paper",

                                  x0="2020-03-25",y0=0,x1="2020-05-31",

                                  y1=1,fillcolor="lightyellow",opacity=0.5,

                                  layer="below",line_width=2 ),

                             dict(type="rect",xref="x2",yref="paper",

                                  x0="2020-03-25",y0=0,x1="2020-05-31",

                                  y1=1,fillcolor="lightyellow",opacity=0.5,

                                  layer="below",line_width=2 ),

                             

                            ]

                     )



fig.show() 
from plotly.subplots import make_subplots

import plotly.graph_objects as go

fig = make_subplots(rows = 1, cols = 2 , subplot_titles= ['<b>Maharashtra<b>' , '<b>Delhi<b>'] )

for i , state in enumerate(states1):

    highan = higha[higha.state == state]

    

    

    fig.add_trace(

        go.Bar(x= highan.date, y= highan.new_deaths, marker_color='red',name= state),

    row=(i // 2) + 1, col=((i % 2) + 1)

        

    )

    

    fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

    fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

    fig.update_layout(showlegend=False,

                     title_text='<b>Daily New Deaths <b>',

                     title_x=0.5,

                     paper_bgcolor='lightcyan',

                     plot_bgcolor = "lightcyan",

                     shapes=[dict(type="rect",xref="x1",yref="paper",

                                  x0="2020-03-25",y0=0,x1="2020-05-31",

                                  y1=1,fillcolor="snow",opacity=0.5,

                                  layer="below",line_width=2 ),

                             dict(type="rect",xref="x2",yref="paper",

                                  x0="2020-03-25",y0=0,x1="2020-05-31",

                                  y1=1,fillcolor="snow",opacity=0.5,

                                  layer="below",line_width=2 ),

                             

                            ]

                     )



fig.show() 
states2 = ['Rajasthan', 'Uttar Pradesh', 'Haryana', 'Andhra Pradesh']
from plotly.subplots import make_subplots

import plotly.graph_objects as go

fig = make_subplots(rows = 2, cols = 2 , subplot_titles=["<b>Rajasthan<b>","<b>Uttar Pradesh<b>","<b>Haryana<b>",'<b>Andhra Pradesh<b>'] )

for i , state in enumerate(states2):

    highan = higha[higha.state == state]

    

    

    fig.add_trace(

        go.Bar(x= highan.date, y= highan.new_confirmed, marker_color='blue',name= state),

    row=(i // 2) + 1, col=((i % 2) + 1)

        

    )

    fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

    fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

    fig.update_layout(showlegend=False,

                     title_text='<b>Daily New Confirmed Cases<b>',

                     title_x=0.5,

                     paper_bgcolor='aqua',

                     plot_bgcolor = "aqua",

                     shapes=[dict(type="rect",xref="x1",yref="paper",

                                  x0="2020-03-25",y0=0,x1="2020-05-31",

                                  y1=1,fillcolor="lightskyblue",opacity=0.5,

                                  layer="below",line_width=2 ),

                             dict(type="rect",xref="x2",yref="paper",

                                  x0="2020-03-25",y0=0,x1="2020-05-31",

                                  y1=1,fillcolor="lightskyblue",opacity=0.5,

                                  layer="below",line_width=2 ),

                             

                            ]

                     )



fig.show() 

    

from plotly.subplots import make_subplots

import plotly.graph_objects as go

fig = make_subplots(rows = 2, cols = 2 , subplot_titles=["<b>Rajasthan<b>","<b>Uttar Pradesh<b>","<b>Haryana<b>",'<b>Andhra Pradesh<b>'] )

for i , state in enumerate(states2):

    highan = higha[higha.state == state]

    

    

    fig.add_trace(

        go.Bar(x= highan.date, y= highan.new_deaths, marker_color='red',name= state),

    row=(i // 2) + 1, col=((i % 2) + 1)

        

    )

    fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

    fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

    fig.update_layout(showlegend=False,

                     title_text='<b>Daily New Deaths<b>',

                     title_x=0.5,

                     paper_bgcolor='lightyellow',

                     plot_bgcolor = "lightyellow",

                     shapes=[dict(type="rect",xref="x1",yref="paper",

                                  x0="2020-03-25",y0=0,x1="2020-05-31",

                                  y1=1,fillcolor="snow",opacity=0.5,

                                  layer="below",line_width=2 ),

                             dict(type="rect",xref="x2",yref="paper",

                                  x0="2020-03-25",y0=0,x1="2020-05-31",

                                  y1=1,fillcolor="snow",opacity=0.5,

                                  layer="below",line_width=2 ),

                             

                            ]

                     )



fig.show() 
states3 = ['Tamil Nadu', 'West Bengal', 'Gujarat', 'Madhya Pradesh']
fig = make_subplots(rows = 2, cols = 2 , subplot_titles=['<b>Tamil Nadu<b>', '<b>West Bengal<b>', '<b>Gujarat<b>', '<b>Madhya Pradesh<b>'] )

for i , state in enumerate(states3):

    highan = higha[higha.state == state]

    

    

    fig.add_trace(

        go.Bar(x= highan.date, y= highan.new_confirmed, marker_color='blue',name= state),

    row=(i // 2) + 1, col=((i % 2) + 1)

        

    )

    fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

    fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

    fig.update_layout(showlegend=False,

                     title_text='<b>Daily New Confirmed Cases<b>',

                     title_x=0.5,

                     paper_bgcolor='aquamarine',

                     plot_bgcolor = "aquamarine",

                     shapes=[dict(type="rect",xref="x1",

                                  yref="paper",x0="2020-03-25",

                                  y0=0,x1="2020-05-31",y1=1,fillcolor="lightgreen",

                                  opacity=0.5,layer="below",line_width=2 ),

                             dict(type="rect",xref="x2",

                                  yref="paper",x0="2020-03-25",

                                  y0=0,x1="2020-05-31",y1=1,fillcolor="lightgreen",

                                  opacity=0.5,layer="below",line_width=2 ),

                            

                            ]

                     )



fig.show() 

    
fig = make_subplots(rows = 2, cols = 2 , subplot_titles=['<b>Tamil Nadu<b>', '<b>West Bengal<b>', '<b>Gujarat<b>', '<b>Madhya Pradesh<b>'] )

for i , state in enumerate(states3):

    highan = higha[higha.state == state]

    

    

    fig.add_trace(

        go.Bar(x= highan.date, y= highan.new_deaths, marker_color='red',name= state),

    row=(i // 2) + 1, col=((i % 2) + 1)

        

    )

    fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

    fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

    fig.update_layout(showlegend=False,

                     title_text='<b>Daily New Deaths<b>',

                     title_x=0.5,

                     paper_bgcolor='cyan',

                     plot_bgcolor = "cyan",

                     shapes=[dict(type="rect",xref="x1",

                                  yref="paper",x0="2020-03-25",

                                  y0=0,x1="2020-05-31",y1=1,fillcolor="lightskyblue",

                                  opacity=0.5,layer="below",line_width=2 ),

                             dict(type="rect",xref="x2",

                                  yref="paper",x0="2020-03-25",

                                  y0=0,x1="2020-05-31",y1=1,fillcolor="lightskyblue",

                                  opacity=0.5,layer="below",line_width=2 ),

                            

                            ]

                     )



fig.show() 
tests.isna().sum()
#keeping only required columns

tests = tests[['Date', 'State','TotalSamples', 'Positive']]
#renaming column names

tests.columns = ['date','state','samples' , 'positive']
import plotly.express as px





fig = px.line(tests, x="date", y="samples", color='state',

             color_discrete_sequence=px.colors.cyclical.Phase,)

fig.update_traces(mode='lines + markers')

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_layout(legend_orientation="h",legend=dict(x= -.1, y=-.3),

                  autosize=False,

                  width= 750,

                  height= 850,

    title_text='<b>Testing of Covid-19 in India<b> ',

                  title_x=0.5,

                 paper_bgcolor='azure',

                 plot_bgcolor = "azure")

fig.show()
tests
today2 = tests[tests.date == '2020-09-23']



today2 = today2.sort_values('samples' , ascending = True)
state = list(today2.state)

samples = list(today2.samples)



fig= go.Figure(go.Treemap(

    

    labels =  state,

    parents=[""]*len(state),

    values =  samples,

    textinfo = "label+percent entry"

))

fig.update_layout(

    title_text= '<b>Distribution of Testing in India<b>',

    title_x=0.5,

    paper_bgcolor='khaki',

    plot_bgcolor = "khaki",

    

    )



fig.show()




fig = px.bar(today2, x="samples", y="state", orientation='h')

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_layout(

    title_text='<b>Number of Samples Tested <b>',

    title_x=0.5,

    paper_bgcolor='blanchedalmond',

    plot_bgcolor = "blanchedalmond",

    autosize=False,

    width= 800,

    height= 900)

fig.update_traces(marker_color='darkred')

fig.show()
today2['posper'] = today2['positive'] / today2['samples'] * 100

today2.posper = today2.posper.round(2)

today2 = today2.sort_values('posper' , ascending = True)


fig = px.bar(today2, x="posper", y="state", orientation='h')

fig.update_layout(

    title_text='<b>Covid Positive per Sample percentage<b>',

    title_x=0.5,

    paper_bgcolor='rgb(245, 219, 147)',

    plot_bgcolor = "rgb(245, 219, 147)",

    autosize=False,

    width= 800,

    height= 850)

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_traces(marker_color='darkviolet')

fig.show()
#keeping date and Confirmed columns

df3 = df2[['date' , 'confirmed']]



#Renaming column names according to fb prophet

df3.columns = ['ds' , 'y']
#importing fbprophet

from fbprophet import Prophet



#model

m = Prophet()



#fitting the model

m.fit(df3)
#forecast

future = m.make_future_dataframe(periods= 20) 

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(21)

#fig1 = m.plot(forecast)

#fig2 = m.plot_components(forecast)

from fbprophet.plot import plot_plotly

fig = plot_plotly(m, forecast)  # This returns a plotly Figure

fig.update_layout(

                  autosize=False,

                  width= 750,

                  height= 800,

    title_text='<b>Covid-19 Total cases Forecast<b>',

    title_x=0.5,

    paper_bgcolor='khaki',

    plot_bgcolor = "khaki",)

fig.show()
df4 = df2[['date' , 'deaths']]



df4.columns = ['ds', 'y']

m = Prophet()

m.fit(df4)

future = m.make_future_dataframe(periods= 20) 

#future.tail()

forecast = m.predict(future)

#forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(21)

fig = plot_plotly(m, forecast)  # This returns a plotly Figure

fig.update_layout(autosize=False,

                  width= 750,

                  height= 800,

    title_text='<b>Covid-19 Deaths  Forecast<b>',

    title_x=0.5,

    paper_bgcolor='rgb(245, 219, 147)',

    plot_bgcolor = "rgb(245, 219, 147)",)

fig.show()
maha = df[df.state =='Maharashtra']

maha = maha[['date', 'confirmed']]

maha.columns = ['ds', 'y']

m = Prophet()

m.fit(maha)

future = m.make_future_dataframe(periods= 20) 

#future.tail()

forecast = m.predict(future)

#forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(21)

fig = plot_plotly(m, forecast)  # This returns a plotly Figure

fig.update_layout(autosize=False,

                  width= 750,

                  height= 800,

    title_text='<b>Covid-19 Maharashtra cases Forecast<b>',

    title_x=0.5,

    paper_bgcolor='aqua',

    plot_bgcolor = "aqua",)

fig.show()
delhi = df[df.state =='Delhi']

delhi = delhi[['date', 'confirmed']]

delhi.columns = ['ds', 'y']

m = Prophet()

m.fit(delhi)

future = m.make_future_dataframe(periods= 20) 

#future.tail()

forecast = m.predict(future)

#forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(21)

fig = plot_plotly(m, forecast)  # This returns a plotly Figure

fig.update_layout(autosize=False,

                  width= 750,

                  height= 800,

    title_text='<b>Covid-19 Delhi cases Forecast<b>',

    title_x=0.5,

    paper_bgcolor='aquamarine',

    plot_bgcolor = "aquamarine",)

fig.show()
