import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from plotly import __version__

import cufflinks as cf

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

cf.go_offline()



df_raw=pd.read_csv("../input/externalourworldindata/international-tourism-number-of-arrivals.csv")

df_raw
df_raw.rename(columns={'Entity':'Country',"Unnamed: 3":"No of Tourist"},inplace=True)

df_raw["No of Tourist in million"]=df_raw["No of Tourist"]/1000000

df_raw.drop(columns=["No of Tourist"],inplace=True)
df_raw.head()
df_raw[df_raw["Country"]=="World"].plot(x="Year",y="No of Tourist in million",kind="bar")

plt.xlabel("Year")



plt.title("Worldwide tourist bar chart")

df_raw[df_raw["Country"]=="World"]["No of Tourist in million"].iplot()
df_raw.set_index("Year",inplace=True)



df_raw[df_raw["Country"]=="World"]["No of Tourist in million"].iplot(title="World Tourist yearwise",xTitle='Years',yTitle='Number of tourist (in millions)',

                                                               theme='Solar',color='red')
df_raw.reset_index(inplace=True)


df_raw[df_raw["Country"]=="World"].plot(x="Year",y="No of Tourist in million",kind="bar")

plt.xlabel("Year")



plt.title("Worldwide tourist bar chart")

#2003

# Annotate arrow

plt.annotate('',                      # s: str. will leave it blank for no text

             xy=(8,700),             # place head of the arrow 

             xytext=(2, 1000),         # place base of the arrow 

             xycoords='data',         # will use the coordinate system of the object being annotated 

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)

            )



# Annotate Text

plt.annotate('Iraq conflict,SARS', # text to display

             xy=(3.3,700),                    # start the text at at point .

             rotation=330,                  # based on trial and error to match the arrow

             va='bottom',                    # want the text to be vertically 'bottom' aligned

             ha='left',                      # want the text to be horizontally 'left' algned.

            )



#2009

# Annotate arrow

plt.annotate('',                      # s: str. will leave it blank for no text

             xy=(14,900),             # place head of the arrow 

             xytext=(9, 1200),         # place base of the arrow 

             xycoords='data',         # will use the coordinate system of the object being annotated 

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)

            )



# Annotate Text

plt.annotate('H1N1 pandemic', # text to display

             xy=(10,950),                    # start the text at at point .

             rotation=325,                  # based on trial and error to match the arrow

             va='bottom',                    # want the text to be vertically 'bottom' aligned

             ha='left',                      # want the text to be horizontally 'left' algned.

            )





plt.show()

df_region=pd.read_csv("../input/tourists-by-region/international-tourist-arrivals-by-world-region.csv")

df_region.head()
df_region["No of tourist in million"]=df_region[" (arrivals)"]/1000000

df_region.rename(columns={"Entity":"Region"},inplace=True)

df_region.drop(columns=["Code"," (arrivals)"],inplace=True)

df_region.set_index("Year",inplace=True)

df_region.tail()
colors_list = ['red','lightgreen','lightcoral','lightskyblue','yellow']

explode_list = [0.1,0.05,0,0,0.1]

df_region[df_region.index==2018]['No of tourist in million'].plot(kind='pie',

                                                                 figsize=(10,6),autopct='%1.1f%%',pctdistance=1.11,

                                                                 colors=colors_list,explode = explode_list,labels=None,

                                                                 startangle=90,shadow=True)

plt.title('Region-wise tourists distribution for Y2018 ',y=1.12)

plt.legend(df_region[df_region.index==2018].Region,loc = 'upper left')

plt.axis('equal')

plt.show()
fig,ax=plt.subplots(figsize=(4,2.5),dpi=144)

color=plt.cm.Dark2(range(5))

y=df_region[df_region.index==2018]["Region"]

width=df_region[df_region.index==2018]["No of tourist in million"]

ax.barh(y=y,width=width,color=color)

ax.set_title("Region-wise tourists distribution for Y2018")

ax.set_xlabel("tourist in million")

def nice_axes(ax): #so that we don't have set grid,facecolor etc everytime.

    ax.set_facecolor('.8')  # 0 to 1 ->black to white resp

    ax.tick_params(labelsize=8, length=0)

    ax.grid(True, axis='x', color='white')

    ax.set_axisbelow(True)                  # make it false and see change 

    [spine.set_visible(False) for spine in ax.spines.values()]  # make it true and see change

    

nice_axes(ax)

fig



years=list(df_region.index.unique())
from IPython.display import HTML

from matplotlib.animation import FuncAnimation

colors=plt.cm.Dark2(range(5))

def bar_chart(year):

    ax.clear()

    y=df_region[df_region.index==year]["Region"]

    width=df_region[df_region.index==year]["No of tourist in million"]

    ax.barh(y=y,width=width,color=color)



    ax.set_title(year)

    ax.set_xlabel("tourist in million")

    nice_axes(ax)

fig, ax = plt.subplots(figsize=(10,6))

animator = FuncAnimation(fig, bar_chart, frames=years)

HTML(animator.to_jshtml())

    
df_raw
import plotly.express as px

fig = px.choropleth(df_raw, locations="Code", color="No of Tourist in million",

                    color_continuous_scale="greens" ,hover_name="Country", 

                    animation_frame="Year",range_color=[0,100])

fig.update_layout(

    title_text='Choropleth map for tourist by country',

    geo=dict(

        showframe=False,

        showcoastlines=False,

        projection_type='equirectangular'

    ),

    annotations = [dict(

        x=0.55,

        y=0.1,

        xref='paper',

        yref='paper',

        text='Source: <a href="https://ourworldindata.org/tourism">\Our World in Data</a>',

        showarrow = False

    )]

)

fig.show()
df_india = df_raw[df_raw['Country'] == 'India']

df_india.set_index('Year',inplace=True)

df_india.head()
fig = px.line(df_india,x= df_india.index,y='No of Tourist in million')

fig.update_layout(title="Line plot <br>" +

           "<i>Number of tourist visiting india </i> <br>",

            title_font_size=20,paper_bgcolor='black',title_font_color = 'yellow',title_font_family='Aerial',

            font={'color':'white','family':'Times New Roman','size':15},

            height = 500,

            template = 'plotly_white'

                 ),

fig.add_layout_image(

        dict(

            source="https://images.yourstory.com/cs/wordpress/2015/05/Yourstory_India_tourism.jpg?fm=png&auto=format",

            xref='x',

            yref='y',

            x=1995,

            y=16,

            sizex=22,

            sizey=16,

            sizing='stretch',

            opacity=0.5,

            layer="below")),

fig.update_xaxes(showgrid=True,linecolor='black',linewidth=2)

fig.update_yaxes(linecolor='black',linewidth=2)


India = pd.read_html('https://en.wikipedia.org/wiki/Tourism_in_India')[3]

India
India.drop(index=[10,12],inplace=True)
fig = px.pie(India,values ='Share in\xa0%',names='Country')

fig.update_layout(title='Pie Chart<br>'+"<i>Top 10 countries visiting India<i>",

                 title_font_family ='Times New Roman')
India_earning = pd.read_html("https://en.wikipedia.org/wiki/Tourism_in_India")[2]

India_earning
fig = px.bar(India_earning,x='Year',y='Earnings (₹ crores)')

fig.update_layout(template = 'plotly_white',

                 title = 'Earning from tourism in India',

                  title_font_family = 'Times New Roman',

                 title_font_size = 28),

fig.show()
India_states = pd.read_html("https://en.wikipedia.org/wiki/Tourism_in_India")[5]

India_states
India_states.drop(index=[10,12],inplace=True)
px.pie(India_states,values='Number',names='State/Union Territory',title= "state-wise foreign tourist visits in 2017")
#Folium
import folium

m=folium.Map(zoom_start=7,location=[48.8566,2.3522])

m
from folium import IFrame

import os

import base64

df = pd.read_excel('../input/francetourist-sites-listed-in-unesco-w-heritage/France_tourist sites.xlsx')

df.columns.values
df.drop(['Unnamed: 1','Unnamed: 3','Unnamed: 5'],axis=1,inplace=True)
df.head()
l=df.iloc[1,2].split(",")

l
float(l[0].split("(")[1])
html='<img src="data:image/jpg;base64,{}">'.format #see basic tutorials of html 

picture=base64.b64encode(open("../input/images/1.jpg","rb").read()).decode()

iframe =IFrame(html(picture),width=150+20,height=100+20)

popup= folium.Popup(iframe,max_width=200)

x=float(l[0].split("(")[1])

y=float(l[1].split(")")[0])

loc=[x,y]

folium.Marker(location=loc,

             popup=popup,tooltip=df.iloc[0,0]).add_to(m)

m
x_list=[]

y_list=[]

for i,row in df.iterrows():

    x=float(row["(lat,long)"].split(",")[0].split('(')[1])

    y=float(row['(lat,long)'].split(",")[1].split(')')[0])

    x_list.append(x)

    y_list.append(y)

    
df['X']=x_list

df['Y']=y_list

df
df["IMG_LOC"] = df["IMG_LOC"].apply(lambda x: "../input/images/"+ x)

df
df.drop([3,4,7,13,19],inplace = True)
for index,row in df.iterrows():

    html='<img src="data:image/jpg;base64,{}">'.format #see basic tutorials of html 

    picture=base64.b64encode(open(row["IMG_LOC"],"rb").read()).decode()

    iframe =IFrame(html(picture),width=150+20,height=120+20)

    popup= folium.Popup(iframe,max_width=250)

    

    folium.Marker(location=[row["X"],row["Y"]],

                 tooltip=row["Property"],

                  popup=popup,

                 icon=folium.Icon(color="red",icon="info-sign")).add_to(m)

m