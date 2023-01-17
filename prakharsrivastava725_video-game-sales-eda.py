import numpy as np

import pandas as pd 

import missingno as mn



import plotly.express as px

import plotly.graph_objects as go

from wordcloud import WordCloud, ImageColorGenerator

import matplotlib.pyplot as plt



import warnings                       

warnings.filterwarnings("ignore")


videoGameData = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv",index_col=0)
data = videoGameData.copy()
data.head()
data.shape
data.size
data.describe()
data.info()
data.duplicated().sum()
data.drop_duplicates(keep = 'first',inplace = True)
data.isnull().mean() * 100
mn.matrix(data)
data.dropna(inplace=True)
data['Year']=data['Year'].astype('int64')
GamesRev = data.groupby('Name')['Global_Sales','NA_Sales','EU_Sales','JP_Sales','Other_Sales'].sum().sort_values('Global_Sales',ascending = False)[:10]

GamesRev.drop('Global_Sales',axis=1,inplace=True)

fig = px.bar(GamesRev,

             labels={

                     "variable": "Region",

                     "value": "Sales(in Million)",

                     "Name": "Game"},

            title = 'Top Grossing Game in different Region')

fig.update_layout(

    title={'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    titlefont=dict(size =28))

fig.update

fig.show()
region =['NA_Sales','EU_Sales','JP_Sales','Other_Sales']

sales = [data[i].sum() for i in region]

fig = px.pie(names=region,values=sales)

fig.update_traces(rotation=90, pull=[0.06,0.06,0.06,0.06], textinfo="percent+label")

fig.show()
publisher = data['Publisher'].value_counts()

publisher2 = data.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False)[:10].reset_index()

publisher = publisher.reindex(index=publisher2['Publisher'])

publisher2['Games Published'] = publisher.values[:10]

fig = px.bar(publisher2 , x =publisher2['Publisher'],y=publisher2['Global_Sales'],color=publisher2['Games Published'],

             labels={"Global_Sales": "Global Sales"},

            title = 'Top Grossing Publisher with number of Publications')

fig.update_layout(

    title={'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    titlefont=dict(size =28))

fig.show()
topPublisher = publisher2['Publisher'][:5]

publisher = data.groupby(['Publisher'])

fig = go.Figure()

for pub in topPublisher:

    pubget = publisher.get_group(pub).groupby('Name')['Global_Sales'].sum().reset_index().sort_values('Global_Sales',ascending = False)[:10]

    fig.add_trace(

    go.Bar(x=pubget['Name'],

            y=pubget['Global_Sales'],

           name=pub,

           visible= True if pub == 'Nintendo' else False,

          marker={'color': pubget['Global_Sales'],'colorscale': 'Portland'}))

    fig.update_layout(xaxis_title="Games",

    yaxis_title="Global Sales")

fig.update_layout(

    updatemenus=[

        dict(

            type="buttons",

            direction="right",

            active=0,

            x=0.85,

            y=1.1,

            buttons=list([

                dict(label=topPublisher[0],

                     method="update",

                     args=[{"visible": [True, False,False, False, False]},

                           {"title": "Top 10 Games per Publisher"}]),

                dict(label=topPublisher[1],

                     method="update",

                     args=[{"visible": [False,True, False, False, False]},

                           {"title": "Top 10 Games per Publisher"}]),

                dict(label=topPublisher[2],

                     method="update",

                     args=[{"visible": [False,False, True, False, False]},

                           {"title": "Top 10 Games per Publisher"}]),

                dict(label=topPublisher[3],

                     method="update",

                     args=[{"visible": [False,False, False, True, False]},

                           {"title": "Top 10 Games per Publisher"}]),

                dict(label=topPublisher[4],

                     method="update",

                     args=[{"visible": [False,False, False, False, True]},

                           {"title": "Top 10 Games per Publisher"}]),

            ]),

        )

    ])



# Set title

fig.update_layout(

    title_text="Top 10 Games per Publisher",

     xaxis_domain=[0.05, 1.0],

    title={'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    titlefont=dict(size =28),

    annotations=[

        dict(text="Publisher :", showarrow=False,

                             x=0.2, y=1.1, yref="paper", align="left")

    ]

)





fig.show()
text = list(set(data['Genre']))

plt.rcParams['figure.figsize'] = (15,15)

wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="black").generate(str(text))



plt.imshow(wordcloud,interpolation="bilinear")

plt.axis("off")

plt.show()
genre = data['Genre'].value_counts()

fig = px.pie(genre, values = genre.values , names=genre.index,title= "Genre Distribution")

fig.update_traces(textinfo="percent+label")

fig.update_layout(title={'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    titlefont=dict(size =28))

fig.show()
genreSales = data.groupby('Genre')['Global_Sales'].sum()

fig = px.bar(genreSales,orientation='h',labels={"value": "Global Sales","variable":"region"},title = 'Top Grossing Genre')

fig.update_layout(title={'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    titlefont=dict(size =28))

fig.show()
text = list(set(data['Platform']))

plt.rcParams['figure.figsize'] = (15,15)

wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="black").generate(str(text))



plt.imshow(wordcloud,interpolation="bilinear")

plt.axis("off")

plt.show()
platform = data['Platform'].value_counts()[:10]

fig = px.pie(platform,names=platform.index,values=platform.values,hole=.3,title='Platform Distribution')

fig.update_traces(textinfo="percent+label",pull=[0.1,0.2,0.15,0.06,0.06,0.06,0.06,0.06,0.06,0.06])

fig.update_layout(title={'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    titlefont=dict(size =28))

fig.show()
platform = data.groupby(['Year','Platform'])['Global_Sales'].sum().reset_index()

platform = platform[(platform['Year']>=2006) & (platform['Year']<=2015)]

platform = platform.loc[platform['Platform'].isin(['DS', 'PS2', 'PS3', 'Wii', 'X360'])]

fig = px.bar(platform,x='Platform',y='Global_Sales',labels={"Global_Sales": "Global Sales"},

             title = 'Top Grossing Platform over 10 Years',animation_frame='Year', 

           animation_group='Platform', color='Platform')

fig.update_layout(title={'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    titlefont=dict(size =28))

fig.show()
yearSales = data.groupby('Year')['Global_Sales','NA_Sales','EU_Sales','JP_Sales','Other_Sales'].sum()

fig = px.line(yearSales,title='Region-wise Sales Distribution over Years',

              labels={"value": "Sales(in Million)","variable":"Region"} )

fig.update_layout(title={'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    titlefont=dict(size =28))

fig.show()
year = data['Year'].value_counts()

year2 = data.groupby('Year')['Global_Sales'].sum().sort_values(ascending=False).reset_index()

year = year.reindex(index=year2['Year'])

year2['Games Published'] = year.values

fig = px.bar(year2 , x =year2['Year'],y=year2['Global_Sales'],color=year2['Games Published'],

             title='Sales Relation with Number of Game Published',labels={"Global_Sales":"Global Sales"})

fig.update_layout(title={'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    titlefont=dict(size =28))

fig.show()