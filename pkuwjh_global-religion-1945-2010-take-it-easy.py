import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from wordcloud import WordCloud, STOPWORDS

from scipy.misc import imread

import base64
# Load in the .csv files as three separate dataframes

Global = pd.read_csv('../input/global.csv') # Put to caps or else name clash

national = pd.read_csv('../input/national.csv')

regional = pd.read_csv('../input/regional.csv')
# Print the top 3 rows

regional.head(3)
print(regional['region'].unique())
#fig = plt.figure(figsize=(8, 5))

fig, axes = plt.subplots(nrows=1, ncols=3)

colormap = plt.cm.inferno_r

# fig = plt.figure(figsize=(20, 10))

# plt.subplot(121)

christianity_year = regional.groupby(['year','region']).christianity_all.sum()

christianity_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False,ax= axes[0],figsize=(11.5,4.5) , legend=False)

axes[0].set_title('Christianity Adherents',y=1.08,size=10)



# plt.subplot(122)

islam_year = regional.groupby(['year','region']).islam_all.sum()

islam_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, ax= axes[1], legend= False)

axes[1].set_title('Islam Adherents',y=1.08,size=10)



judaism_year = regional.groupby(['year','region']).judaism_all.sum()

judaism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, ax= axes[2])

axes[2].legend(bbox_to_anchor=(-1.7, -0.3, 1.8, 0.1), loc=10,prop={'size':12},

           ncol=5, mode="expand", borderaxespad=0.)

axes[2].set_title('Judaism Adherents',y=1.08,size=10)



plt.tight_layout()

plt.show()
#fig = plt.figure(figsize=(8, 5))

fig, axes = plt.subplots(nrows=1, ncols=3)

colormap = plt.cm.inferno

# fig = plt.figure(figsize=(20, 10))

# plt.subplot(121)

christianity_year = regional[regional['region'] != 'Asia'].groupby(['year','region']).hinduism_all.sum()

christianity_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False,ax= axes[0],figsize=(13.5,6.5) , legend=False)

axes[0].set_title('Hindusim Adherents',y=1.08,size=12)



# plt.subplot(122)

islam_year = regional[regional['region'] != 'Asia'].groupby(['year','region']).sikhism_all.sum()

islam_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, ax= axes[1], legend= False)

axes[1].set_title('Sikhism Adherents',y=1.08,size=12)



judaism_year = regional[regional['region'] != 'Asia'].groupby(['year','region']).jainism_all.sum()

judaism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, ax= axes[2])

axes[2].legend(bbox_to_anchor=(-1.7, -0.3, 2, 0.1), loc=10,prop={'size':12},

           ncol=5, mode="expand", borderaxespad=0.)

axes[2].set_title('Jainism Adherents',y=1.08,size=12)



plt.tight_layout()

plt.show()
#fig = plt.figure(figsize=(8, 5))

fig, axes = plt.subplots(nrows=1, ncols=3)

colormap = plt.cm.Purples

# fig = plt.figure(figsize=(20, 10))

# plt.subplot(121)

christianity_year = regional[regional['region'] != 'Asia'].groupby(['year','region']).buddhism_all.sum()

christianity_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False,ax= axes[0],figsize=(13.5,6.5) , legend=False)

axes[0].set_title('Buddhist Adherents',y=1.08,size=12)



# plt.subplot(122)

islam_year = regional[regional['region'] != 'Asia'].groupby(['year','region']).taoism_all.sum()

islam_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, ax= axes[1], legend= False)

axes[1].set_title('Taoist Adherents',y=1.08,size=12)



judaism_year = regional[regional['region'] != 'Asia'].groupby(['year','region']).shinto_all.sum()

judaism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, ax= axes[2])

axes[2].legend(bbox_to_anchor=(-1.7, -0.3, 2, 0.1), loc=10,prop={'size':12},

           ncol=5, mode="expand", borderaxespad=0.)

axes[2].set_title('Shinto Adherents',y=1.08,size=12)



plt.tight_layout()

plt.show()
national.head(3)
# Create a dataframe with only the 2010 data

national_2010 = national[national['year'] == 2010]

# Extract only the parent religion with the "_all" and ignoring their denominations for now

religion_list = []

for col in national_2010.columns:

    if '_all' in col:

        religion_list.append(col)

metricscale1=[[0.0,"rgb(20, 40, 190)"],[0.05,"rgb(40, 60, 190)"],[0.25,"rgb(70, 100, 245)"],[0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]]

data = [ dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = 'Viridis',

        reversescale = True,

        showscale = True,

        locations = national_2010['state'].values,

        z = national_2010['christianity_all'].values,

        locationmode = 'country names',

        text = national_2010['state'].values,

        marker = dict(

            line = dict(color = 'rgb(200,200,200)', width = 0.5)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = 'Number of Christian Adherents')

            )

       ]



layout = dict(

    title = 'Christian Adherents in 2010',

    geo = dict(

        showframe = True,

        showocean = True,

        oceancolor = 'rgb(0,0,0)',

        #oceancolor = 'rgb(222,243,246)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )

fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap2010')



data2 = [ dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = 'Viridis',

        reversescale = True,

        showscale = True,

        locations = national_2010['state'].values,

        z = national_2010['islam_all'].values,

        locationmode = 'country names',

        text = national_2010['state'].values,

        marker = dict(

            line = dict(color = 'rgb(200,200,200)', width = 0.5)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = 'Number of Islamic Adherents')

            )

       ]



layout2 = dict(

    title = 'Islamic Adherents in the Year 2010',

    geo = dict(

        showframe = True,

        showocean = True,

        oceancolor = 'rgb(28,10,16)',

        #oceancolor = 'rgb(222,243,246)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )

fig = dict(data=data2, layout=layout2)

py.iplot(fig, validate=False, filename='worldmap2010') 



data3 = [ dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = 'Viridis',

        reversescale = True,

        showscale = True,

        locations = national_2010['state'].values,

        z = national_2010['judaism_all'].values,

        locationmode = 'country names',

        text = national_2010['state'].values,

        marker = dict(

            line = dict(color = 'rgb(200,200,200)', width = 0.5)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = 'Number of Judaism Adherents')

            )

       ]



layout3 = dict(

    title = 'Judaism Adherents in the Year 2010',

    geo = dict(

        showframe = True,

        showocean = True,

        oceancolor = 'rgb(28,10,16)',

        #oceancolor = 'rgb(222,243,246)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )

fig = dict(data=data3, layout=layout3)

py.iplot(fig, validate=False, filename='worldmap2010') 
metricscale1=[[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],[0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]]

# Mercator plots for the Buddhism

data = [ dict(

        type = 'choropleth',

        locations = national_2010['code'],

        z = national_2010['buddhism_all'],

        text = national_2010['code'].unique(),

        colorscale = metricscale1,

        autocolorscale = False,

        reversescale = True,

        marker = dict(

            line = dict (

                color = 'rgb(200,200,200)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            title = 'Number of Buddhist Adherents'),

      ) ]



layout = dict(

    title = 'Spread of Buddhist adherents in 2010',

    geo = dict(

        scope = 'asia',

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(0,0,50)',

#         oceancolor = 'rgb(232,243,246)',

        #oceancolor = ' rgb(28,107,160)',

        showcoastlines = True,

        projection = dict(

            type = 'Mercator'

        )

    )

)



fig = dict( data=data, layout=layout )

py.iplot( fig, validate=False, filename='d3-world-map' )



# Mercator plots for Hinduism

data1 = [ dict(

        type = 'choropleth',

        locations = national_2010['code'],

        z = national_2010['hinduism_all'],

        text = national_2010['code'].unique(),

        colorscale = metricscale1,

        autocolorscale = False,

        reversescale = True,

        marker = dict(

            line = dict (

                color = 'rgb(200,200,200)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            title = 'Number of Hinduism Adherents'),

      ) ]



layout1 = dict(

    title = 'Spread of Hinduism adherents in 2010',

    geo = dict(

        scope = 'asia',

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(0,0,50)',

#         oceancolor = 'rgb(232,243,246)',

        #oceancolor = ' rgb(28,107,160)',

        showcoastlines = True,

        projection = dict(

            type = 'Mercator'

        )

    )

)



fig = dict( data=data1, layout=layout1 )

py.iplot( fig, validate=False, filename='world-map' )



# Mercator plots for Shinto

data2 = [ dict(

        type = 'choropleth',

        locations = national_2010['code'],

        z = national_2010['shinto_all'],

        text = national_2010['code'].unique(),

        colorscale = metricscale1,

        autocolorscale = False,

        reversescale = True,

        marker = dict(

            line = dict (

                color = 'rgb(200,200,200)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            title = 'Number of Shinto Adherents'),

      ) ]



layout2 = dict(

    title = 'Spread of Shinto adherents in 2010',

    geo = dict(

        scope = 'asia',

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(0,0,50)',

#         oceancolor = 'rgb(232,243,246)',

        #oceancolor = ' rgb(28,107,160)',

        showcoastlines = True,

        projection = dict(

            type = 'Mercator'

        )

    )

)



fig = dict( data=data2, layout=layout2 )

py.iplot( fig, validate=False, filename='world-map2' )
# Although I know that Thailand, Cambodia, Lao, Vietnam, Malaysia, Singapore, Philippines, Indonesia, Brunei

# are South-East Asian countries to be exact, I decided to group these together

East_asian_countries = ['China', 'Mongolia', 'Taiwan', 'North Korea',

       'South Korea', 'Japan','Thailand', 'Cambodia',

       'Laos', 'Vietnam',  'Malaysia', 'Singapore',

       'Brunei', 'Philippines', 'Indonesia']



South_asian_countries = ['India', 'Bhutan', 'Pakistan', 'Bangladesh',

       'Sri Lanka', 'Nepal']



East_european_countries = [

    'Poland', 'Czechoslovakia', 'Czech Republic', 'Slovakia','Malta', 'Albania', 'Montenegro', 'Macedonia',

       'Croatia', 'Yugoslavia', 'Bosnia and Herzegovina', 'Kosovo',

       'Slovenia', 'Bulgaria', 'Moldova', 'Romania','Estonia', 'Latvia', 'Lithuania', 'Ukraine', 'Belarus',

       'Armenia', 'Georgia',

]



West_european_countries = [

    'United Kingdom', 'Ireland', 'Netherlands', 'Belgium', 'Luxembourg',

       'France', 'Liechtenstein', 'Switzerland', 'Spain', 'Portugal', 'Germany','Greece', 'Italy'

]



Africa = ['Mali', 'Senegal',

       'Benin', 'Mauritania', 'Niger', 'Ivory Coast', 'Guinea',

       'Burkina Faso', 'Liberia', 'Sierra Leone', 'Ghana', 'Togo',

       'Cameroon', 'Nigeria', 'Gabon', 'Central African Republic', 'Chad',

       'Congo', 'Democratic Republic of the Congo', 'Uganda', 'Kenya',

       'Tanzania', 'Burundi', 'Rwanda', 'Somalia']



South_america = ['Peru', 'Brazil',

       'Bolivia', 'Paraguay', 'Chile', 'Argentina', 'Uruguay','Colombia',

       'Venezuela']

#European_countries = ['United Kingdom', 'Ireland', 'Netherlands', 'Belgium', 'Luxembourg',

#       'France', 'Monaco', 'Liechtenstein', 'Switzerland', 'Spain',

#       'Andorra', 'Portugal', 'Germany', 'Poland', 'Austria', 'Hungary',

#       'Czechoslovakia', 'Czech Republic', 'Slovakia', 'Italy',

#       'San Marino', 'Malta', 'Albania', 'Montenegro', 'Macedonia',

#       'Croatia', 'Yugoslavia', 'Bosnia and Herzegovina', 'Kosovo',

#       'Slovenia', 'Greece', 'Cyprus', 'Bulgaria', 'Moldova', 'Romania',

#       'Russia', 'Estonia', 'Latvia', 'Lithuania', 'Ukraine', 'Belarus',

#       'Armenia', 'Georgia',  'Finland', 'Sweden', 'Norway',

#       'Denmark', 'Iceland',]
plt.figure(figsize=(10, 10))





# East Asian Numbers

christianity_year = national[ national['state'].isin(East_asian_countries) ].groupby(['year','state']).christianity_all.sum()

christianity_year.unstack().plot(kind='area',stacked=True,  colormap= 'cubehelix', grid=False,figsize=(10,8))

plt.title('Christanity in East Asia')

# Place a legend above this subplot, expanding itself to

# fully use the given bounding box.

plt.gca().legend_.remove()

plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,

            ncol=4, mode="expand", borderaxespad=0.)

plt.ylabel('Number of Christian adherents')

plt.show()



christianity_year = national[ national['state'].isin(Africa) ].groupby(['year','state']).christianity_all.sum()

christianity_year.unstack().plot(kind='area',stacked=True,  colormap= 'cubehelix_r', grid=False,figsize=(10,8))

plt.title('Christanity in Africa', size=12)

# Place a legend above this subplot, expanding itself to

# fully use the given bounding box.

plt.gca().legend_.remove()

plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,

            ncol=4, mode="expand", borderaxespad=0.)

plt.ylabel('Number of Christian adherents')

plt.show()





# South American Numbers

christianity_year = national[ national['state'].isin(South_america) ].groupby(['year','state']).christianity_all.sum()

christianity_year.unstack().plot(kind='area',stacked=True,  colormap= 'cubehelix_r', grid=False,figsize=(10,8))

plt.title('Christanity in South America')

# Place a legend above this subplot, expanding itself to

# fully use the given bounding box.

plt.gca().legend_.remove()

plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,

            ncol=4, mode="expand", borderaxespad=0.)

plt.ylabel('Number of Christian adherents')

plt.show()





christianity_year = national[ national['state'].isin(West_european_countries) ].groupby(['year','state']).christianity_all.sum()

christianity_year.unstack().plot(kind='area',stacked=True,  colormap= 'cubehelix', grid=False,figsize=(10,8))

plt.title('Christanity in Western Europe')

# Place a legend above this subplot, expanding itself to

# fully use the given bounding box.

plt.gca().legend_.remove()

plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,

            ncol=4, mode="expand", borderaxespad=0.)

plt.ylabel('Number of Christian adherents')

plt.show()
islam_year = national[ national['state'].isin(West_european_countries) ].groupby(['year','state']).islam_all.sum()

islam_year.unstack().plot(kind='area',stacked=True,  colormap= 'PuBuGn', grid=False,figsize=(10,8))

plt.title('Islam in Western Europe')

# Place a legend above this subplot, expanding itself to

# fully use the given bounding box.

plt.gca().legend_.remove()

plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,

            ncol=4, mode="expand", borderaxespad=0.)

plt.ylabel('Number of Islam adherents')

plt.show()



islam_year = national[ national['state'].isin(Africa) ].groupby(['year','state']).islam_all.sum()

islam_year.unstack().plot(kind='area',stacked=True,  colormap= 'PuBuGn_r', grid=False,figsize=(10,8))

plt.title('Islam in Africa')

# Place a legend above this subplot, expanding itself to

# fully use the given bounding box.

plt.gca().legend_.remove()

plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,

            ncol=4, mode="expand", borderaxespad=0.)

plt.ylabel('Number of Islam adherents')

plt.show()



islam_year = national[ national['state'].isin(South_america) ].groupby(['year','state']).islam_all.sum()

islam_year.unstack().plot(kind='area',stacked=True,  colormap= 'PuBuGn', grid=False,figsize=(10,8))

plt.title('Islam in South America')

# Place a legend above this subplot, expanding itself to

# fully use the given bounding box.

plt.gca().legend_.remove()

plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,

            ncol=4, mode="expand", borderaxespad=0.)

plt.ylabel('Number of Islam adherents')

plt.show()



islam_year = national[ national['state'].isin(East_asian_countries) ].groupby(['year','state']).islam_all.sum()

islam_year.unstack().plot(kind='area',stacked=True,  colormap= 'PuBuGn', grid=False,figsize=(10,8))

plt.title('Islam in Eastern Asia')

# Place a legend above this subplot, expanding itself to

# fully use the given bounding box.

plt.gca().legend_.remove()

plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,

            ncol=4, mode="expand", borderaxespad=0.)

plt.ylabel('Number of Islam adherents')

plt.show()
Budd_year = national[ national['state'].isin(South_asian_countries) ].groupby(['year','state']).buddhism_all.sum()

Budd_year.unstack().plot(kind='area',stacked=True,  colormap= 'viridis_r', grid=False,figsize=(10,8))

plt.title('Buddhism in South Asian countries')

# Place a legend above this subplot, expanding itself to

# fully use the given bounding box.

plt.gca().legend_.remove()

plt.legend(bbox_to_anchor=(-0.2, -0.7, 1.4, .5), loc=5,

            ncol=4, mode="expand", borderaxespad=0.)

plt.ylabel('Number of Buddhist adherents')

plt.show()



Budd_year = national[ national['state'].isin(West_european_countries) ].groupby(['year','state']).buddhism_all.sum()

Budd_year.unstack().plot(kind='area',stacked=True,  colormap= 'viridis_r', grid=False,figsize=(10,8))

plt.title('Buddhism in Western Europe')

# Place a legend above this subplot, expanding itself to

# fully use the given bounding box.

plt.gca().legend_.remove()

plt.legend(bbox_to_anchor=(-0.2, -0.7, 1.4, .5), loc=5,

            ncol=4, mode="expand", borderaxespad=0.)

plt.ylabel('Number of Buddhist adherents')

plt.show()



Budd_year = national[ national['state'].isin(Africa) ].groupby(['year','state']).buddhism_all.sum()

Budd_year.unstack().plot(kind='area',stacked=True,  colormap= 'viridis', grid=False,figsize=(10,8))

plt.title('Buddhism in Africa')

# Place a legend above this subplot, expanding itself to

# fully use the given bounding box.

plt.gca().legend_.remove()

plt.legend(bbox_to_anchor=(-0.2, -0.7, 1.4, .5), loc=5,

            ncol=4, mode="expand", borderaxespad=0.)

plt.ylabel('Number of Buddhist adherents')

plt.show()



Budd_year = national[ national['state'].isin(South_america) ].groupby(['year','state']).buddhism_all.sum()

Budd_year.unstack().plot(kind='area',stacked=True,  colormap= 'viridis_r', grid=False,figsize=(10,8))

plt.title('Buddhism in South America')

# Place a legend above this subplot, expanding itself to

# fully use the given bounding box.

plt.gca().legend_.remove()

plt.legend(bbox_to_anchor=(-0.2, -0.7, 1.4, .5), loc=5,

            ncol=4, mode="expand", borderaxespad=0.)

plt.ylabel('Number of Buddhist adherents')

plt.show()
# Stacked area plot of Hinduism in West Europe

Hin_year = national[ national['state'].isin(West_european_countries) ].groupby(['year','state']).hinduism_all.sum()

Hin_year.unstack().plot(kind='area',stacked=True,  colormap= 'gist_earth', grid=False,figsize=(10,8))

plt.title('Hinduism in West Europe')

# Place a legend above this subplot, expanding itself to

# fully use the given bounding box.

plt.gca().legend_.remove()

plt.legend(bbox_to_anchor=(-0.2, -0.7, 1.4, .5), loc=5,

            ncol=4, mode="expand", borderaxespad=0.)

plt.ylabel('Number of Hindu adherents')

plt.show()



Hin_year = national[ national['state'].isin(East_european_countries) ].groupby(['year','state']).hinduism_all.sum()

Hin_year.unstack().plot(kind='area',stacked=True,  colormap= 'gist_earth', grid=False,figsize=(10,8))

plt.title('Hinduism in East Europe')

# Place a legend above this subplot, expanding itself to

# fully use the given bounding box.

plt.gca().legend_.remove()

plt.legend(bbox_to_anchor=(-0.2, -0.7, 1.4, .5), loc=5,

            ncol=4, mode="expand", borderaxespad=0.)

plt.ylabel('Number of Hindu adherents')

plt.show()



Hin_year = national[ national['state'].isin(South_asian_countries) ].groupby(['year','state']).hinduism_all.sum()

Hin_year.unstack().plot(kind='area',stacked=True,  colormap= 'gist_earth', grid=False,figsize=(10,8))

plt.title('Hinduism in South Asia')

# Place a legend above this subplot, expanding itself to

# fully use the given bounding box.

plt.gca().legend_.remove()

plt.legend(bbox_to_anchor=(-0.2, -0.7, 1.4, .5), loc=5,

            ncol=4, mode="expand", borderaxespad=0.)

plt.ylabel('Number of Hindu adherents')

plt.show()



Hin_year = national[ national['state'].isin(East_asian_countries) ].groupby(['year','state']).hinduism_all.sum()

Hin_year.unstack().plot(kind='area',stacked=True,  colormap= 'gist_earth_r', grid=False,figsize=(10,8))

plt.title('Hinduism in East Asia')

# Place a legend above this subplot, expanding itself to

# fully use the given bounding box.

plt.gca().legend_.remove()

plt.legend(bbox_to_anchor=(-0.2, -0.7, 1.4, .5), loc=5,

            ncol=4, mode="expand", borderaxespad=0.)

plt.ylabel('Number of Hindu adherents')

plt.show()
# Prevalence of Confucianism in Asian countries

christianity_year = national[ national['state'].isin(East_asian_countries) ].groupby(['year','state']).confucianism_all.sum()

christianity_year.unstack().plot(kind='area',stacked=True,  colormap= 'hot_r', grid=False,figsize=(7,7))



#plt.figure(figsize=(10,5))

plt.title('Stacked area plot of the trend in Confucianism over the years', y=1.09)

# Place a legend above this subplot, expanding itself to

# fully use the given bounding box.

plt.gca().legend_.remove()

plt.legend(bbox_to_anchor=(-0.5, -0.7, 1.8, .5), loc=5,

            ncol=4, mode="expand", borderaxespad=0.)

plt.ylabel('Number of Confucianism adherents')

plt.show()
# Plotting Animism for both East and South Asian countries

christianity_year = national[ national['state'].isin(East_asian_countries) ].groupby(['year','state']).animism_all.sum()

christianity_year.unstack().plot(kind='area',stacked=True,  colormap= 'hot', grid=False,figsize=(10,8))

plt.title('Stacked Area plot of Animism over the years for Central-East-SE Asian countries', y =1.09)

# Place a legend above this subplot, expanding itself to

# fully use the given bounding box.

plt.gca().legend_.remove()

plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,

            ncol=4, mode="expand", borderaxespad=0.)

plt.ylabel('Number of Animism adherents')

plt.show()



christianity_year = national[ national['state'].isin(South_asian_countries) ].groupby(['year','state']).animism_all.sum()

christianity_year.unstack().plot(kind='area',stacked=True,  colormap= 'hot', grid=False,figsize=(10,8))

plt.title('Stacked Area plot of Animism over the years for South Asian countries', y=1.09)

# Place a legend above this subplot, expanding itself to

# fully use the given bounding box.

plt.gca().legend_.remove()

plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,

            ncol=4, mode="expand", borderaxespad=0.)

plt.ylabel('Number of Animism adherents')

plt.show()
'''

print ('this is my second time kaggle code')

'''