# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



import plotly.express as px

import plotly.graph_objects as go



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")

data.head()
data.columns
# lowercasing columns

data.columns = [each.lower() for each in data.columns]  

# combining seperate words with "_" in columns

data.columns = [each.split()[0]+"_"+each.split()[1]  if len(each.split())==2 else each for each in data.columns]

data.columns
# renaming column "size" to "sizes"

data.rename({"size":"sizes"}, inplace=True, axis=1)

data.info()
# Dropping some columns from dataframe

df = data.drop(['genres', 'last_updated', 'current_ver', 'android_ver'],axis=1)

df.info()
# dropping NaN values from df

df.dropna(inplace=True)

df.info()
# change type of column "sizes" to int

# there are **M and **k format values, "." and "Varies with device" rows in "sizes" column

# at first: those values to be replaced 

df.sizes.value_counts()

df.sizes.unique()
# replacing some values on "sizes" column as below:

# replacing:

# "k" with "",

# "M" with "000",

# "." with "",

# "Varies with device" with "0"



df.sizes = [each.replace("k","").replace("M","000").replace(".","").replace("Varies with device","0") for each in df.sizes]



# changing type to integer 

df.sizes = df.sizes.astype(int)



# type of "sizes" can be changed to integer also with int() method as below 

# df.sizes = [int(each.replace("k","").replace("M","000").replace(".","").replace("Varies with device","0")) for each in df.sizes]



df.info()
# changing type of "reviews" to int

df.reviews = df.reviews.astype(int)

df.info()
# changing type of "installs"to int

# There are "," and "+" forms in data, at first those forms to be replaced

df.installs.unique()

# In "installs" below replacements to be done:

# "," will be replaced with "",

# "+" will be replaced with ""



df.installs = [int(each.replace(",","").replace("+","")) for each in df.installs] 



df.info()
# for "price" column: "$" and "." to be replaced and type to be changed to float

df.price.unique()
df.price = [float(each.replace("$", "")) for each in df.price] 

df.info()
df.head()
# creating new index from descending rating values in order to sort df

new_index = df.rating.sort_values(ascending = False).index.values

new_index
# sorting df reindexing data according to new_index

sorted_data = df.reindex(new_index)

sorted_data.head()
# VISUALISATION: BUBBLE CHART with plotly express (df that has ratings grater than 3.5)

# x axis --> category

# y axis --> rating

# bubble size --> installs

# bubble color --> reviews

# hover_name --> app



df_ratingOver35 = sorted_data[sorted_data.rating > 3.5] #df that has ratings grater than 3.5

df_ratingOver35.tail()
# VISUALIZATION  --- BUBBLE CHART



size = df_ratingOver35.installs   # Setting bubble sizes as "size" and defining it as column "installs"



# creating trace for bubble chart

trace = go.Scatter(

    x=df_ratingOver35.category,  # setting x axis

    y=df_ratingOver35.rating,    # setting y axis

    mode='markers',              

    text = df_ratingOver35.app,  # assigning "app" to text argument to show up in hover

    customdata = df_ratingOver35.reviews.values,  # assigning "reviews" to customdata argument to show up in hover

    hovertemplate =                          # creating hovertemplate

    "<b>%{text}</b><br><br>" +  # showing "text" in hover: <b> for starting bold, </b> for ending bold, <br> for line break 

    "Category: %{x}<br>" + 

    "Rating: %{y}<br>" + 

    "Installation: %{marker.size:,}<br>" + 

    "Reviews: %{customdata:,}"+

    "<extra></extra>",

    marker=dict(

        size=size,  # setting size of marker

        sizemode='area',  # setting sizemode to area in order to see the hovertext as soon as cursor is in the buble area

        sizeref=2.*max(size)/(40.**2), # referensing the buble sizes

        sizemin=0.5, # setting min buble size

        color = df_ratingOver35.reviews,  # setting "color" argument for reviews column

        opacity=0.3,  # setting opacity of bubbles

        showscale=True,  # showing color bar 

        colorscale = "bluered",  # setting colorpalett of colorbar

        colorbar = dict(title="Reviews")  # setting title of colorbar

    )

)



data =[trace]



layout = dict(title = "Ratings of G.Playstore Apps over 3.5",

              plot_bgcolor="white",    # setting background of plot

              xaxis = dict(title = "APP Category", ticklen=5 , zeroline= False, tickangle = -45,

                           titlefont=dict(color="red",

                                         size=18,

                                         family="Courier New, monospace")),

              yaxis = dict(title = "Rating", ticklen = 5, zeroline = False,

                           titlefont= dict(color="blue",

                                          size= 18,

                                          family ="Courier New, monospace")))

      

            

fig = go.Figure(data=data, layout= layout)

plt.savefig("Ratings of G.Playstore Apps over 3.5.png")

fig.show()



# color codes-->

#'aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance','blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg','brwnyl',

#'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl','darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric','emrld', 

#'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys','haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet','magenta', 'magma', 

#'matter', 'mint', 'mrybm', 'mygbm', 'oranges','orrd', 'oryel', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg','plasma', 'plotly3', 

#'portland', 'prgn', 'pubu', 'pubugn', 'puor','purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdgy','rdpu', 'rdylbu', 

#'rdylgn', 'redor', 'reds', 'solar', 'spectral','speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn', 'tealrose','tempo', 'temps', 

#'thermal', 'tropic', 'turbid', 'twilight','viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'
# Visualisation --- Scatter --- "Rating vs Review of Applications"

trace1 = go.Scatter(

                    x=df.reviews,

                    y=df.rating,

                    mode="markers",

                    name="Ratings",

                    marker= dict(

                        color = df.installs,

                        showscale = True,

                        colorbar = dict(title="Installs")

                        ),

                    text=sorted_data.app

                   )





data = [trace1]



layout = dict(title = "Rating vs Review of Applications",

              xaxis=dict(title="Review"),

              yaxis=dict(title="Rating"),

              plot_bgcolor="white"              

             )



fig=go.Figure(data=data, layout=layout)

plt.savefig("Rating vs Review of Applications.png")



fig.show()

# Visualisation :  Rating Histogram

trace1 = go.Histogram(x=df.rating,

                     name="Rating",

                     marker = dict(

                         color="blue", 

                         line=dict(color="rgb(0,0,0)",width=1.5)

                         ),

                     )                     

data=[trace1]

layout= dict(title="Rating vs Frequency of Applications",

            xaxis=dict(title="Ratings", showgrid=True, gridcolor="grey",zeroline=True),

            yaxis=dict(title="Frequency", showgrid=True, gridcolor="grey",zeroline=True),

            plot_bgcolor="rgb(255,255,232)",

            )

fig=go.Figure(data=data, layout = layout)

plt.savefig("Rating vs Frequency of Applications.png")

fig.show()
trace1 = go.Scatter(x=sorted_data.category,

                    y=sorted_data.rating,

                    mode="markers",

                    marker= dict(

                                color = sorted_data.installs,

                                showscale = True,

                                colorbar = dict(title="Installs"),

                                opacity = 0.8,

                                colorscale = "bluered_r",

                                size=5

                                ),

                    text = sorted_data.app)

layout= dict(title="Category vs Rating",

            xaxis=dict(title="Category", showgrid=True, gridcolor="grey",zeroline=True, tickangle = -45),

            yaxis=dict(title="Rating", showgrid=True, gridcolor="grey",zeroline=True),

            plot_bgcolor="rgb(255,255,240)",

            )

                                  

data = [trace1] 



fig = go.Figure(data=data, layout = layout)

plt.savefig("Category vs Rating.png")

fig.show()
category_list = list(sorted_data.category.unique())

mean_list=[]

installs_list = []

review_list=[]

total_installs_list=[]

app_qty_list=[]



for each in category_list:

    mean_rating= sum(sorted_data[sorted_data.category==each].rating)/len(sorted_data[sorted_data.category==each].rating)

    mean_list.append(mean_rating)

    mean_installs =sum(sorted_data[sorted_data.category==each].installs)/len(sorted_data[sorted_data.category==each].installs)/10000000

    installs_list.append(mean_installs)

    mean_reviews =sum(sorted_data[sorted_data.category==each].reviews)/len(sorted_data[sorted_data.category==each].reviews)/600000

    review_list.append(mean_reviews)

    total_installs = sum(sorted_data[sorted_data.category==each].installs)

    total_installs_list.append(total_installs)

    app_qty = len(sorted_data[sorted_data.category==each].app)

    app_qty_list.append(app_qty)  



new_df = pd.DataFrame(list(zip(category_list, mean_list, installs_list, review_list,total_installs_list, app_qty_list )), 

                      columns=["category","mean_rating","mean_installs","mean_reviews","total_installs","app_qty"])

new_df
trace1 = go.Scatter(

                    x = new_df.category,

                    y = new_df.mean_rating,

                    mode = "lines+markers",

                    name = "Mean Rating",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    )



trace2 = go.Scatter(

                    x = new_df.category,

                    y = new_df.mean_installs,

                    mode = "lines+markers",

                    name = "Mean Installs",

                    marker = dict(color = 'rgba(49, 29, 226, 0.8)'),

                    )



trace3 = go.Scatter(

                    x = new_df.category,

                    y = new_df.mean_reviews,

                    mode = "lines+markers",

                    name = "Mean Reviews",

                    marker = dict(color = 'rgba(255, 0, 128, 0.8)'),

                    )



data = [trace1, trace2, trace3]

layout = dict(title = 'Category vs Reviews-Installs-Rating',

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False, tickangle=-45)

             )

fig = go.Figure(data = data, layout = layout)

plt.savefig('Category vs Reviews-Installs-Rating.png')

fig.show()
slice1 = new_df.loc[:, ["category", "total_installs"]]

new_index = slice1.total_installs.sort_values(ascending=False).index.values

sorted_slice1= slice1.reindex(new_index)

sorted_slice1.head()
trace =go.Bar(

              x = sorted_slice1.category,

              y = sorted_slice1.total_installs,

              marker = dict(color="rgb(133,239,116)",

                            line=dict(color="rgb(0,0,0)",

                            width=1.5)

                           ),

                )                     

data=[trace]

layout= dict(title="Category vs Istallation",

            xaxis=dict(title="Category", showgrid=True, gridcolor="grey",zeroline=True, tickangle = -45),

            yaxis=dict(title="Installation", showgrid=True, gridcolor="grey",zeroline=True),

            plot_bgcolor="rgb(255,255,232)",

            )

fig=go.Figure(data=data, layout = layout)

plt.savefig("Category vs Istallation.png")

fig.show()

             
slice2 = new_df.loc[:, ["category","app_qty"]]

new_index = slice2.app_qty.sort_values(ascending=False).index.values

sorted_slice2=slice2.reindex(new_index)

sorted_slice2.head()
trace =go.Bar(

              x = sorted_slice2.category,

              y = sorted_slice2.app_qty,

              marker = dict(color="rgb(255,190,125)",

                            line=dict(color="rgb(0,0,0)",

                            width=1.5)

                           ),

                )                     

data=[trace]

layout= dict(title="Category vs App QTY",

            xaxis=dict(title="Category", showgrid=True, gridcolor="grey",zeroline=True, tickangle = -45),

            yaxis=dict(title="App QTY", showgrid=True, gridcolor="grey",zeroline=True),

            plot_bgcolor="rgb(255,255,232)",

            )

fig=go.Figure(data=data, layout = layout)

plt.savefig("Category vs App QTY.png")

fig.show()