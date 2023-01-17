# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

import numpy as np

warnings.filterwarnings('ignore')

%matplotlib inline

sns.set(style='whitegrid')

import missingno as msmo



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'

from IPython.display import display
data = pd.read_csv('../input/googleplaystore.csv')

display(data.head(7))

print('Number of rows in dataset='+str(data.shape[0]))
#将数据中的日期转换成数字

data['Last Updated'] = pd.to_datetime(data['Last Updated'],format='%B %d, %Y',errors='coerce').astype('str')

def split_mul(data):

    try:

        data = list(map(int,data.split('-')))

        return data[0]+(data[1]*12)+data[2]

    except:

        return 'Nan'

data['Last Updated'] = [split_mul(x) for x in data['Last Updated']]





data["Android Ver"] = data['Android Ver'].str.split(n=1,expand=True)



def deal_with_abnormal_strings(data):

    data[data.str.isnumeric()==False]=-1

    data = data.astype(np.float32)

    return data



data.Installs = [x.strip().replace('+','').replace(',','') for x in data.Installs]

data.Size = [x.strip().replace('M','').replace(',','') for x in data.Size]

def covert_float(val):

    try:

        return float(val)

    except ValueError:

        try:

            val = val.split('.')

            return float(val[0]+'.'+val[1])

        except:

            return np.nan
data.head(7)
def plot_number_category():

    fig,ax = plt.subplots()

    fig.set_size_inches(15,7)

    fig.autofmt_xdate()

    countplot = sns.categorical.countplot(data.Category,ax=ax)

    plt.show(countplot)

plot_number_category()

top_cat = data.groupby('Category').size().reset_index(name='Count').nlargest(6,'Count')

display(top_cat)
data = data[data["Rating"]<=5]
data.Category.unique()
#object(string) values transform to float in Category Feature 

CategoryString = data["Category"]

categoryVal = data["Category"].unique()

categoryValCount = len(categoryVal)

category_dict = {}

for i in range(0,categoryValCount):

    category_dict[categoryVal[i]] = i

data["Category"] = data["Category"].map(category_dict).astype(int)
#Genres unique val

data["Genres"].unique()
#object(string) values transform to int in Genres Feature  

genresString = data["Genres"]

genresVal = data["Genres"].unique()

genresValCount = len(genresVal)

genres_dict = {}

for i in range(0,genresValCount):

    genres_dict[genresVal[i]] = i

data["Genres"] = data["Genres"].map(genres_dict).astype(int)
#get unique values in Contant Rating feature 

data['Content Rating'].unique()
#object(string) values transform to float in Content Rating Feature without nan

data['Content Rating'] = data['Content Rating'].map({'Everyone':0,'Teen':1,'Everyone 10+':2,'Mature 17+':3,

                                                     'Adults only 18+':4}).astype(float)
#clean 'M'and transform to float 

data['Reviews'] = [ float(i.split('M')[0]) if 'M'in i  else float(i) for i in data['Reviews']]
#clean 'M'and transform to float 

data["Size"] = [ float(i.split('M')[0]) if 'M' in i else float(0) for i in data["Size"]  ]
#clean '$' and transform to float 

data['Price'] = [ float(i.split('$')[1]) if '$' in i else float(0) for i in data['Price'] ] 
data.Installs.unique()
data["Installs"]
#deleted unnecessary features

data.drop(["Last Updated","Current Ver","Android Ver","App","Type"],axis=1,inplace=True)
#fill nan values

data["Rating"] = data.groupby("Category")["Rating"].transform(lambda x: x.fillna(x.mean()))

data["Content Rating"] = data[["Content Rating"]].fillna(method="ffill")
fig, ax = plt.subplots(figsize=(12,8))

ax = sns.countplot(x=CategoryString)

plt.xticks(rotation=90)

plt.show()
data[["Rating"]].plot(kind="hist",color="red",figsize=(8,8),bins=30)

plt.show()
fig,ax = plt.subplots(figsize=(25,10))

plt.scatter(x=genresString,y=data["Rating"],color="green",marker="o")

plt.xticks(rotation=90)

plt.grid()

plt.show()
fig,ax = plt.subplots(figsize=(12,8))

plt.scatter(x=CategoryString,y=data["Rating"],color="blue",marker="*",s=100)

plt.xticks(rotation=90)

plt.grid()

plt.show()
fig,ax = plt.subplots(figsize=(12,8))

plt.scatter(x=CategoryString,y=data["Reviews"],color="#F4511E",marker="^",s=75)

plt.xticks(rotation=90)

plt.grid()

plt.show()
data['Price'].max()
paid_filter = data["Price"]!=0

paid_apps = data[paid_filter]
paid_apps.plot(kind="scatter",x="Price",y="Rating",figsize=(8,8),color="red",s=70)

plt.show()
fig,ax = plt.subplots(figsize=(12,8))

plt.scatter(x=CategoryString[paid_filter],y=paid_apps["Price"],color="#EC407A",marker="^",s=75)

plt.xticks(rotation=90)

plt.grid()

plt.show()
fig,ax = plt.subplots(figsize=(8,7))

ax = sns.heatmap(data.corr(), annot=True,linewidths=.5,fmt='.1f')

plt.show()
#Loading Libraries

import plotly



import warnings

warnings.filterwarnings("ignore")



from plotly import tools

import plotly.figure_factory as ff

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

import plotly.figure_factory as ff

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import pandas as pd

import numpy as np
playstore = pd.read_csv('../input/googleplaystore.csv')

user_reviews = pd.read_csv("../input/googleplaystore_user_reviews.csv")

playstore.head(3)
playstore.isna().sum().sort_values().tail(5)
#lets remove them

playstore.dropna(inplace = True)
def KB_to_MB(d): ##converting KB to MB

    converted=[]

    for i in d:

        try:

            if i.endswith('k'):

                r=i.replace(i[len(i)-1],'')

                converted.append(float(r)/1000)

            elif i.endswith('M'):

                converted.append(float(i[:len(i)-1]))

            else:

                converted.append(float(i.replace(i,'nan')))

        except Exception as e:

            converted.append(float(i.replace(i,'nan')))

    return converted



def float_price(d): ##Removing the $ from the price

    converted=[]

    for i in d:

        try:

            if i.startswith('$'):

                converted.append(float(i.replace('$','')))



            else:

                converted.append(float(i))

        except Exception as e:

            converted.append(float(i.replace(i,'nan')))

    return converted



def rev_to_numeric(d):

    converted=[]

    for ele in d:

        try:

            converted.append(float(ele))

        except Exception as e:

            converted.append(float(ele.replace(ele,'nan')))

    return converted



def remove_plus(d):

    converted=[]

    for i in d:

        try:

            b=i.replace(',','')

            converted.append(float(b.replace('+','')))

        except Exception as e:

            converted.append(float(i.replace(i,'nan')))

    return converted
playstore['Reviews']=rev_to_numeric(playstore['Reviews'])

playstore['Price']=float_price(playstore['Price'])  

playstore['Size']=KB_to_MB(playstore['Size'])

playstore['Installs']=remove_plus(playstore['Installs'])
content = playstore["Content Rating"].value_counts()

app_type = playstore.Type.value_counts()
fig = {

    "data" : [

        {

            "labels" : app_type.index,"values" : app_type.values,

            "name" : "App Type","hoverinfo" : "label+value+name",

            "textinfo" : "percent",

            "domain" : {"x" : [0, .45]},"legendgroup" : "group2",

            "hole" : .4,"type" : "pie"

            

        },

        {

            "labels" : content.index,"values" : content.values,

            "name" : "Content Rating","hoverinfo" : "label+value+name",

            "textinfo" : "percent",

            "domain" : {"x" : [.55, 1]},"legendgroup" : "group",

            "hole" : .4,"type" : "pie"

        }

        ],

    

    "layout" : {"titlefont" : {"family" : "Ariel"},

                "legend" : {"x" : .36,

                            "y" : 0,

                            "orientation" : "h"},

                "annotations" : [{

                        "text" : "App Type","font" : {"size" : 15, "color" : "black"},

                        "showarrow" : False,"x" : .210,"y" : 0.5 },

                    

                        {"text" : "Content Rating",

                        "font" : {"size" : 15, "color" : "black"},

                        "showarrow" : False, "x" : .827, "y" : 0.5 }]

                }

    }



iplot(fig)
filtered_data = playstore[playstore["Content Rating"]=="Adults only 18+"]

filtered_data
category_count = playstore.Category.value_counts()

category_list = []



for i in category_count.index:

    name = i.lower().capitalize()

    name = name.replace("_", " ")

    category_list.append(name)
fig = {

    "data" : [

        {

            "x" : playstore.Category,

            "name" : "Category ",

            "marker" : {"color" : "rgba(15,15,15)"},

            "text" : category_list,

            "type" : "histogram"

        }

    ],

    "layout" : {"title" : "Category Count",

                "titlefont" : {"size" : 20},

                "xaxis" : {"tickangle" : 45},

                "yaxis" : {"title" : "Count"},

                "margin" : {"b" : 121}

    }

}



iplot(fig)
genre_count = playstore.Genres.value_counts()
fig = {

    "data" : [

        {

            "x" : playstore.Genres,

            "name" : "Genre",

            "marker" : {"color" : "rgba(15,250,120)"},

            "text" : genre_count.index,

            "type" : "histogram"

        }

    ],

    "layout" : {"title" : "Genre Count",

                "xaxis" : {"tickangle" : 45},

                "yaxis" : {"title" : "Genre Count"},

                "margin" : {"b" : 150,

                            "r" : 100}

    }

}



iplot(fig)
user_reviews.dropna(inplace=True)

sentiment_count = user_reviews.Sentiment.value_counts()
fig = {

    "data" : [

        {

            "labels" : sentiment_count.index,

            "values" : sentiment_count.values,

            "name" : "Sentiment",

            "hoverinfo" : "label+value+name",

            "textinfo" : "percent",

            "marker" : {"line" : {"width" : 1.1, "color" : "black"}},

            "hole" : .3,

            "type" : "pie"

            

        },

    ],

    

    "layout" : {

        "title" : "Sentiment Ratio of Apps",

        "titlefont" : {"size" : 20}

    }

}



iplot(fig)

category_count = playstore.Category.value_counts()

category_list = []



for i in category_count.index:

    name = i.lower().capitalize()

    name = name.replace("_", " ")

    category_list.append(name)



genre_count = playstore.Genres.value_counts()
fig = {

    "data" : [

        {

            "labels" : category_list[:20],

            "values" : category_count.values[:20],

            "name" : "Category",

            "hoverinfo" : "label+value+name",

            "textinfo" : "percent",

            "marker" : {"line" : {"width" : 1.1, "color" : "black"}},

            "hole" : .3,

            "type" : "pie"

            

        },

    ],

    

    "layout" : {

        "title" : "Category Ratio of Apps",

        "titlefont" : {"color" : "black",

                       "size" : 20}

    }

}



iplot(fig)





fig = {

    "data" : [

        {

            "labels" : genre_count.index[:20],

            "values" : genre_count.values[:20],

            "name" : "Genre",

            "hoverinfo" : "label+value+name",

            "textinfo" : "percent",

            "marker" : {"line" : {"width" : 1.1, "color" : "black"}},

            "hole" : .3,

            "type" : "pie"

            

        },

    ],

    

    "layout" : {

        "title" : "Genre Ratio of Apps",

        "titlefont" : {"color" : "black",

                       "size" : 20}

    }

}



iplot(fig)
free = playstore[playstore.Installs >= 10000]

free = free[free.Reviews >= 100]

new_index=free["Rating"].sort_values(ascending=False).index.values

bestrated=free.reindex(new_index)
bestrated.head()
top20_best_apps=bestrated.head(20)

data=[

    {

        "y": top20_best_apps.Size,

        "x": top20_best_apps.App,

        

        "mode":"markers",

        "marker":{"color": "rgba(15,15,15)","size" :50,'showscale': True},

        "text":top20_best_apps.Category  

    }]



iplot(data)
playstore.Category = playstore.Category[playstore.Category != "1.9"]

rating_of_category = playstore.groupby("Category")["Rating"].mean()



category_list = []

category_count = playstore.Category.value_counts()



for i in category_count.index:

    name = i.lower().capitalize()

    name = name.replace("_", " ")

    category_list.append(name)
fig = {

    "data" : [

        {

            "x" : category_list,"y" : rating_of_category,

            "name" : "rating","text" : category_list,

            "marker" : {"color" : "rgba(255,15,255,0.7)"},

            "type" : "scatter","mode" : "lines+markers"

        }

    ],

    

    "layout" : {

        "title" : "Rating Rate of Categories",

        "titlefont" : {"family" : "Arial", "size" : 17, "color" : "black"},

        "showlegend" : True,

        "xaxis" : {"ticklen" : 5,"zeroline" : True,

                   "autorange" : True,"showgrid" : False,

                   "zeroline" : False,"showline" : True,

                   "gridcolor" : "rgba(0,0,0,.2)","tickangle" : 45},

        

        "yaxis" : {"title" : "Ratings","titlefont" : {"color" : "black"},

                   "ticklen" : 5,"zeroline" : False,

                   "autorange" : True,"showgrid" : False,"showline" : True,

                   "gridcolor" : "rgba(0,0,0,.2)"},

        

        "margin" : {"b" : 111}

        

    }

}



iplot(fig)
playstore.Reviews = playstore.Reviews.replace("3.0M", "3000000")

playstore.Reviews = playstore.Reviews.astype(int)



review_of_category = playstore.groupby("Category")["Reviews"].mean()

category_count = playstore.Category.value_counts()



reviews_rate = []



for i in category_count.index:

    foo = (review_of_category[i]/category_count[i])

    reviews_rate.append(foo)
df = playstore.copy()

install = df.groupby("Category")["Installs"].mean().round()

install = install/800000





fig = {

    "data" : [

        {

            "x" : category_list,

            "y" : rating_of_category,

            "name" : "category",

            "text" : category_list,

            "marker" : {"color" : reviews_rate,

                        "size" : install,

                        "showscale" : True,

                        "colorscale" : "Blackbody"

},

            "mode" : "markers",

            "type" : "scatter"

        }

    ],

    "layout" : { "title" : "Rating of Categories With Installs",

        "titlefont" : {"family" : "Arial", "size" : 17, "color" : "black"},

        "showlegend" : True,

        "xaxis" : {"ticklen" : 5,"zeroline" : True,

                   "autorange" : True,"showgrid" : False,

                   "zeroline" : False,"showline" : True,

                   "gridcolor" : "rgba(0,0,0,.2)","tickangle" : 45},

        

        "yaxis" : {"title" : "Ratings","titlefont" : {"color" : "black"},

                   "ticklen" : 5,"zeroline" : False,

                   "autorange" : True,"showgrid" : False,"showline" : True,

                   "gridcolor" : "rgba(0,0,0,.2)"},

        

        "margin" : {"b" : 111}

    }

}



iplot(fig)
fig = {

    "data" : [

        {

            "x" : category_list,

            "y" : reviews_rate,

            "z" : rating_of_category,

            "marker" : {"size" : 10, "color" : "blue"},

            "mode" : "markers",

            "type" : "scatter3d"

        }

    ],

    "layout" : {

        "margin" : {"l" : 0, "r" : 0, "b" : 0, "t" : 0,},

    }

}



iplot(fig)
rating_of_genres = playstore.groupby("Genres")["Rating"].mean()



genres_list = []

genres_count = playstore.Genres.value_counts()



for i in genres_count.index:

    name = i.lower().capitalize()

    name = name.replace("_", " ")

    genres_list.append(name)
review_of_genres = playstore.groupby("Genres")["Reviews"].mean()

reviews_rate_genres = []



for i in genres_count.index:

    foo = (review_of_genres[i]/genres_count[i])

    reviews_rate_genres.append(foo)
fig = {

    "data" : [

        {

            "x" : genres_list,

            "y" : reviews_rate_genres,

            "name" : "review",

            "text" : category_list,

            "marker" : {"color" : "red"},

            "type" : "scatter",

            "mode" : "lines+markers"

        },

        {

            "x" : genres_list,

            "y" : rating_of_genres,

            "name" : "rating",

            "text" : category_list,

            "marker" : {"color" : "blue"},

            "type" : "scatter",

            "mode" : "lines+markers",

            "xaxis" : "x2",

            "yaxis" : "y2",

        }

    ],

    "layout" : {

        "xaxis2" : {"domain" : [0, 1],

                    "anchor" : "y2",

                    "showticklabels" : False},

        "yaxis2" : {"domain" : [.55, 1],

                    "anchor" : "x2"},

        "margin" : {"b" : 111},

        "xaxis" : {"tickangle" : 45,

                   "domain" : [0, 1]},

        "yaxis" : {"domain" :[0, .45]}

    }

}



iplot(fig)
paid_filter = playstore["Price"]!=0

paid_apps = playstore[paid_filter]
trace = go.Scatter(

    x = paid_apps.Price,

    y =  paid_apps.Rating,

    mode = 'markers',

    name = 'markers')



data = [trace]



iplot(data)
fig = {

    "data" : [

        {

            "y" : playstore.Rating,

            "x" : playstore.Category,

            "marker" : {"color" : "blue"},

            "type" : "box",

            "boxpoints" : "outliers",

            "boxmean" : True

        }

    ],

    "layout" : {

        "boxmode" : "group"

    }

}



iplot(fig)
fig = {

    "data" : [

        {

            "y" : playstore.Rating,

            "x" : playstore.Genres,

            "marker" : {"color" : "red"},

            "type" : "box",

            "boxpoints" : "outliers",

            "boxmean" : True

        }

    ],

    "layout" : {

        "boxmode" : "group"

    }

}



iplot(fig)