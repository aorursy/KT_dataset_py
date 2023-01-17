# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import plotly.plotly as py
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/googleplaystore.csv")
user_reviews = pd.read_csv("../input/googleplaystore_user_reviews.csv")
data.head()
data.info()
data.isna().sum()
data.dropna(inplace = True)
content = data["Content Rating"].value_counts()
app_type = data.Type.value_counts()


fig = {
    "data" : [
        {
            "labels" : app_type.index,
            "values" : app_type.values,
            "name" : "App Type",
            "hoverinfo" : "label+value+name",
            "textinfo" : "percent",
            "marker" : {"line" : {"width" : 1.1, "color" : "black"}},
            "domain" : {"x" : [0, .45]},
            "legendgroup" : "group2",
            "hole" : .4,
            "type" : "pie"
            
        },
        {
            "labels" : content.index,
            "values" : content.values,
            "name" : "Content Rating",
            "hoverinfo" : "label+value+name",
            "textinfo" : "percent",
            "marker" : {"line" : {"width" : 1.1, "color" : "black"}},
            "domain" : {"x" : [.55, 1]},
            "legendgroup" : "group",
            "hole" : .4,
            "type" : "pie"
            
        }
        
    ],
    
    "layout" : {"titlefont" : {"family" : "Ariel"},
                "legend" : {"x" : .36,
                            "y" : 0,
                            "orientation" : "h"},
                "annotations" : [
                    {
                        "text" : "App Type",
                        "font" : {"size" : 20, "color" : "black"},
                        "showarrow" : False,
                        "x" : .158,
                        "y" : 1.2
                    },
                    {
                        "text" : "Content Rating",
                        "font" : {"size" : 20, "color" : "black"},
                        "showarrow" : False,
                        "x" : .867,
                        "y" : 1.2
                    }
                ]
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
price_count = data.Price.value_counts()

fig = {
    "data" : [
        {
            "labels" : price_count.index[1:20],
            "values" : price_count.values[1:20],
            "name" : "Price",
            "hoverinfo" : "label+value+name",
            "textinfo" : "percent",
            "marker" : {"line" : {"width" : 1.1, "color" : "black"}},
            "hole" : .3,
            "type" : "pie"
            
        },
    ],
    
    "layout" : {
        "title" : "Price Ratio of Apps",
        "titlefont" : {"size" : 20}
    }
}

iplot(fig)
fig = {
    "data" : [
        {
            "x" : data.Price,
            "name" : "Price",
            "marker" : {"color" : "rgba(15,250,120)"},
            "type" : "histogram"
        }
    ],
    "layout" : {"title" : "Price Count",
                "titlefont" : {"size" : 20},
                "xaxis" : {"tickangle" : 45,
                           "type" : "category"}
    }
}

iplot(fig)
price_count = data.Price.value_counts()

color = []
for each in range(1, len(price_count.index)+1):
    color.append(each)

fig = {
    "data" : [
        {
            "x" : price_count.index[1:],
            "y" : price_count.values[1:],
            "name" : "Price",
            "marker" : {"color" : color,
                        "colorscale" : "Jet",
                        "cmax" : max(color),
                        "cmin" : 0},
            "type" : "bar"
        }
    ],
    
    "layout" : {"title" : "Count of Prices (Except Free)",
                "titlefont": {"color" : "black",
                              "size" : 20},
                "xaxis" : {"autorange" : True,
                           "type" : "category",
                           "tickangle" : 45}
    }
}
iplot(fig)
category_count = data.Category.value_counts()
category_list = []

for i in category_count.index:
    name = i.lower().capitalize()
    name = name.replace("_", " ")
    category_list.append(name)


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
            "x" : data.Category,
            "name" : "Category ",
            "marker" : {"color" : "rgba(15,250,120)"},
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
fig = {
    "data" : [
        {
            "x" : category_list,
            "y" : data.Category.value_counts().values,
            "name" : "Category",
            "marker" : {"color" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                        "cmax" : 30,
                        "cmin" : 0,
                        "colorscale" : "Jet",
                        "colorbar" : {"title" : "Colorbar"}},
            "type" : "bar",
        },
    ],
    "layout" : {"title" : "Count of Categories",
                "titlefont" : {"size" : 20},
                "xaxis" : {"tickangle" : 45},
                "margin" : {"b" : 100}
                
    }
}

iplot(fig)
genre_count = data.Genres.value_counts()

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
fig = {
    "data" : [
        {
            "x" : data.Genres,
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
genre_count = data.Genres.value_counts()
genre_color = []
for each in range(1, len(genre_count.index)+1):
    genre_color.append(each)

fig = {
    "data" : [
        {
            "x" : genre_count.index[:],
            "y" : genre_count.values[:],
            "name" : "Genre",
            "marker" : {"color" : genre_color,
                        "cmax" : max(genre_color),
                        "cmin" : 0,
                        "colorscale" : "Jet",
                        "colorbar" : {"title" : "Colorbar"}},
            "type" : "bar"
        }
    ],
    "layout" : {"title" : "Genre Count",
                "xaxis" : {"tickangle" : 45},
                "yaxis" : {"title" : "Count"},
                "margin" : {"b" : 174,
                            "r" : 110}
    }
}

iplot(fig)
data.Category = data.Category[data.Category != "1.9"]

rating_of_category = data.groupby("Category")["Rating"].mean()

category_list = []

for i in category_count.index:
    name = i.lower().capitalize()
    name = name.replace("_", " ")
    category_list.append(name)


fig = {
    "data" : [
        {
            "x" : category_list,
            "y" : rating_of_category,
            "name" : "rating",
            "text" : category_list,
            "marker" : {"color" : "rgba(255,15,255,0.7)"},
            "type" : "scatter",
            "mode" : "lines+markers"
        }
    ],
    
    "layout" : {
        "title" : "Rating Rate of Categories",
        "titlefont" : {"family" : "Arial", "size" : 17, "color" : "black"},
        "showlegend" : True,
        "xaxis" : {"ticklen" : 5,
                   "zeroline" : True,
                   "autorange" : True,
                   "showgrid" : False,
                   "zeroline" : False,
                   "showline" : True,
                   "gridcolor" : "rgba(0,0,0,.2)",
                   "tickangle" : 45
        },
        "yaxis" : {"title" : "Ratings",
                   "titlefont" : {"color" : "black"},
                   "ticklen" : 5,
                   "zeroline" : False,
                   "autorange" : True,
                   "showgrid" : False,
                   "showline" : True,
                   "gridcolor" : "rgba(0,0,0,.2)"},
        "margin" : {"b" : 111}
        
    }
}


iplot(fig)
data.Reviews = data.Reviews.replace("3.0M", "3000000")
data.Reviews = data.Reviews.astype(int)

review_of_category = data.groupby("Category")["Reviews"].mean()
category_count = data.Category.value_counts()

reviews_rate = []

for i in category_count.index:
    foo = (review_of_category[i]/category_count[i])
    reviews_rate.append(foo)


fig = {
    "data" : [
        {
            "x" : category_list,
            "y" : reviews_rate,
            "name" : "review",
            "text" : category_list,
            "marker" : {"color" : "rgba(255,15,255,0.7)"},
            "type" : "scatter",
            "mode" : "lines+markers"
        }
    ],
    
    "layout" : {
        "title" : "Reviews Rate of Categories",
        "titlefont" : {"family" : "Arial", "size" : 17, "color" : "black"},
        "showlegend" : True,
        "xaxis" : {"ticklen" : 5,
                   "zeroline" : True,
                   "autorange" : True,
                   "showgrid" : True,
                   "zeroline" : False,
                   "showline" : True,
                   "gridcolor" : "rgba(0,0,0,.2)",
                   "tickangle" : 45
        },
        "yaxis" : {"title" : "Reviews",
                   "titlefont" : {"color" : "black"},
                   "ticklen" : 5,
                   "zeroline" : False,
                   "autorange" : True,
                   "showgrid" : True,
                   "showline" : True,
                   "gridcolor" : "rgba(0,0,0,.2)"},
        "margin" : {"b" : 111}
    }
}


iplot(fig)
df = data.copy()
df.Installs = [each.replace("+", "") for each in df.Installs]
df.Installs = [float(each.replace(",", "")) for each in df.Installs[df.Installs != "Free"]]
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
    "layout" : {"xaxis" : {"tickangle" : 45},
                "margin" : {"b" : 111}
    }
}

iplot(fig)
plt.subplots(figsize=[8,8])

wordcloud = WordCloud(
    background_color = "white",
).generate(" ".join(data.Genres))
plt.axis("off")
plt.imshow(wordcloud, interpolation="bilinear")
plt.title("Genre WordCloud", size = 20)
plt.show()
plt.subplots(figsize=[8,8])

wordcloud = WordCloud(
    background_color = "white",
).generate(" ".join(category_list))
plt.axis("off")
plt.imshow(wordcloud, interpolation="bilinear")
plt.title("Category WordCloud", size = 20)
plt.show()
box1 = data[(data.Category == "ART_AND_DESIGN")]
box2 = data[(data.Category == "AUTO_AND_VEHICLES")]
box3 = data[(data.Category == "BEAUTY")]

fig = {
    "data" : [
        {
            "y" : box1.Rating,
            "x" : box1.Category,
            "marker" : {"color" : "red"},
            "type" : "box",
            "boxpoints" : "outliers",
            "boxmean" : True
        },
        {
            "y" : box2.Rating,
            "x" : box2.Category,
            "marker" : {"color" : "blue"},
            "type" : "box",
            "boxpoints" : "outliers"
        },
        {
            "y" : box3.Rating,
            "x" : box3.Category,
            "marker" : {"color" : "green"},
            "type" : "box",
            "boxpoints" : "outliers"
        }
        
    ],
    "layout" : {
        "boxmode" : "group"
    }
}

iplot(fig)
df5 = data.loc[:,["Rating", "Reviews"]]
df5["index"] = np.arange(1,len(df5)+1)

fig = ff.create_scatterplotmatrix(df5[:2000], diag="box", index="index",
                                  colormap="Portland", colormap_type="cat",
                                  height=500, width=600)

iplot(fig)
fig = {
    "data" : [
        {
            "x" : category_list,
            "y" : reviews_rate,
            "name" : "review",
            "text" : category_list,
            "marker" : {"color" : "red"},
            "type" : "scatter",
            "mode" : "lines+markers"
        },
        {
            "x" : category_list,
            "y" : rating_of_category,
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
        "xaxis2" : {"domain" : [.47, .95],
                    "anchor" : "y2",
                    "showticklabels" : False},
        "yaxis2" : {"domain" : [.6, .95],
                    "anchor" : "x2"},
        "margin" : {"b" : 111},
        "xaxis" : {"tickangle" : 45}
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
fig = {
    "data" : [
        {
            "x" : category_list,
            "y" : reviews_rate,
            "name" : "review",
            "text" : category_list,
            "marker" : {"color" : "red"},
            "type" : "scatter",
            "mode" : "lines+markers"
        },
        {
            "x" : category_list,
            "y" : rating_of_category,
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
