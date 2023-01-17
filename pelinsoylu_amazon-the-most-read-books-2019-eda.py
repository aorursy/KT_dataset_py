import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline  #to add plots to your Jupyter notebook

# plotly

# import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

# word cloud library

from wordcloud import WordCloud

# matplotlib

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read the data

amazon_best_sellers1= pd.read_csv("../input/amazon-the-most-read-books-of-the-2019-dataset/Amazon-bestsellers-2019.csv", decimal=",")
amazon_best_sellers1.head(2)
# information about the data types.

amazon_best_sellers1.info()
# dropping "Price_Old" and "By" columns which have only nan values.

amazon_best_sellers= amazon_best_sellers1.drop(columns=['Price_Old','By'])
# drop "$" from the "Price" feature.

amazon_best_sellers["Price"] = amazon_best_sellers["Price"].str.replace(r'\D', '').astype(float)

#clean the nan values

amazon_best_sellers['Rating'].value_counts(dropna=False)

amazon_best_sellers['Rating'].dropna(inplace=True)

amazon_best_sellers['Reviews'].value_counts(dropna=False)

amazon_best_sellers['Reviews'].dropna(inplace=True)

amazon_best_sellers["Price"].value_counts(dropna=False)

amazon_best_sellers['Price'].dropna(inplace=True)
#Control the features. If it returns nothing that means nan values are dropped.

assert amazon_best_sellers['Rating'].notnull().all()

assert amazon_best_sellers['Reviews'].notnull().all()

assert amazon_best_sellers['Price'].notnull().all()
#convert data types.

amazon_best_sellers['Rating'] = amazon_best_sellers['Rating'].astype(float)

amazon_best_sellers['Reviews'] = amazon_best_sellers['Reviews'].astype(float)
# give the value "1" for the bestseller books and "0" for the others.

amazon_best_sellers["Best_Seller"]= [ "1" if i=="Best Seller" else "0" for i in amazon_best_sellers.Badge ]

amazon_best_sellers["Best_Seller"].value_counts()
amazon_best_sellers["Name"].unique()
#categorized popular and unpopular books by their reviwes.If they have over 10.000 reviews, called them popular.

amazon_best_sellers["Popularity"]=["Popular" if i>=10.00 else "Unpopular" for i in amazon_best_sellers.Reviews]

amazon_best_sellers.Popularity.value_counts()
amazon_best_sellers.describe()
# the avarage ratings and values of best seller books.

amazon_best_sellers.groupby("Best_Seller")[["Rating","Reviews"]].mean() 

plt.figure(figsize = (10,7))

sns.boxplot(x="Best_Seller", y="Rating", hue="Popularity", data=amazon_best_sellers, palette="husl") 

plt.show()


sns.countplot(amazon_best_sellers["Best_Seller"])

plt.show()

g = sns.catplot(x="Popularity", hue="Popularity", col="Best_Seller",

                 data=amazon_best_sellers, kind="count",

                 height=4, aspect=.7);
amazon_best_sellers['Best_Seller'] = amazon_best_sellers['Best_Seller'].astype(float)

b1= amazon_best_sellers.Rating > 4.5

amazon_best_sellers.Best_Seller.astype(float)

b2= amazon_best_sellers.Best_Seller>0

df=amazon_best_sellers[b1 & b2]



labels = df.Rating.value_counts().index

colors = ['blue','red','yellow','green']

explode = [0,0,0,0]

sizes =df.Rating.value_counts().values

plt.figure(figsize = (5,5))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%') 

plt.title('Best Sellers Ratings',color = 'blue',fontsize = 15) 

plt.show()
# Another way of showing the popularity vs. unpopularity of 1(Best_Seller) and 0(Not).

plt.figure(figsize = (7,7))

sns.swarmplot(x="Best_Seller", y="Rating", hue="Popularity", data=amazon_best_sellers) 

plt.show()
# Correlations.

amazon_best_sellers.corr()
# Correlation Map.

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(amazon_best_sellers.corr(), annot=True, linewidths=0.6,linecolor="red", fmt= '.1f',ax=ax)

plt.show() 
# Books Ratings by 0 and 1.

# Importig Plotly Library.

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



df5 = amazon_best_sellers.Rating [amazon_best_sellers["Best_Seller"]>0]

df6= amazon_best_sellers.Rating [amazon_best_sellers["Best_Seller"]<1]

trace1 = go.Histogram(

    x=df5,

    opacity=0.75,

    name = "1",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))

trace2 = go.Histogram(

    x=df6,

    opacity=0.75,

    name = "0",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



data = [trace1, trace2]

layout = go.Layout(barmode='overlay',

                   title='',

                   xaxis=dict(title='Best_Seller'),

                   yaxis=dict( title=''),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
#best sellers rating,review and price comparison with 3D plot.

amazon_best_sellers1 = amazon_best_sellers.iloc[:200,:]

trace1 = go.Scatter3d(

    

    x=amazon_best_sellers1.Rating,

    y=amazon_best_sellers1.Reviews,

    z=amazon_best_sellers1.Price,

    mode='markers',

    marker=dict(

        size=10,

        color='rgb(255,0,0)',       

    )

)

data = [trace1]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    )   

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)
from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objects as go

# import figure factory

import plotly.figure_factory as ff



df = amazon_best_sellers.loc[:,["Rating","Reviews", "Price"]]

df["index"] = np.arange(1,len(df)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(df, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)