# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

from collections import Counter

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

   

from subprocess import check_output

print(check_output(["ls","../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')

data.info()
data.corr()
f,ax = plt.subplots(figsize=(16, 16))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.NA_Sales.plot(kind = 'line', color = 'g',label = 'NA_Sales',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.EU_Sales.plot(color = 'r',label = 'EU_Sales',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

data.plot(kind='scatter', x='NA_Sales', y='EU_Sales',alpha = 0.5,color = 'red')

plt.xlabel('NA_Sales')              # label = name of label

plt.ylabel('EU_Sales')

plt.title('NA_Sales EU_Sales Scatter Plot')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

data.NA_Sales.plot(kind = 'hist',bins = 70,figsize = (12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

data.NA_Sales.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
data = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')

series = data['NA_Sales']        # data['NA_Sales'] = series

print(type(series))

data_frame = data[['NA_Sales']]  # data[['NA_Sales']] = data frame

print(type(data_frame))
# 1 - Filtering Pandas data frame

x = data['NA_Sales']>200     # There are only 3 pokemons who have higher defense value than 200

print(x)

data[x]
 #2 - Filtering pandas with logical_and

data[np.logical_and(data['NA_Sales']>10, data['EU_Sales']>11 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data['NA_Sales']>2) & (data['EU_Sales']>1)]
threshold = sum(data.EU_Sales)/len(data.EU_Sales)

data["new_EU"] = ["high" if i > threshold else "middle" if  i>45 else "low" for i in data.EU_Sales]

data.loc[:10,["new_EU","EU_Sales"]] # we will learn loc more detailed later

data
data.info()
data.describe()
print(data['NA_Sales'].value_counts(dropna =False))
data['EU_Sales'].value_counts(dropna =False)
data['EU_Sales'].describe()
data.dropna(inplace = True)  

data.describe()
data.boxplot(column='NA_Sales',by = 'EU_Sales')
data_new = data.head()    # I only take 5 rows into new data

data_new
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['NA_Sales','EU_Sales'])

melted
melted.pivot(index = 'Name', columns = 'variable',values='value')
data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
data3 = data['Name'].head()

data1 = data['NA_Sales'].head()

data2= data['EU_Sales'].head()

conc_data_col = pd.concat([data3,data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
data.dtypes
data['Name'] = data['Name'].astype('category')

data['NA_Sales'] = data['NA_Sales'].astype('category')

data.dtypes
# data.info()

data
data["NA_Sales"].value_counts(dropna=False)
data1=data

data1.dropna(inplace=True)

data1.info()
data = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')

data.head()
year_total = data.groupby(['Year']).sum()

sum_year_total = year_total['Global_Sales']

#veri setindeki float olan yillarin int donusumu yapiliyor

yil = year_total.index.astype(int)



plt.figure(figsize=(12,8))

ax = sns.barplot(y = sum_year_total, x = yil)

ax.set_xlabel(xlabel='yillar', fontsize=16)

ax.set_xticklabels(labels = yil, fontsize=12, rotation=50)

ax.set_ylabel(ylabel='toplam satislar', fontsize=16)

ax.set_title(label='YILLIK BAZDA SATISLAR', fontsize=20)

plt.show();
year_total = data.groupby(['Year']).count()

x = year_total['Global_Sales']

y = x.index.astype(int)

plt.figure(figsize=(12,8))

colors = sns.color_palette("muted")

ax = sns.barplot(y = y, x = x, orient='h', palette=colors)

ax.set_xlabel(xlabel='Sayilar', fontsize=16)

ax.set_ylabel(ylabel='Yil', fontsize=16)

ax.set_title(label='YILLIK BAZDA SATISLAR', fontsize=20)

plt.show();
table = data.pivot_table('Global_Sales', index='Genre', columns='Year', aggfunc='sum')

table

genres = table.idxmax()

sales = table.max()

years = table.columns.astype(int)

data = pd.concat([genres, sales], axis=1)

data.columns = ['Genre', 'Global Sales']



plt.figure(figsize=(12,8))

ax = sns.pointplot(y = 'Global Sales', x = years, hue='Genre', data=data, size=15, palette='Dark2')

ax.set_xlabel(xlabel='Yillar', fontsize=16)

ax.set_ylabel(ylabel='Dunyadaki Yillik Satislar', fontsize=16)

ax.set_title(label='Yillik Bazda Oyun Turlerinden en Yuksek Satislar Grafigi', fontsize=20)

ax.set_xticklabels(labels = years, fontsize=12, rotation=50)

plt.show();
q = table.quantile(0.90)

data = table[table < q]

years = table.columns.astype(int)



plt.figure(figsize=(12,8))

ax = sns.boxplot(data=data)

ax.set_xticklabels(labels=years, fontsize=12, rotation=50)

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_ylabel(ylabel='Revenue Per Game in $ Millions', fontsize=16)

ax.set_title(label='Distribution of Revenue Per Game by Year in $ Millions', fontsize=20)

plt.show()

plt.show()

data = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')

data = data.dropna()



data.Year=data.Year.astype(int)

#World rank of the top 100 video games, north america sales and europe sales.

df=data.loc[:99,:] # data.iloc[:100,:] -- data.head(100)



import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot



trace1=go.Scatter(

                x=df.Rank,

                y=df.NA_Sales,

                mode="lines+markers",

                name="North America Sales",

                marker= dict(color = 'rgba(158, 90, 10, 0.7)'),

                text=df.Name)

trace2=go.Scatter(

                x=df.Rank,

                y=df.EU_Sales,

                mode="lines",

                name="Europe Sales",

                marker=dict(color = 'rgba(56, 140, 200, 0.7)'),

                text=df.Name)



edit_df=[trace1,trace2]

layout=dict(title="World rank of the top 100 video games, north america and europe sales .",

            xaxis=dict(title="World Rank",tickwidth=5,ticklen=8,zeroline=False))

fig=dict(data=edit_df,layout=layout)

iplot(fig)
#Games according to Genre

labels=data.Genre.value_counts().index

explode = [0,0,0,0,0,0,0,0,0,0,0,0]

sizes = data.Genre.value_counts().values

# visual

plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=sns.color_palette('Set2'), autopct='%1.1f%%')

plt.title('Games According to Genre',fontsize = 17,color = 'green')
# Count of Platforms

plt.subplots(figsize = (15,8))

sns.countplot(data.Platform)

plt.title("Platform",color = 'blue',fontsize=20)
#The genres according to the average sales

# data preparation

dataGenre_list=list(data.Genre.unique())

na_sales=[]

eu_sales=[]

jp_sales=[]

other_sales=[]

global_sales=[]

for i in dataGenre_list:

    val=data[data.Genre==i]

    na_sales.append(val.NA_Sales.mean())

    eu_sales.append(val.EU_Sales.mean())

    jp_sales.append(val.JP_Sales.mean())

    other_sales.append(val.Other_Sales.mean())

    global_sales.append(val.Global_Sales.mean())

    

df1=pd.DataFrame({"Genre":dataGenre_list,"NA_Sales":na_sales,"EU_Sales":eu_sales,"JP_Sales":jp_sales,"Other_Sales":other_sales,"Global_Sales":global_sales})



# data visualization

trace1=go.Scatter(

                x=df1.Genre,

                y=df1.NA_Sales,

                mode="lines+markers",

                name="North America Sales",

                marker= dict(color = 'rgba(25, 25, 255, 0.7)'))

trace2=go.Scatter(

                x=df1.Genre,

                y=df1.EU_Sales,

                mode="lines",

                name="Europe Sales",

                marker=dict(color = 'rgba(25, 255, 25, 0.7)'))

trace3=go.Scatter(

                x=df1.Genre,

                y=df1.JP_Sales,

                mode="lines+markers",

                name="Japan Sales",

                marker=dict(color = 'rgba(255, 25, 25, 0.7)'))

trace4=go.Scatter(

                x=df1.Genre,

                y=df1.Other_Sales,

                mode="lines",

                name="Other Sales",

                marker=dict(color = 'rgba(25, 25, 25, 0.7)'))

trace5=go.Scatter(

                x=df1.Genre,

                y=df1.Global_Sales,

                mode="lines+markers",

                name="Global Sales",

                marker=dict(color = 'rgba(250, 150, 0, 0.7)'))

edit_df=[trace1,trace2,trace3,trace4,trace5]

layout=dict(title="The genres according to the average sales",

            xaxis=dict(title="  Genre of Video Games",tickwidth=5,ticklen=8,zeroline=False))

fig=dict(data=edit_df,layout=layout)

plt.savefig('graph.png')

iplot(fig)
# Sales of the top 4 Video Games in 2016

df2016=data[data.Year==2016].iloc[:4,:]



import plotly.graph_objs as go



trace1=go.Bar(

                x=df2016.Name,

                y=df2016.NA_Sales,

                name="NA_Sales",

                marker= dict(color = 'rgba(158, 90, 10, 0.7)',

                            line=dict(color='rgb(0,0,0)',width=1.9)),

               text=df2016.Publisher)

trace2=go.Bar(

                x=df2016.Name,

                y=df2016.EU_Sales,

                name="EU_Sales",

                marker=dict(color = 'rgba(56, 140, 200, 0.7)',

                           line=dict(color='rgb(0,0,0)',width=1.9)),

                text=df2016.Publisher)

trace3=go.Bar(

                x=df2016.Name,

                y=df2016.JP_Sales,

                name="JP_Sales",

                marker=dict(color = 'rgba(156, 30, 130, 0.7)',

                           line=dict(color='rgb(0,0,0)',width=1.9)),

                text=df2016.Publisher)

trace4=go.Bar(

                x=df2016.Name,

                y=df2016.Other_Sales,

                name="Other_Sales",

                marker=dict(color = 'rgba(100, 240,50 , 0.7)',

                           line=dict(color='rgb(0,0,0)',width=1.9)),

                text=df2016.Publisher)



edit_df=[trace1,trace2,trace3,trace4]

layout=go.Layout(barmode="group",title='Sales of the top 4 Video Games in 2016')

fig=dict(data=edit_df,layout=layout)

#fig = go.Figure(data = edit_df, layout = layout)

plt.savefig('graph.png')

iplot(fig)
#Rates of Video Games Global Sales in 2016

df2016=data[data.Year==2016].iloc[:10,:]

pie_list=list(df2016.Global_Sales)

labels=df2016.Publisher

fig={

    "data":[

        {

            "values":pie_list,

            "labels":labels,

            "domain": {"x": [.2, 1]},

            "name": "Rates of Video Games ",

            "hoverinfo":"label+percent+name",

            "hole": .4,

            "type": "pie"

        },],

    "layout":{

        "title":"Rates of Video Games Global Sales in 2016",

        "annotations":[

            {

                "font":{"size":17},

                "showarrow": False,

                "text": "Video Games",

                "x": 0.75,

                "y": 0.5

            },

        ]

    }  

}

iplot(fig)
# Which video games is mentioned most at Genre of Action..

# data prepararion

from wordcloud import WordCloud  # word cloud library

w_data = data[data.Genre == 'Action'].iloc[:50,:]

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                        background_color='#f2f2f2',

                        width=532,

                        height=374

                     ).generate(" ".join(w_data.Name))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')

plt.show()
# World rank of the top 100 video games, japan sales and europe sales.

dframe=data.iloc[:100,:]

trace0=go.Scatter(

    x=dframe.Rank,

    y=dframe.JP_Sales,

    name = "Japan Sales",

    marker = dict(color = 'rgba(56, 182, 32, 0.8)'),

)

trace1=go.Scatter(

    x=dframe.Rank,

    y=dframe.EU_Sales,

    xaxis='x2',

    yaxis='y2',

    name = "Europe Sales",

     marker = dict(color = 'rgba(196, 20, 92, 0.8)'),

)

edit_data=[trace0,trace1]

layout = go.Layout(

    xaxis2=dict(

        domain=[0.5, 0.95],

        anchor='y2',        

    ),

    yaxis2=dict(

        domain=[0.5, 0.95],

        anchor='x2',

    ),

    title = ' World rank of the top 100 video games, japan sales and europe sales.'

)

fig = go.Figure(data=edit_data, layout=layout)

iplot(fig)
# World rank of the top 100 video games, japan sales and europe sales.

trace1=go.Scatter3d(

    x=dframe.Rank,

    y=dframe.EU_Sales,

    z=dframe.JP_Sales,

    mode='markers',

    marker=dict(

        size=12,

        #color='rgb(135,25,25)',

        color=dframe.Rank,                # set color to an array/list of desired values

        colorscale='Viridis',             # choose a colorscale

        opacity=0.9

    )

)

edit_data=[trace1]

layout=go.Layout(

     margin=dict(

        l=1,

        r=1,

        b=1,

        t=1  

    )

)

fig=go.Figure(data=edit_data,layout=layout)

iplot(fig)