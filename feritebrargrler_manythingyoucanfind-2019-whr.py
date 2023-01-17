# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid") # --> helps to visulize tools with grids. You can use another ones with looking plt.style.available

import seaborn as sns

from collections import Counter

import warnings

warnings.filterwarnings("ignore") # dont show warnings based on python

import plotly.graph_objs as go

# plotly

#import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/world-happiness/2019.csv") # reading csv data which we want to analyze
data.columns # columns in data 
data.head() # first 5 information about data (default is 5. If you want to see more you can write specific number you want yo see eg: data.head(10))--> reads 10
data.tail() # last 5 information about data.
data.info() # 
data.describe() # this will be used for mathematical calculations.
data.corr() # corrolations between columns. 
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data1 = data.rename(columns={'GDP per capita':'GDPPC'}) # Changed for making readable. Unless the system will give us SyntaxError. Becouse column's name is discreate.
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data1.GDPPC.plot(kind = 'line', color = 'g',label = 'GDPPC',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data1.Score.plot(color = 'r',label = 'Score',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()

# Scatter Plot 

# x = GDPPC, y = Score

data1.plot(kind='scatter', x='GDPPC', y='Score',alpha = 0.5,color = 'red')

plt.xlabel('GDPPC')              # label = name of label

plt.ylabel('Score')

plt.title('GDPPC Score Scatter Plot')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

data1.GDPPC.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
data.info()
def bar_plot(variable):

    """

        input: variable ex: "Country or region"

        output: bar plot & value count

    """

    # get feature

    var = data2[variable]

    # count number of categorical variable(value/sample)

    varValue = var.value_counts()

    

    # visualize

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{} : \n{} : ".format(variable,varValue))
data2 = data.rename(columns={'Country or region':'CountryOrRegion'})
category1 = ["CountryOrRegion"] # In this data it didnt work properly, but for learning I putted this there.

for c in category1:

    bar_plot(c)
category2 = ["CountryOrRegion"] # Secondly

for c in category2 : 

    print("{} \n".format(data2[c].value_counts()))
data3 = data.rename(columns={'Country or region':'CountryOrRegion', 'Overall rank':'OverallRank', 'Perceptions of corruption':'PerceptionsOfCorruption', 'GDP per capita':'GDPPC', 'Social support':'SocialSupport', 'Healthy life expectancy':'HealthyLifeExpectancy', 'Freedom to make life choices':'FreedomToMakeLifeChoices'})
def plot_hist(variable):

    plt.figure(figsize = (9,3))

    plt.hist(data3[variable], bins = 50) # frequency of bars, default = 10

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} disturbiton with hist".format(variable))

    plt.show()
numericVar = ["OverallRank", "PerceptionsOfCorruption", "Generosity", "FreedomToMakeLifeChoices", "HealthyLifeExpectancy", "SocialSupport", "GDPPC", "Score"]

for n in numericVar:

    plot_hist(n)

    plt.show()
data3[["CountryOrRegion","Generosity"]].groupby(["CountryOrRegion"], as_index = False).mean().sort_values(by ="Generosity", ascending = False)
data3[["CountryOrRegion","HealthyLifeExpectancy"]].groupby(["CountryOrRegion"], as_index = False).mean().sort_values(by ="HealthyLifeExpectancy", ascending = False)
data3[["CountryOrRegion","GDPPC","Score"]].groupby(["CountryOrRegion","Score"], as_index = False).mean().sort_values(by ="GDPPC", ascending = False)
data3.describe()
def detect_outliers(data,features):

    outlier_indices = []

    

    for c in features:

        # 1st quartile

        Q1 = np.percentile(data3[c],25)

        # 3rd quartile

        Q3 = np.percentile(data3[c],75)

        # IQR

        IQR = Q3 - Q1

        # Outlier step

        outlier_step = IQR * 1.5

        # detect outlier and their indeces

        outlier_list_col = data3[(data3[c] < Q1 - outlier_step) | (data3[c] > Q3 + outlier_step)].index

        # store indeces

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 1)

    

    return multiple_outliers
data3.loc[detect_outliers(data3, ["OverallRank","Score","GDPPC","SocialSupport", "HealthyLifeExpectancy", "FreedomToMakeLifeChoices", "Generosity", "PerceptionsOfCorruption" ])]
# drop outliers

data3 = data3.drop(detect_outliers(data3,["OverallRank","Score","GDPPC","SocialSupport", "HealthyLifeExpectancy", "FreedomToMakeLifeChoices", "Generosity", "PerceptionsOfCorruption"]),axis = 0).reset_index(drop = True)

# if we had outlier, we could drop outliers with this code.
data_len = len(data)

data = pd.concat([data], axis = 0).reset_index(drop = True)
data.head()
data.columns[data.isnull().any()] # in which columns there are missing values? 
data.isnull().sum() # how many ? 
"""

Lets say that Score and Social support have 2 missing values each.

data[data["Score"].isnull()] --> NaN objects of Score

****

data.boxplot(column ="Scoure", by ="Scial support")

plt.show()--> graphic of outliers and values.

***

train_df["Score"] = train_df["Score"].fillna("5.40") # filling missing values with 5.40 (mean value) or we can find index of missing value and checking upper score and lower score taking avarage of these scores and write it instead of 5.40

train_df[train_df["Score"].isnull()] # checking --> Will be no missing values

***



"""
data.describe()
data3.head() # I used data3 becouse of make data readable. I used it before.
new_index = (data3['GDPPC'].sort_values(ascending=False)).index.values # values were sorted according to descending order.

sorted_data = data.reindex(new_index)

sorted_data
#Visiualization

plt.figure(figsize=(45,20))

sns.barplot(x = sorted_data['Country or region'], y = sorted_data['GDP per capita'])

plt.xticks(rotation = 90) # rotation of countries's names

plt.xlabel('Country or region')

plt.ylabel('GDP per capita')

plt.title('GDP per capita Rates Given Countries')

plt.show()
#Happy Countries Respectively

plt.figure(figsize=(45,20))

ax= sns.barplot(x=data3['CountryOrRegion'], y=data3['Score'],palette = sns.cubehelix_palette())

plt.xticks(rotation = 90)

plt.xlabel('Country Or Region')

plt.ylabel('Happiness Frequency')

plt.title('Happy Countries Respectively')

plt.show()
data3.corr()
f,ax1 = plt.subplots(figsize =(40,20))

sns.pointplot(x=data3['CountryOrRegion'],y=data3['GDPPC'],data=data3,color='lime',alpha=0.8)

plt.xticks(rotation = 90)

sns.pointplot(x=data3['CountryOrRegion'],y=data3['HealthyLifeExpectancy'],data=data3,color='red',alpha=0.8)

plt.text(40,0.6,'Healthy Life Expectancy',color='red',fontsize = 25,style = 'italic')

plt.text(40,0.55,'GDPPC',color='lime',fontsize = 25,style = 'italic')

plt.xlabel('Countries',fontsize = 25,color='blue')

plt.ylabel('Values',fontsize = 25,color='blue')

plt.title('GDPPC and Healthy Life Expectancy',fontsize = 25,color='blue')

plt.grid()


# joint kernel density

# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.

# If it is zero, there is no correlation between variables

# Show the joint distribution using kernel density estimation 

g = sns.jointplot(data3['HealthyLifeExpectancy'], data3['GDPPC'], kind="kde", size=7)

plt.savefig('graph.png')

plt.show()

# you can change parameters of joint plot

# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }

# Different usage of parameters but same plot with previous one

g = sns.jointplot(data3["HealthyLifeExpectancy"], data3["GDPPC"], data=data,size=5, ratio=3, color="r")
# Visualization of Score rate vs GDPPC of each country with different style of seaborn code

# lmplot 

# Show the results of a linear regression within each dataset

sns.lmplot(x='GDPPC', y='Score', data=data3)

plt.show()
# Visualization of GDPPC vs Score of each Country with different style of seaborn code

# cubehelix plot

sns.kdeplot(data3.GDPPC, data3.Score, shade=True, cut=3) # shade maviliğin içinin dolu olmasu cut büyüklüğü

plt.show()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(data.corr(), annot=True, linewidths=0.5, linecolor="red", fmt=".1f", ax=ax)

plt.show()
data3.head()
data3.CountryOrRegion
sns.pairplot(data3)

plt.show()
data3 = data.rename(columns={'Country or region':'CountryOrRegion', 'Overall rank':'OverallRank', 'Perceptions of corruption':'PerceptionsOfCorruption', 'GDP per capita':'GDPPC', 'Social support':'SocialSupport', 'Healthy life expectancy':'HealthyLifeExpectancy', 'Freedom to make life choices':'FreedomToMakeLifeChoices'})
# Line Charts Example: GDPPC and HealthyLifeExpectancy vs OverallRank of Top 100 Countries, I choose these 2 bec. there are good corrolation between them.



# prepare data frame

df = data3.iloc[:100,:]



# import graph objects as "go"

import plotly.graph_objs as go



# Creating trace1

trace1 = go.Scatter(

                    x = df.OverallRank,

                    y = df.GDPPC,

                    mode = "lines",

                    name = "GDPPC",

                    marker = dict(color = 'rgba(1, 1, 250, 0.8)'),

                    text= df.CountryOrRegion)

# Creating trace2

trace2 = go.Scatter(

                    x = df.OverallRank,

                    y = df.HealthyLifeExpectancy,

                    mode = "lines+markers",

                    name = "HealthyLifeExpectancy",

                    marker = dict(color = 'rgba(40, 200, 5, 0.8)'),

                    text= df.CountryOrRegion)

data3 = [trace1, trace2]

layout = dict(title = 'GDPPC and HealhyLifeExpectancy vs World Rank of Top 100 Countries',

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)

             )

fig = dict(data = data3, layout = layout)

iplot(fig)



data3 = data.rename(columns={'Country or region':'CountryOrRegion', 'Overall rank':'OverallRank', 'Perceptions of corruption':'PerceptionsOfCorruption', 'GDP per capita':'GDPPC', 'Social support':'SocialSupport', 'Healthy life expectancy':'HealthyLifeExpectancy', 'Freedom to make life choices':'FreedomToMakeLifeChoices'})
# First Bar Charts Example: GDPPC and SocialSupport of top 5 cities in 2019 (style1)



df = data3.iloc[:5,:]



# import graph objects as "go"

import plotly.graph_objs as go

# create trace1 

trace1 = go.Bar(

                x = df.CountryOrRegion,

                y = df.GDPPC,

                name = "GDPPC",

                marker = dict(color = 'rgba(255, 174, 255, 1)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df.CountryOrRegion)

# create trace2 

trace2 = go.Bar(

                x = df.CountryOrRegion,

                y = df.SocialSupport,

                name = "Social Support",

                marker = dict(color = 'rgba(255, 255, 128, 1)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df.CountryOrRegion)

data3 = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data3, layout = layout)

iplot(fig)



data3 = data.rename(columns={'Country or region':'CountryOrRegion', 'Overall rank':'OverallRank', 'Perceptions of corruption':'PerceptionsOfCorruption', 'GDP per capita':'GDPPC', 'Social support':'SocialSupport', 'Healthy life expectancy':'HealthyLifeExpectancy', 'Freedom to make life choices':'FreedomToMakeLifeChoices'})
# Generosity Rates of first 7 countries



# data preparation

df = data3.iloc[:7,:]

pie = df.Generosity



#Plotly calculates the rates automaticaly.



#figure

fig = {

    "data" : [

        {

            "values" : pie,

            "labels" : df.CountryOrRegion,

            "domain": {"x": [0.1, .6]},

            "name" : "Generosity Rates",

            "hoverinfo" : "label+percent+name", # cursor shows the rates and name of country

            "hole" : 0.2, # Magnitude of white hole

            "type" : "pie"

        }

    ],

    "layout" : {

        "title" : "Generosity Rates Of First 5 Countries",

        "annotations" : [

            {"font" : { "size": 10}, #Magnitude of Generosity name at the top of pie chart

            "showarrow" : False,

             "text": "Generosity",

            "x": 0.20,

            "y": 1

            }

        ]

    } 

}

iplot(fig)
data3 = data.rename(columns={'Country or region':'CountryOrRegion', 'Overall rank':'OverallRank', 'Perceptions of corruption':'PerceptionsOfCorruption', 'GDP per capita':'GDPPC', 'Social support':'SocialSupport', 'Healthy life expectancy':'HealthyLifeExpectancy', 'Freedom to make life choices':'FreedomToMakeLifeChoices'})
df = data3.iloc[:20,:]

size  = df.Score

color = [each for each in df.FreedomToMakeLifeChoices]

data = [

    {

        'y' : df.PerceptionsOfCorruption,

        'x' : df.OverallRank,

        'mode' : 'markers',

        'marker':{

            'color': color,

            'size': size,

            'showscale': True # Determines whether or not a colorbar is displayed for this trace. Default

        },

        'text' : df.CountryOrRegion

    }

]

iplot(data)
# prepare data

G = data3.GDPPC

H = data3.HealthyLifeExpectancy



trace1 = go.Histogram(

    x=G,

    opacity=0.75,

    name = "GDPPC",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))

trace2 = go.Histogram(

    x=H,

    opacity=0.75,

    name = "HealthyLifeExpectancy",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



data = [trace1, trace2]

layout = go.Layout(barmode='overlay', # another mode is stack you can also use it.

                   title=' GDPPC and HealthyLifeExpectancy ratio',

                   xaxis=dict(title='X'),

                   yaxis=dict( title='Y'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
# data prepararion

happy = data3.CountryOrRegion

plt.subplots(figsize=(7,7))

wordcloud = WordCloud(

                          background_color='purple',

                          width=512,

                          height=384

                         ).generate(" ".join(happy))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
"""

Median (50th percentile) = middle value of the data set. Sort and take the data in the middle. It is also called 50% percentile that is 50% of data are less that median(50th quartile)(quartile)



25th percentile = quartile 1 (Q1) that is lower quartile

75th percentile = quartile 3 (Q3) that is higher quartile

height of box = IQR = interquartile range = Q3-Q1

Whiskers = 1.5 * IQR from the Q1 and Q3

Outliers = being more than 1.5*IQR away from median commonly.

"""

# data preparation





trace1= go.Box(

    y=data3.PerceptionsOfCorruption,

    name = 'PerceptionsOfCorruption',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

trace2 = go.Box(

    y=data3.Generosity,

    name = 'Generosity',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

data = [trace1, trace2]

iplot(data)
# import figure factory

import plotly.figure_factory as ff

# prepare data

dataframe = data3

df = dataframe.loc[:,["GDPPC","SocialSupport", "HealthyLifeExpectancy"]]

df["index"] = np.arange(1,len(df)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(df, diag='box', index='index',colormap='Picnic',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)
# first line plot

trace1 = go.Scatter(

    x=data3.OverallRank,

    y=data3.HealthyLifeExpectancy,

    name = "HealthyLifeExpectancy",

    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

)

# second line plot

trace2 = go.Scatter(

    x=data3.OverallRank,

    y=data3.Score,

    xaxis='x2',

    yaxis='y2',

    name = "Score",

    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),

)

data = [trace1, trace2]

layout = go.Layout(

    xaxis2=dict(

        domain=[0.6, 0.95],

        anchor='y2',        

    ),

    yaxis2=dict(

        domain=[0.6, 0.95],

        anchor='x2',

    ),

    title = 'Score and HealthyLifeExpectancy vs World Rank of Countries'



)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
# create trace 1 that is 3d scatter

trace1 = go.Scatter3d(

    x=data3.GDPPC,

    y=data3.SocialSupport,

    z=data3.HealthyLifeExpectancy,

    mode='markers',

    marker=dict(

        size=10,

        color=[each for each in data3.Score],   # set color to an array/list of desired values      

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
trace1 = [go.Choropleth(

               colorscale = 'Balance',

               locationmode = 'country names',

               locations = data3['CountryOrRegion'],

               text = data3['CountryOrRegion'], 

               z = data3['GDPPC'],

               )]



layout = dict(title = 'GDPPC',

                  geo = dict(

                      showframe = True,

                      showocean = True,

                      showlakes = True,

                      showcoastlines = True,

                      projection = dict(

                          type = 'hammer'

        )))





projections = [ "equirectangular", "mercator", "orthographic", "natural earth","kavrayskiy7", 

               "miller", "robinson", "eckert4", "azimuthal equal area","azimuthal equidistant", 

               "conic equal area", "conic conformal", "conic equidistant", "gnomonic", "stereographic", 

               "mollweide", "hammer", "transverse mercator", "albers usa", "winkel tripel" ]



buttons = [dict(args = ['geo.projection.type', y],

           label = y, method = 'animate' ) for y in projections]

annot = list([ dict( x=0.1, y=0.8, text='Projection', yanchor='bottom', 

                    xref='paper', xanchor='right', showarrow=False )])





# Update Layout Object



layout[ 'updatemenus' ] = list([ dict( x=0.1, y=0.8, buttons=buttons, yanchor='top' )])

layout[ 'annotations' ] = annot





fig = go.Figure(data = trace1, layout = layout)

iplot(fig)
trace1 = [go.Choropleth(

                colorscale = 'Viridis',

                locationmode = 'country names',

                locations = data3['CountryOrRegion'],

                text = data3['CountryOrRegion'],

                z = data3['PerceptionsOfCorruption'],

)]



layout = dict(title = 'Perceptions Of Corruption',

                geo = dict(

                    showframe=True,

                    showocean=True,

                    showlakes=True,

                    showcoastlines=True,

                    projection = dict(

                        type='hammer'

                    )))



projections = [ "equirectangular", "mercator", "orthographic", "natural earth","kavrayskiy7", 

               "miller", "robinson", "eckert4", "azimuthal equal area","azimuthal equidistant", 

               "conic equal area", "conic conformal", "conic equidistant", "gnomonic", "stereographic", 

               "mollweide", "hammer", "transverse mercator", "albers usa", "winkel tripel" ]



buttons = [dict(args = ['geo.projection.types', y],

               label = y, method = 'animate') for y in projections]

annot = list([dict(x=0.1, y=0.8, text = 'Projection', yanchor='top',

                    xref='paper', xanchor = 'right', showarrow=True)])



# Update Layout Object

layout[ 'updatemenus' ] = list([ dict( x=0.1, y=0.8, buttons=buttons, yanchor='top' )])

layout[ 'annotations' ] = annot





fig = go.Figure(data = trace1, layout = layout)

iplot(fig)
data3.head()