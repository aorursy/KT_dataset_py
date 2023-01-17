# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will l

#ist the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import seaborn as sns

# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

movies = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_movies.csv")

happy2015 = pd.read_csv("../input/world-happiness/2015.csv")

print(happy2015.info())



#dataset looks normal since there are 158 entries and all of them are not null. 

#Also strings are in object form and numerical values are in float or int form which is good

# Since I do not want work with long column name whit spaces I created new column with same values then drop the long column named column.

happy2015['GDP'] = happy2015['Economy (GDP per Capita)']



happy2015.drop(['Economy (GDP per Capita)'],axis = 1,inplace=True)



#Sorting according to GDP . From High GDP to LOW  GDP

new_index = happy2015['GDP'].sort_values(ascending = False).index.values

sorted_happy2015 = happy2015.reindex(new_index)


#BAR PLOT

# In here I tried to visualize countries with respect to its GDP's



plt.figure(figsize = (45,25))

sns.barplot(x = sorted_happy2015['Country'] , y =sorted_happy2015['GDP'] )

plt.title("Countries vs GDP per Capita")

plt.xticks(rotation = 90)

plt.xlabel("Countries")

plt.ylabel("GDP")

plt.show()



#%%

#In here I tried to understand which region is the happiest in 2015

trace1 = go.Histogram(

    x = happy2015['Region'],

    opacity=0.75,

    name = "Reg",

)

data = [trace1]

layout = go.Layout(barmode='overlay',

                   title=' #Number of Happy Countries in Region',

                   xaxis=dict(title='Regions'),

                   yaxis=dict( title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
#%% Correlation Map

f,ax = plt.subplots( figsize = (5,5))

sns.heatmap(happy2015.corr() , annot = True , linewidths = 2 ,linecolor = 'blue', fmt = '.1f' , ax = ax )

plt.title("Correlation Matrix")

plt.show()





#%% Point Plots     #Point Plot data Preparation



#SORTİNG according to Happines Score

new_index2 = happy2015['Happiness Score'].sort_values(ascending = False).index.values

sorted2_happy2015 = happy2015.reindex(new_index2)



#NORMALIZATATION

sorted_happy2015['GDP'] = sorted_happy2015['GDP'] /max(sorted_happy2015['GDP']) 

sorted2_happy2015['Happiness Score'] = sorted2_happy2015['Happiness Score'] / max(sorted_happy2015['Happiness Score'])



data2 = pd.concat([sorted_happy2015,sorted2_happy2015['Happiness Score']],axis = 1,)

column_names = data2.columns.values

column_names[3] = "CHANGED"

data2.columns = column_names

data2.drop(["CHANGED"],inplace = True,axis=1)


#%%

#In this plot I have tried to see whether is there any relationship between Happiness score and GDP ratio

# Point Plot visualization

f,ax1 = plt.subplots(figsize = (30,10))

sns.pointplot(x='Region' , y ='GDP',data = data2 , color = 'lime' ,alpha = 0.8)

sns.pointplot(x = 'Region' , y = 'Happiness Score' ,data = data2 , color = 'red' , alpha = 0.8 )

plt.title('GDP vs Happiness Score ',fontsize = 20 , color = 'blue' )



plt.text(8,0.75,'Happiness Score', color = 'red' , fontsize = 17 ,style = 'italic' )

plt.text(8,0.70,'GDP ratio' , color = 'lime',fontsize = 17 , style = 'italic')



plt.xlabel('Regions',fontsize = 15 ,color = 'blue' )

plt.ylabel ('Values' , fontsize = 15 , color ='blue')

plt.grid()

plt.show()
#%% SNS lmplot

#For understand distribution and correlation of GDP and Happiness Score

sns.lmplot(x = 'GDP',y ='Happiness Score',data = data2)

#%% SNS KDE PLOT

#For understand distribution and correlation of GDP and Family

sns.kdeplot(data2.GDP,data2.Family,shade = True ,cut = 1)
#%%

#Another visualization For understand distribution and correlation of GDP and Happiness Score

sns.kdeplot(data2.GDP,data2['Happiness Score'],shade = True , cut = 1)

#Horizontal Bar



#Creating list for unique Regions

region_list = data2['Region'].unique()



f,ax = plt.subplots(figsize=(9,15))



sns.barplot(x='Family', y='Region', data = data2,color = 'green' , alpha = 0.5 , label = 'Family')

sns.barplot(x='Freedom', y='Region', data = data2,color = 'blue' , alpha = 0.5 , label = 'Freedom')

sns.barplot(x='Generosity', y='Region', data = data2,color = 'purple' , alpha = 0.5 , label = 'Generosity')

sns.barplot(x='Trust (Government Corruption)', y='Region', data = data2,color = 'red' , alpha = 0.5 , label = 'Trust (Government Corruption)')



ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu framen gorunurlugu bölme 

ax.set(xlabel='Percentage', ylabel='Region',title = "Percentage of properties According to States ")

#%% Joint Plots Freedom vs Corruption



g = sns.jointplot(x = data2.Freedom , y = data2['Trust (Government Corruption)'], kind ='kde' ,size = 7 )

plt.show()

#%% Another Joint plot 

#For understand distribution and correlation of Freedom and Trust( Government Corruption ) Score

g = sns.jointplot(x = data2.Freedom , y = data2['Trust (Government Corruption)'], kind ='scatter' ,size = 7 )

plt.show()
#%% Another Joint plot

#Different version of upper plot with same data

g = sns.jointplot(x = data2.Freedom , y = data2['Trust (Government Corruption)'], kind ='reg' ,size = 7 )

plt.show()
#%% Another Joint plot

#Different version of upper plot with same data

g = sns.jointplot(x = data2.Freedom , y = data2['Trust (Government Corruption)'], kind ='resid' ,height = 7 )

plt.show()
#%% Another Joint plot

#Different version of upper plot with same data

g = sns.jointplot(x = data2.Freedom , y = data2['Trust (Government Corruption)'], kind ='hex' ,size = 7 )

plt.show()
#%% data3 is dataframe with family and happiness score



data3 = data2[['Family','Happiness Score']]

data3['Family'] = data3['Family']/max(data3['Family'])



sns.pairplot(data3)

plt.show()
#To see distriubt

sns.jointplot(x = data2.Family, y = data2['Happiness Score'], kind = 'kde' , height = 7 )


#%%  JointPlot with another example



g = sns.jointplot("Happiness Score", "Family", data=data3,size=5, ratio=3, color="r")
#%%  Pie Chart  matplot visualization



#Distribution of regions of first 30 happiest countries



#DATA PREPARATİON

happy2015.Region.dropna(inplace = True)

happy2015_top30 = happy2015.iloc[0:30,:]

labels  = happy2015_top30.Region.value_counts().index

explode  = [0,0,0,0,0,0]



sizes = happy2015_top30.Region.value_counts().values







#Visualization 

plt.figure( figsize = (7,7))

plt.pie(sizes, explode = explode ,labels=labels , autopct = '%1.1f%%')

plt.show()
#%% Count Plot



#In here I classified the countries according to their GDP's as rich and poor . 

#Then I tried to see how many of the countries rich in the dataframe . 

#It can bee seen there are more rich countries than poor countries in our data

avg = np.mean(happy2015.GDP) #average with numpy method



Richness = ["rich" if each > avg else "poor" for each in happy2015.GDP ] #classification according to GDP

dataframe = pd.DataFrame({ 'richness' : Richness})

sns.countplot(x = dataframe.richness)





#%% Violin Plot

pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=data3, palette=pal, inner="points") #sadece sayısal verileri alıp gösteriyor 

#en şişman olduğu yerler datanın en çok nerede olduğunu gösteriyor

plt.show()
movies.info()
#AFTER THIS PART , we are going with movies dataset

#DATA CLEANING

#to_drop = ['homepage','id','tagline','spoken_languages']

#movies.drop(to_drop,inplace = True , axis = 1)

        

#movies[movies.budget == 0]

movies.budget.replace(0,"NA",inplace = True)

movies.dropna(axis = 0,inplace = True)



df  = movies[movies.budget !="NA"]

df.reset_index(inplace  = True)

df = df.drop(['index'] , axis = 1)



df.budget = df.budget.astype(float)


new_index = (df['popularity'].sort_values(ascending = False).index.values)



sortedDfPop = df.reindex(new_index)



dfGraph = sortedDfPop[['original_title','popularity','vote_count']]

dfGraph = dfGraph.iloc[:100,:]



dfGraph['popularity'] = dfGraph['popularity'] / max(dfGraph['popularity']) #Normalization

dfGraph['vote_count'] = dfGraph['vote_count'] / max(dfGraph['vote_count']) #Normalization
#Plotly Line Plot



#Creating trace1 

trace1 = go.Scatter(

        x = dfGraph.original_title,

        y = dfGraph.popularity,

        mode = "lines",

        name ="popularity",

        marker = dict(color = 'rgba(16,112,2,0.8)'),

        text = dfGraph.original_title)



trace2 = go.Scatter(

        x = dfGraph.original_title,

        y = dfGraph.vote_count,

        mode ="lines",

        name = "vote_count",

        text = dfGraph.original_title)

data = [trace1 , trace2]



layout = dict (title = 'Percentage of popularity vs vote_count of Top 100 Movies',

               xaxis = dict(title = 'Movies',ticklen = 5 , zeroline =False),

               yaxis = dict(title ="Percentages",ticklen=20, zeroline = True))



fig = dict(data = data , layout = layout)

iplot(fig,filename='simple-line')

#%% Plotly Bar Plot



# DATA PREPARATION



sortedDfPop.reset_index(inplace = True)

sortedDfPop = sortedDfPop.drop(['index'] , axis = 1)

date = []

for each in sortedDfPop['release_date']:

    date.append(each.split("-")[0])

sortedDfPop["date"] = date

#%% Plotly Bar Plot 



top5 = sortedDfPop[:5]

top5



trace1 = go.Bar(

        x = top5.original_title,

        y = top5.revenue,

        name = "revenue",

        marker = dict( color = 'rgba(255,174,255,0.5)',

                      line = dict(color = 'rgba(0,0,0,1)',

                      width = 0.5)),

                      text=top5.date,

        )

        

trace2 = go.Bar(

        x = top5.original_title,

        y = top5.budget,

        name = "budget",

        marker = dict ( color = 'rgba(0,255,146,0.5)',

                       line = dict(color = 'rgba(0,0,0,1)',

                                   width = 0.5)),

                       text = top5.date)

data=[trace1,trace2]

layout = go.Layout(

        barmode = "group",

        title = "Top5 Popular Movies's Revenues vs budget")

fig = go.Figure(data = data , layout =layout)

iplot(fig)
#%% Farklı bir Bar plot





# prepare data frames

from plotly import tools

import matplotlib.pyplot as plt



y_saving =[ each for each in top5.revenue] 

y_net_worth  = [float(each) for each in top5.budget]

x_saving = [each for each in top5.original_title]

x_net_worth  = [each for each in top5.original_title]



trace0 = go.Bar(

                x=y_saving,

                y=x_saving,

                marker=dict(color='rgba(171, 50, 96, 0.6)',line=dict(color='rgba(171, 50, 96, 1.0)',width=1)),

                name='research',

                orientation='h',

)

trace1 = go.Scatter(

                x=y_net_worth,

                y=x_net_worth,

                mode='lines+markers',

                line=dict(color='rgb(63, 72, 204)'),

                name='income',

)



layout = dict(

                title='Citations and income',

                yaxis=dict(showticklabels=True,domain=[0, 0.85]),

                yaxis2=dict(showline=True,showticklabels=False,linecolor='rgba(102, 102, 102, 0.8)',linewidth=2,domain=[0, 0.85]),

                xaxis=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0, 0.42]),

                xaxis2=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0.47, 1],side='top',dtick=25),

                legend=dict(x=0.029,y=1.038,font=dict(size=10) ),

                margin=dict(l=200, r=20,t=70,b=70),

                paper_bgcolor='rgb(248, 248, 255)',

                plot_bgcolor='rgb(248, 248, 255)',

)



annotations = []

y_s = np.round(y_saving, decimals=2)

y_nw = np.rint(y_net_worth)

# Adding labels

for ydn, yd, xd in zip(y_nw, y_s, x_saving):

    # labeling the scatter savings

    annotations.append(dict(xref='x2', yref='y2', y=xd, x=ydn - 4,text='{:,}'.format(ydn),font=dict(family='Arial', size=12,color='rgb(63, 72, 204)'),showarrow=False))

    # labeling the bar net worth

    annotations.append(dict(xref='x1', yref='y1', y=xd, x=yd + 3,text=str(yd),font=dict(family='Arial', size=12,color='rgb(171, 50, 96)'),showarrow=False))



layout['annotations'] = annotations





# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,

                          shared_yaxes=False, vertical_spacing=0.001)



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)



fig['layout'].update(layout)

iplot(fig)

#%% Pie Chart Plotly 



uniqueYears = sortedDfPop.date.unique()

movie2014 = sortedDfPop[sortedDfPop.date == "2014"].iloc[:10,:]

movie2015 = sortedDfPop[sortedDfPop.date == "2015"].iloc[:10,:]

movie2016 = sortedDfPop[sortedDfPop.date == "2016"].iloc[:10,:]



pop2014 = sum(movie2014.popularity)/len(movie2014.popularity)

pop2015 = sum(movie2015.popularity)/len(movie2015.popularity)

pop2016 = sum(movie2016.popularity)/len(movie2016.popularity)





labels = ['Top10Popularity-2014','Top10Popularity-2015','Top10Popularity-2016']



pie_list = [pop2014,pop2015,pop2016]



colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']





trace1  = go.Pie(labels = labels ,

                 values =pie_list ,

                 hoverinfo = "label+percent+value",

                 textinfo = "label+percent",

                 textfont = dict(size = 20),

                 hole = .1,

                 marker=dict(colors=colors,line=dict(color='#000000', width=2))

                                                     )

                                     

layout = go.Layout(

        title = "Avg.Popularity Of Top10Movies with respect to Years",

        annotations = [

                       {

                       "font":  {"size" : 14},

                       "showarrow" : False ,

                       "text" : "Popularity",

                       "x" : 0.30,

                       "y" : 1 

                       }])

                       



data = [trace1]



fig  = go.Figure( data = data , layout = layout)

iplot(fig)

#data preparation

first20 = movies.iloc[:20,:]

first20.budget =first20.budget.astype(int)

first20.vote_average = first20.vote_average.astype(int)



#%% Buble Charts



moviesBudget  = [float(each)/5000000for each in first20.budget]

runtime = [float(each) for each in first20.runtime]

data = [

    {

        'y': first20.vote_average,

        'x': first20.title,

        'mode': 'markers',

        'marker': {

            'color': runtime,

            'size': moviesBudget,

            'showscale': True,

        },

        "text" :  first20.title

    }

]



iplot(data)
#%% Word Cloud

from wordcloud import WordCloud

from collections import Counter



word_list = []



for index,row in movies.iterrows():

    all_words = row['overview'].split(" ")

    for each in all_words :

        word_list.append(each)



word_count1 = Counter(word_list)

most_common_names =word_count1.most_common(90)



most_common_names = most_common_names[33:]

#%%

plt.subplots(figsize = (9,9))

wordcloud = WordCloud(

        background_color = "white",

        width = 512 ,

        height = 384).generate(" ".join(word_list))

plt.imshow(wordcloud) #img ı plot ettirmek için kullanılıyor resim bu aslında

plt.axis('on')

plt.show()
#%% box Plot





trace1 = go.Box(

        y = sortedDfPop.budget/max(sortedDfPop.budget),

        name = "distribution of budget of movies",

        marker = dict(color = 'rgba(220,54,22,0.6)')

)

trace2 = go.Box(

        y = sortedDfPop.revenue/max(sortedDfPop.revenue),

        name = "dist of revenue of movies",

        marker = 

        dict(color = 'rgb(12,128,128)')

        )

data = [trace1,trace2]

fig = go.Figure(data = data)

iplot(fig)



#%% Scatter matrix

#is used to see covariance and relation between more than 2 features



import plotly.figure_factory as ff



sortedDfPop['budget'] = sortedDfPop['budget'].astype(float)/max(sortedDfPop['budget'])

sortedDfPop['revenue'] = sortedDfPop['revenue'].astype(float)/max(sortedDfPop['revenue'])

sortedDfPop['popularity'] = sortedDfPop['popularity'].astype(float)/max(sortedDfPop['popularity'])



moviesData = sortedDfPop.loc[:,["budget","revenue","popularity"]]



moviesData["index"] = np.arange(1,len(moviesData)+ 1)



fig = ff.create_scatterplotmatrix(moviesData, diag = 'box' , index = 'index' , colormap ='Portland',

                                  colormap_type = 'cat',

                                  height = 700 , width = 700)



iplot(fig)



#%% Inset Plot 2 Plots in one frame



#first line plot



movies30 = movies.iloc[:150,:]

trace1  = go.Scatter(

        x= movies30.original_title,

        y= movies30.budget,

        name = "budget",

        marker = dict(color = 'rgba(16 , 112, 2 , 0.8)')

        )



trace2 = go.Scatter( 

        x = movies30.original_title,

        y=movies30.popularity,

        xaxis = 'x2',

        yaxis = 'y2',

        name = "revenue",

        marker = dict(color = 'rgba(160,112,20,0.8)')

        )



data = [trace1,trace2]

layout = go.Layout(

        xaxis2 = dict(

                domain = [0.6,0.95],

                anchor = "y2",),

        yaxis2 = dict(

                domain = [0.7,0.95],

                anchor = 'x2'),

                title = "budget vs popularity")

        

        

        

fig = go.Figure(data = data , layout = layout)



iplot(fig)

#%% multiple subplots

#While comparing more than 1 plot 

 

trace1 = go.Scatter(

        x = movies.original_title,

        y = movies.popularity ,

        name = "popularity" ,

        )





trace2 = go.Scatter(

        x = movies.original_title,

        y = movies.budget ,

        name = "budget" ,

        xaxis = "x2",

        yaxis = "y2" ,

        )





trace3 = go.Scatter(

        x = movies.original_title,

        y = movies.revenue ,

        name = "revenue" ,

        xaxis = "x3",

        yaxis = "y3" ,

        )





trace4 = go.Scatter(

        x = movies.original_title,

        y = movies.vote_count ,

        name = "vote_count" ,

        xaxis = "x4",

        yaxis = "y4" ,

        )



data = [trace1 , trace2 , trace3 , trace4]



layout = go.Layout(

        xaxis = dict(

                domain = [0,0.45]

                ),

                

        yaxis = dict(

                domain = [0,0.45]

                ),

        xaxis2 = dict(

                domain = [0.55,1]

                ),

        xaxis3 = dict(

                domain = [0,0.45],

                anchor = "y3"

                ),

        xaxis4 = dict(

                domain = [0.55,1],

                anchor = "y4",

                ),

                

        yaxis2 = dict(

                domain = [0,0.45],

                anchor = "x2"),

        

                

        yaxis3 = dict(

                domain = [0.55,1]

                ),

        yaxis4 = dict(

                domain = [0.55,1],

                anchor = "x4",

        

                )        

        )



fig = go.Figure( data = data , layout = layout)

iplot(fig )