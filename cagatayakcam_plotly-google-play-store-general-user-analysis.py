# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly import tools

import plotly.figure_factory as ff

from plotly.offline import init_notebook_mode, iplot

%matplotlib inline 
df = pd.read_csv('../input/googleplaystore.csv')

df1= pd.read_csv('../input/googleplaystore_user_reviews.csv')
len(df.App.unique())
df.head()
#Lets take a look to missing data.

df.info()

pd.concat([df.isnull().sum(), 100 * df.isnull().sum()/len(df)], 

              axis=1).rename(columns={0:'Missing Records', 1:'Percentage (%)'})
df.Size.value_counts()
df['Category'].value_counts()
#data manupitaion and data clearance

df.Size = df.Size.str.replace("M", "MB")

df.Size = df.Size.str.replace("k", "KB")

df = df[~(df.Category.isin(['1.9']))]

df = df[~(df.Rating.isin(['NaN']))]



df.Installs = df.Installs.str.replace("+", "")

df.Installs = df.Installs.str.replace(",", "")

df.Installs = df.Installs.astype("float")



df["categoryInstallSum"] = df.groupby('Category').Installs.transform('sum')

df["categoryCount"] = df.groupby('Category').Installs.transform('count')

df['meanInstall'] = df['categoryInstallSum'] / df['categoryCount']
colors = ['aqua', 'lightgrey', 'lightgreen', '#D0F9B1', 'khaki', 'grey']





Category = df['Category'].value_counts()[:15]

label = Category.index

size = Category.values



trace = go.Pie(labels=label, 

               values=size, 

               marker=dict(colors=colors))



data = [trace]

layout = go.Layout(title=' App Category Distribution')



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
colors = ['aqua', 'lightgrey', 'lightgreen', '#D0F9B1', 'khaki', 'grey']





df_cat_user = df.drop_duplicates('Category')[['Category', 'meanInstall']].sort_values('meanInstall', 

                                                                                      

                                                                                      ascending=False)[:15]

label = df_cat_user.Category

size = df_cat_user.meanInstall



trace = go.Pie(labels=label, 

               values=size, 

               marker=dict(colors=colors))



data = [trace]

layout = go.Layout(title='User Pie Distribution')



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
Content = df['Content Rating'].str.split(',')

Content_set = []

for i in Content.dropna():

    Content_set.extend(i)

    Content = pd.Series(Content_set).value_counts()[:6] 

    

label = Content.index

size = Content.values



colors = ['#FEBFB3', 'skyblue', '#96D38C', '#D0F9B1', 'tan', 'lightgrey']



trace = go.Pie(labels=label, 

               values=size, 

               marker=dict(colors=colors)

              )



data = [trace]

layout = go.Layout(

    title='User Audience ', 

    legend=dict(orientation="h")

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
Type = df['Type'].value_counts()

label = Type.index

size = Type.values

colors = ['skyblue', 'lightgreen']



trace = go.Pie(labels=label, 

               values=size, 

               marker=dict(colors=colors)

              )



data = [trace]

layout = go.Layout(title='Type',

                   legend=dict(orientation="h")

                  )



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
df.Reviews = df.Reviews.astype("float")

data = df[df.Reviews>1000000][['Reviews','Rating']]

data["index"] = np.arange(len(data))



fig = ff.create_scatterplotmatrix(data, diag='box', index='index',size= 3,

                               height=700, width=700, colormap='RdBu')

py.iplot(fig)
plt.figure(figsize=(15,6))

sns.boxplot(x = "Category", y = "Rating",palette="tab20", data= df)

plt.xticks(rotation=80)

plt.show()
plt.figure(figsize=(15,6))



sns.kdeplot(df[df.Category=='FAMILY'].Rating,label="FAMILY")

sns.kdeplot(df[df.Category=='GAME'].Rating,label="GAME")

sns.kdeplot(df[df.Category=='TOOLS'].Rating,label="TOOLS")

sns.kdeplot(df[df.Category=='MEDICAL'].Rating,label="MEDICAL")

sns.kdeplot(df[df.Category=='COMMUNICATION'].Rating,label="COMMUNICATION")





plt.legend();
Rating = round(df['Rating'].value_counts(normalize=True), 4)

trace = go.Bar(

    x=Rating.index,

    y=Rating.values,

    marker=dict(

        color = Rating.values,

        colorscale='Reds',

        showscale=True)

)



data = [trace]

layout = go.Layout(title='Rating distribution', 

                       yaxis = dict(title = '% of App')

                  )



fig = go.Figure(data=data, layout=layout)

fig['layout']['xaxis'].update(dict(title = 'Rating', 

                                   tickfont = dict(size = 12)))

py.iplot(fig)
hist_data = [df[df.Type == 'Paid'].Rating, df[df.Type == 'Free'].Rating]



labels = ['Paid', 'Free']

colors = ['navy', 'red']



fig = ff.create_distplot(hist_data, labels, colors=colors,

                         show_hist=False, bin_size=.2)





fig['layout'].update(title='Rating by Type')

py.iplot(fig)
dfchart = df.groupby(['Category', 'Type']).agg({'App' : 'count'}).reset_index()



outer_circle = ['GAME', 'FAMILY', 'MEDICAL', 'TOOLS']

outer_circle_values = [len(df[df.Category == category]) for category in outer_circle]



a,b,c,d =[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples]



inner_circle_names = ['Paid', 'Free'] * 4

inner_circle_values = []



for category in outer_circle:

    for t in ['Paid', 'Free']:

        x = dfchart[dfchart.Category == category]

        try:

            inner_circle_values.append(int(x.App[x.Type == t].values[0]))

        except:

            inner_circle_values.append(0)



explode = (0.015,0.015,0.015,0.015)



# Outer ring

fig, ax = plt.subplots(figsize=(10,10))

ax.axis('equal')

mypie, texts, _ = ax.pie(outer_circle_values, radius=1.5, labels=outer_circle, autopct='%1.1f%%', pctdistance=1.1,

                                 labeldistance= 0.75,  explode = explode, colors=[a(0.6), b(0.6), c(0.6), d(0.6)], textprops={'fontsize': 12})

plt.setp( mypie, width=1, edgecolor='black')

 

# Inner ring

mypie2, _ = ax.pie(inner_circle_values, radius=0.7, labels=inner_circle_names, labeldistance= 0.7, 

                   textprops={'fontsize': 12}, colors = [a(0.4), a(0.2), b(0.4), b(0.2), c(0.4), c(0.2), d(0.4), d(0.2)])

plt.setp( mypie2, width=0.7, edgecolor='black')

 

plt.show()
dflast = pd.merge(df, df1, on = "App", how = "inner")

dflast = dflast.dropna(subset=['Sentiment', 'Translated_Review'])
grouped_sentiment_category_count = dflast.groupby(['Category', 'Sentiment']).agg({'App': 'count'}).reset_index()

grouped_sentiment_category_sum = dflast.groupby(['Category']).agg({'Sentiment': 'count'}).reset_index()



df_review = pd.merge(grouped_sentiment_category_count, grouped_sentiment_category_sum, on=["Category"])



df_review['Sentiment_Normalized'] = df_review.App/df_review.Sentiment_y

df_review = df_review.groupby('Category').filter(lambda x: len(x) ==3)



df_review



trace1 = go.Bar(

    x=list(df_review.Category[::3])[6:-5],

    y= df_review.Sentiment_Normalized[::3][6:-5],

    name='Negative',

    marker=dict(color = 'Red')

)



trace2 = go.Bar(

    x=list(df_review.Category[::3])[6:-5],

    y= df_review.Sentiment_Normalized[1::3][6:-5],

    name='Neutral',

    marker=dict(color = 'seashell')

)



trace3 = go.Bar(

    x=list(df_review.Category[::3])[6:-5],

    y= df_review.Sentiment_Normalized[2::3][6:-5],

    name='Positive',

    marker=dict(color = 'Green')

)



data = [trace1, trace2, trace3]

layout = go.Layout(

    title = 'Review analysis',

    barmode='stack',

    xaxis = {'tickangle': -45},

    yaxis = {'title': 'Breakdown of reviews'}

)



fig = go.Figure(data=data, layout=layout)



plotly.offline.iplot({'data': data, 'layout': layout})