import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



from wordcloud import WordCloud



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("../input/wiki_movie_plots_deduped.csv")
# Show first 10 rows of data

data.head(10)
# Shape of data

data.shape
# Describe the data

data.describe()
# Info of data

data.info()
# Show columns of data

data.columns
data.dtypes
ax = data['Origin/Ethnicity'].value_counts().sort_index().plot.bar(

    figsize = (10, 5),

    fontsize = 14)



ax.set_title("Count of Origin/Ethnicity of Movies", fontsize=16)

plt.xlabel('Origin/Ethnicity', fontsize=20)

plt.ylabel('Counts', fontsize=20)

sns.despine(bottom=True, left=True)
ax = data['Release Year'].value_counts().sort_index(ascending=True)



Sct = [go.Scatter(x = ax.index, y = ax.values, mode = 'lines', name = 'lines')]

layout = go.Layout(title = 'Movies by year')

fig = go.Figure(data = Sct, layout = layout)

iplot(fig)
wordcloud = WordCloud(width = 1000, height = 600, max_font_size = 120, max_words = 50).generate(" ".join(data.Plot))



plt.subplots(figsize=(18,8))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
plt.figure(figsize=(16,8))

plt.title('Top titles',fontsize=25)

plt.xlabel('Title', fontsize=30)



sns.countplot(data.Title,order=pd.value_counts(data.Title).iloc[:15].index,palette=sns.color_palette("plasma", 15))



plt.xticks(size=16,rotation=90)

plt.yticks(size=16)

sns.despine(bottom=True, left=True)

plt.show()
# Removing unknown genres

Gen = data[data.Genre != "unknown"]



plt.figure(figsize=(16,8))

plt.title('Most frequent Genre types',fontsize=30)

plt.xlabel('Genre', fontsize=25)

plt.ylabel('Count', fontsize=25)



sns.countplot(Gen.Genre,order=pd.value_counts(Gen.Genre).iloc[:15].index,palette=sns.color_palette("copper", 15))



plt.xticks(size=16,rotation=90)

plt.yticks(size=16)

sns.despine(bottom=True, left=True)

plt.show()
# from https://www.kaggle.com/tatianasnwrt/wikipedia-movie-plots-eda



# Getting rid of null values and invisible characters (non-breaking spaces) 

top_cast = data[(data.Cast.notnull()) & (data.Cast != " ")] 

top_cast.set_index("Cast",inplace=True) 

top_cast.rename(index={'Three Stooges':'The Three Stooges'},inplace=True)



plt.figure(figsize=(22,15))

plt.title('Top Cast',fontsize=30)



sns.countplot(y=top_cast.index,order=pd.value_counts(top_cast.index)[:20].index,palette=sns.color_palette("rocket", 25)) 



plt.ylabel('Cast',fontsize=30)

plt.xlabel('Number of movies participated',fontsize=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
# Removing Unknown directors to get a clear picture of our data

Dir = data[data.Director != "Unknown"]



plt.figure(figsize=(22,8))

plt.title('Top Directors',fontsize=30)



sns.countplot(Dir.Director, order=pd.value_counts(Dir.Director)[:20].index, palette=sns.color_palette("seismic", 20))



plt.xlabel('Directors',fontsize=25)

plt.ylabel('Number of movies directed',fontsize=25)

plt.xticks(size=16,rotation=90)

plt.show()