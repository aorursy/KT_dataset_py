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

sns.set_style("whitegrid")



import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

py.init_notebook_mode(connected = True) 



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



df = pd.read_csv("../input/netflix-shows/netflix_titles.csv")

df.head()

df["date_added"] = pd.to_datetime(df['date_added'])

df['year_added'] = df['date_added'].dt.year

df['month_added'] = df['date_added'].dt.month





df['season_count'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" in x['duration'] else "", axis = 1)

df['duration'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" not in x['duration'] else "", axis = 1)

df.head()

#Generate a Wordcloud

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.rcParams['figure.figsize'] = (13, 13)

wordcloud = WordCloud(stopwords=STOPWORDS,background_color = 'white', width = 1000,  height = 1000, max_words = 121).generate(" ".join(df['title']))

  

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Words in Title',fontsize = 30)

plt.show()


labels = 'Movies','TV Series'

sr = pd.Series(df['type'])

x = sr.value_counts()

print(x)

fig1, ax1 = plt.subplots()

ax1.pie(x, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()

from sklearn.preprocessing import MultiLabelBinarizer # Similar to One-Hot Encoding



data= df['listed_in'].astype(str).apply(lambda s : s.replace('&',' ').replace(',', ' ').split()) 



test = data

mlb = MultiLabelBinarizer()

res = pd.DataFrame(mlb.fit_transform(test), columns=mlb.classes_)

corr = res.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(35, 34))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()
temp_df1 = df['release_year'].value_counts().reset_index()





# create trace1

trace1 = go.Bar(

                x = temp_df1['index'],

                y = temp_df1['release_year'],

                marker = dict(color = 'rgb(255,165,0)',

                             line=dict(color='rgb(0,0,0)',width=1.5)))

layout = go.Layout(template= "plotly_dark",title = 'CONTENT RELEASE OVER THE YEAR' , xaxis = dict(title = 'Rating'), yaxis = dict(title = 'Count'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()
df1 = df[df["type"] == "TV Show"]

df2 = df[df["type"] == "Movie"]



temp_df1 = df1['release_year'].value_counts().reset_index()

temp_df2 = df2['release_year'].value_counts().reset_index()





# create trace1

trace1 = go.Bar(

                x = temp_df1['index'],

                y = temp_df1['release_year'],

                name="TV Shows",

                marker = dict(color = 'rgb(249, 6, 6)'))

# create trace2 

trace2 = go.Bar(

                x = temp_df2['index'],

                y = temp_df2['release_year'],

                name = "Movies",

                marker = dict(color = 'rgb(26, 118, 255)'))





layout = go.Layout(template= "plotly_dark",title = 'CONTENT RELEASE OVER THE YEAR BY CONTENT TYPE' , xaxis = dict(title = 'Year'), yaxis = dict(title = 'Count'))

fig = go.Figure(data = [trace1, trace2], layout = layout)

fig.show()
trace = go.Histogram(

                     x = df['duration'],

                     xbins=dict(size=0.5),

                     marker = dict(color = 'rgb(26, 118, 255)'))

layout = go.Layout(template= "plotly_dark", title = 'Distribution of Movies Duration', xaxis = dict(title = 'Minutes'))

fig = go.Figure(data = [trace], layout = layout)

fig.show()
temp_df = df['country'].value_counts().reset_index()[:20]





# create trace1

trace1 = go.Bar(

                x = temp_df['index'],

                y = temp_df['country'],

                marker = dict(color = 'rgb(153,255,153)',

                              line=dict(color='rgb(0,0,0)',width=1.5)))

layout = go.Layout(template= "plotly_dark",title = 'TOP 20 COUNTIES WITH MOST CONTENT' , xaxis = dict(title = 'Countries'), yaxis = dict(title = 'Count'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()
new_df = df[['title','director','cast','listed_in','description']]

new_df.head()

!pip install rake-nltk



from rake_nltk import Rake

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer



#Clean Data

new_df.dropna(inplace = True)



blanks = []  # start with an empty list



col=['title','director','cast','listed_in','description']

for i,col in new_df.iterrows():  # iterate over the DataFrame

    if type(col)==str:            # avoid NaN values

        if col.isspace():         # test 'review' for whitespace

            blanks.append(i)     # add matching index numbers to the list



new_df.drop(blanks, inplace=True)
# initializing the new column

new_df['Key_words'] = ""



for index, row in new_df.iterrows():

    description = row['description']

    

    # instantiating Rake, by default it uses english stopwords from NLTK

    # and discards all puntuation characters as well

    r = Rake()



    # extracting the words by passing the text

    r.extract_keywords_from_text(description)



    # getting the dictionary whith key words as keys and their scores as values

    key_words_dict_scores = r.get_word_degrees()

    

    # assigning the key words to the new column for the corresponding movie

    row['Key_words'] = list(key_words_dict_scores.keys())



# dropping the Plot column

new_df.drop(columns = ['description'], inplace = True)

new_df['cast'] = new_df['cast'].map(lambda x: x.split(',')[:3])



new_df['listed_in'] = new_df['listed_in'].map(lambda x: x.lower().split(','))



new_df['director'] = new_df['director'].map(lambda x: x.split(' '))



for index, row in new_df.iterrows():

    row['cast'] = [x.lower().replace(' ','') for x in row['cast']]

    row['director'] = ''.join(row['director']).lower()

    

new_df.set_index('title', inplace = True)
new_df['bag_of_words'] = ''

columns = new_df.columns

for index, row in new_df.iterrows():

    words = ''

    for col in columns:

        if col != 'director':

            words = words + ' '.join(row[col])+ ' '

        else:

            words = words + row[col]+ ' '

    row['bag_of_words'] = words

    

new_df.drop(columns = [col for col in new_df.columns if col!= 'bag_of_words'], inplace = True)
new_df.head()
count = CountVectorizer()

count_matrix = count.fit_transform(new_df['bag_of_words'])



indices = pd.Series(new_df.index)

indices[:5]
cosine_sim = cosine_similarity(count_matrix, count_matrix)

cosine_sim
def recommendations(Title, cosine_sim = cosine_sim) :

    

    recommended_movies = []    

    idx = indices[indices == Title].index[0]

    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)



    top_10_indexes = list(score_series.iloc[1:11].index)

    

    for i in top_10_indexes:

        recommended_movies.append(list(new_df.index)[i])

        

    return recommended_movies
recommendations('Love')