# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
'''Customize visualization

Seaborn and matplotlib visualization.'''

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

%matplotlib inline



'''Plotly visualization .'''

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

py.init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook



import cufflinks as cf #importing plotly and cufflinks in offline mode  

import plotly.offline  

cf.go_offline()  

cf.set_config_file(offline=False, world_readable=True)



'''Display markdown formatted output like bold, italic bold etc.'''

from IPython.display import Markdown

def bold(string):

    display(Markdown(string))
df = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")
df.head()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.rcParams['figure.figsize'] = (13, 13)

wordcloud = WordCloud(stopwords=STOPWORDS,background_color = 'black', width = 1000,  height = 1000, max_words = 121).generate(' '.join(df['title']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Words in Title',fontsize = 30)

plt.show()
df.info()
df.isnull().sum(axis = 0)
df = df.replace(np.nan, "No Info")

#fillna()
df.head()
country_list = ['South Korea','Japan','United Kingdom','United States','China','India']

sel_df = df[df['country'].str.contains('|'.join(country_list))]
sel_df.head()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.rcParams['figure.figsize'] = (13, 13)

wordcloud = WordCloud(stopwords=STOPWORDS,background_color = 'black', width = 1000,  height = 1000, max_words = 121).generate(' '.join(sel_df['title']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Words in Title',fontsize = 30)

plt.show()
new_df = sel_df[['title','director','cast','listed_in','description','rating']]

new_df.head()
!pip install rake-nltk

from rake_nltk import Rake

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer
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
new_df['cast'] = new_df['cast'].map(lambda x: x.lower().split(','))



# putting the genres in a list of words

new_df['listed_in'] = new_df['listed_in'].map(lambda x: x.lower().split(','))

new_df['director'] = new_df['director'].map(lambda x: x.lower().split(','))

new_df.head()
# merging together first and last name for each actor and director, so it's considered as one word 

# and there is no mix up between people sharing a first name

for index, row in new_df.iterrows():

    row['cast'] = [x.lower().replace(" ","") for x in row['cast']]

    row['director'] = [x.lower().replace(" ","") for x in row['director']]
new_df.head()
new_df.set_index('title', inplace = True)

new_df.head()
new_df['bag_of_words'] = ''

columns = new_df.columns

for index, row in new_df.iterrows():

    words = ''

    for col in columns:

        words = words + ' '.join(row[col])+ ' '

    row['bag_of_words'] = words

    

new_df.drop(columns = [col for col in new_df.columns if col!= 'bag_of_words'], inplace = True)
new_df.head()
# instantiating and generating the count matrix

count = CountVectorizer()

count_matrix = count.fit_transform(new_df['bag_of_words'])



# creating a Series for the movie titles so they are associated to an ordered numerical

# list I will use later to match the indexes

indices = pd.Series(new_df.index)

indices[:5]
import tensorflow_hub as hub



embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
new_df['vector']=new_df['bag_of_words'].apply(lambda x: embed([x]).numpy()[0])
new_df.head()
from sklearn.metrics.pairwise import euclidean_distances

distances = euclidean_distances(list(new_df['vector']))
cos_sim = cosine_similarity(list(new_df['vector']))
cosine_sim = cosine_similarity(count_matrix, count_matrix)
recommended_movies = set()
def recommendations_dist(Title, dist, recommended_movies):

    

    # gettin the index of the movie that matches the title

    idx = indices[indices == Title].index[0]



    # creating a Series with the similarity scores in descending order

    score_series = pd.Series(dist[idx]).sort_values(ascending = True)



    # getting the indexes of the 10 most similar movies

    top_10_indexes = list(score_series.iloc[0:10].index)

    

    # populating the list with the titles of the best 10 matching movies

    for i in top_10_indexes:

        recommended_movies.add(list(new_df.index)[i])

        

    return recommended_movies - set([Title])
def recommendations_sim(Title, sim, recommended_movies):

    

    # gettin the index of the movie that matches the title

    idx = indices[indices == Title].index[0]



    # creating a Series with the similarity scores in descending order

    score_series = pd.Series(sim[idx]).sort_values(ascending = False)



    # getting the indexes of the 10 most similar movies

    top_10_indexes = list(score_series.iloc[1:11].index)

    

    # populating the list with the titles of the best 10 matching movies

    for i in top_10_indexes:

        recommended_movies.add(list(new_df.index)[i])

        

    return recommended_movies

        
recommended_movies =recommendations_dist('Stranger', distances, recommended_movies)
recommended_movies = recommendations_sim('Stranger', cosine_sim, recommended_movies)
recommended_movies = recommendations_sim('Stranger', cos_sim, recommended_movies)

recommended_movies
def recommend_list(Title):

    recommended_movies = set()

    recommended_movies = recommendations_dist(Title, distances, recommended_movies)

    recommended_movies = recommendations_sim(Title, cosine_sim, recommended_movies)

    recommended_movies = recommendations_sim(Title, cos_sim, recommended_movies)

    

    return list(recommended_movies)

    

    
recommend_list('When the Camellia Blooms')
recommend_list("Live")
recommend_list('Strong Girl Bong-soon')
recommend_list('Tunnel')