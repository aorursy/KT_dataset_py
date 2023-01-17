import re

import nltk

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

from ast import literal_eval

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

from matplotlib.font_manager import FontProperties

from nltk.corpus import wordnet

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.cluster import KMeans

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import plotly

import plotly.io as pio

from os import path

from PIL import Image

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



%matplotlib inline

warnings.simplefilter('ignore')

pd.set_option('display.max_columns', 50)
#Loading the datasets

#metadata of the movies

md = pd.read_csv('/kaggle/input/the-movies-dataset/movies_metadata.csv')

#movie credits

credits = pd.read_csv('/kaggle/input/the-movies-dataset/credits.csv') 

#movie keywords

keywords = pd.read_csv('/kaggle/input/the-movies-dataset/keywords.csv') 
credits.head()
#Converting the string into list of dictionaries

credits.cast = credits.cast.apply(literal_eval)

credits.crew = credits.crew.apply(literal_eval)
# Extracting the Casts into a list from Dictionaries

credits['cast'] = credits['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# Extracting the Director from the Crew

def extract_director(x):

    for crew_mem in x:

        if crew_mem['job'] == 'Director':

            return crew_mem['name']

        else:

            return np.nan



credits['director'] = credits['crew'].apply(extract_director)

credits['director'].fillna('',inplace = True)
credits.drop(['crew'],axis = 1,inplace = True)

credits.head()
keywords.head()
#Converting the string into list of dictionaries

keywords.keywords = keywords.keywords.apply(literal_eval)
# Extracting the Keywords into a list from Dictionaries

keywords['keywords'] = keywords['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
keywords.head()
md.head()
md.describe(include = 'all')
md[(md.adult != "True") & (md.adult != "False")]
idx = [19729,29502,35586]

lst_1 = ['popularity', 'poster_path', 'production_companies','production_countries', 'release_date', 'revenue',

         'runtime', 'spoken_languages', 'status', 'tagline', 'title', 'video', 'vote_average', 'vote_count']

lst_2 = ['belongs_to_collection', 'budget', 'genres', 'homepage', 'id', 'imdb_id', 'original_language', 'original_title', 

         'overview','popularity', 'poster_path', 'production_companies','production_countries', 'release_date']

for i in idx:

    for col_seq in range(len(lst_1)):

            md[lst_1[col_seq]][i] = md[lst_2[col_seq]][i+1]
idx = [x+1 for x in idx]

md.drop(index = idx,inplace = True)
md.adult = md.adult.apply(lambda x : True if (x == 'True') else False)
# Extracting the Genres into a list from Dictionaris

md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# Dropping Duplicates 

credits.drop_duplicates('id',inplace = True)

keywords.drop_duplicates('id',inplace = True)

md.drop_duplicates('id',inplace = True)
#Converting IDs into same data type

md.id = md.id.astype(int)
#Merging DataFrames into one

md = md.merge(credits, on = 'id', how = 'left')

md = md.merge(keywords, on = 'id', how = 'left')

md.head()
# Selecting required columns from the master dataframe

movies = md[['id','original_title','title','cast', 'director', 'keywords', 'genres', 'release_date', 'overview', 

             'original_language', 'adult', 'runtime', 'tagline', 'vote_average', 'vote_count','popularity']]

movies.head(30)
# Missing Value

movies.isna().sum()
movies.original_language.fillna('',inplace = True)

# Fill NA of Tagline with empty strings

movies.tagline.fillna('',inplace = True)

# Fill NA of overview with empty strings

movies.overview.fillna('',inplace = True)

movies.loc[movies.overview == 'No overview found.','overview'] = ''

# Fill NA of runtime with 0

movies.runtime.fillna(0,inplace = True)



movies.cast = movies.cast.apply(lambda x: x if isinstance(x, list) else [])

movies.director.fillna('',inplace = True)

movies.keywords = movies.keywords.apply(lambda x: x if isinstance(x, list) else [])



# If the release_Date is missing, as of now we're putting the date of 2050-01-01 in order to be able to convert in into datetime object

movies.loc[movies['release_date'].isna(),'release_date'] = '2050-01-01'

movies.release_date = pd.to_datetime(movies.release_date,format = '%Y-%m-%d')
movies.head(5)
movies["popularity"] = pd.to_numeric(movies["popularity"], downcast="float")

movies = movies.sort_values(by='popularity',axis=0, ascending=False)[0:20000].reset_index()

movies=movies.drop(['index'], axis=1)

movies.head(5)
#combining the overview and taglines

movies['plot_corpus'] = movies['overview'] + movies['tagline']



def listtostr(txt):

    '''

    Returns string by joining the elements of the list

    '''

    

    txt_clean = ' '.join([str(elem) for elem in txt])

    return txt_clean



movies['keywords'] = movies['keywords'].apply(listtostr)

movies['genres'] = movies['genres'].apply(listtostr)



#movies['plot_corpus_1'] = movies['overview'] + movies['tagline'] + movies['keywords']

movies['genre_corpus'] = movies['keywords'] + movies['genres']
def get_wordnet_pos(word):

    '''

    Returns the tag for the word

    '''

    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": wordnet.ADJ,

                "N": wordnet.NOUN,

                "V": wordnet.VERB,

                "R": wordnet.ADV}



    return tag_dict.get(tag, wordnet.NOUN)



lemmatizer=WordNetLemmatizer()



def clean_plot(txt):

    '''

    Returns the cleaned plot text 

    '''

    

    regex = re.compile(r"[!@%&;?'',.""-]")

    txt_clean = re.sub(regex,'',txt)

    txt_clean = txt_clean.lower()

    txt_clean = txt_clean.split(' ')

    txt_clean = [word for word in txt_clean if word not in stopwords.words('english')]

    txt_clean = ' '.join(txt_clean)

    word_list = nltk.word_tokenize(txt_clean)

    txt_clean = ' '.join([lemmatizer.lemmatize(w,get_wordnet_pos(w)) for w in word_list])

    return txt_clean



def clean_cast(txt):

    '''

    Returns the cleaned cast string

    '''

    

    for i in range(len(txt)):

        txt[i] = re.sub(r"[.,']","",txt[i])

        txt[i] = re.sub(r"[-]"," ",txt[i])

        txt[i] = re.sub(" ","_",txt[i])

        txt[i] = txt[i].lower()

    return txt



def clean_director(txt):

    '''

    Returns the cleaned director string

    '''

    

    txt_clean = re.sub(r"[.,']","",txt)

    txt_clean = re.sub(r"[-]"," ",txt_clean)

    txt_clean = re.sub(" ","_",txt_clean)

    txt_clean = txt_clean.lower()

    return txt_clean
movies['plot_corpus'] = movies['plot_corpus'].apply(clean_plot)

movies['genre_corpus'] = movies['genre_corpus'].apply(clean_plot)

movies['genre_pure'] = movies['genres'].apply(clean_plot)
movies['genre_pure']
movies['cast'] = movies['cast'].apply(clean_cast)

movies['cast'] = movies['cast'].apply(listtostr)

movies['director'] = movies['director'].apply(clean_director)
movies['genre_corpus'] = movies['genre_corpus'] + movies['cast']

movies['mixed_corpus'] = movies['genre_corpus'] + movies['plot_corpus']
tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1,2), min_df = 0, stop_words = 'english')

cv = CountVectorizer(analyzer = 'word', ngram_range = (1,2), min_df = 0, stop_words = 'english')



plot_vector = tf.fit_transform(movies['plot_corpus'])

genre_vector = cv.fit_transform(movies['genre_corpus'])

cast_vector = cv.fit_transform(movies['cast'])

director_vector = cv.fit_transform(movies['director'])

genre_only_vector = cv.fit_transform(movies['genre_pure'])

from sklearn.metrics.pairwise import cosine_similarity



plot_score = cosine_similarity(plot_vector,plot_vector)

genre_score = cosine_similarity(genre_vector,genre_vector)

cast_score = cosine_similarity(cast_vector,cast_vector)

director_score = cosine_similarity(director_vector, director_vector)

genre_only_score = cosine_similarity(genre_only_vector,genre_only_vector)



plot_score = pd.DataFrame(plot_score)

genre_score = pd.DataFrame(genre_score)

cast_score = pd.DataFrame(cast_score)

director_score = pd.DataFrame(director_score)

genre_only_score = pd.DataFrame(genre_only_score)
vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')

vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')

C = vote_averages.mean()

m = vote_counts.quantile(0.95)



print(C,m)



def weighted_rating(x):

    v = x['vote_count']

    R = x['vote_average']

    return (v/(v+m) * R) + (m/(m+v) * C)



movies['wr'] = movies.apply(weighted_rating, axis=1)
movies['genres'] = movies['genres'].apply(lambda x : x.split())

movies['release_year'] = movies.release_date.apply(lambda x: x.year)
def score(value,index_list,feature):

    '''

    Returns list of scores for the passed feature

    '''

    if feature == 'genre':

        df_temp = pd.DataFrame(genre_only_score[value])

    if feature == 'plot':

        df_temp = pd.DataFrame(plot_score[value])

    if feature == 'plot_1':

        df_temp = pd.DataFrame(plot_score_1[value])

    if feature == 'cast':

        df_temp = pd.DataFrame(cast_score[value])

    if feature == 'director':

        df_temp = pd.DataFrame(director_score[value])

    df_temp = df_temp.loc[df_temp.index.isin(index_list)]

    my_list = df_temp[value].tolist()

    return my_list



    
def get_feature_set(df1,df2,df3,title):

    

    '''

    idx : index value of the target movie

    top : index value of top 500 movies(sorted(descending) by genre similarity score w.r.t. target movie)

    feature_set : Data frame containing plot score matrix of movies which had their index in "top"

    movie_set : Name of the movies which had their index in "top"

    '''   

    

    idx = movies.index[movies.title == title].values.astype(int)[0]

    top = df1[idx].sort_values(ascending = False)[0:500].index.values.tolist()

    top = df1[idx].sort_values(ascending = False)[0:500].index.values.tolist()

    feature_set = df2[df2.index.isin(top)]

    movies_set = pd.DataFrame(movies.loc[movies.index.isin(top),'title'])

    return feature_set,movies_set



def get_recommendations(title,cluster_num,df1=genre_score,df2=plot_score,df3=cast_score):

    

    '''

    movie_set = dataframe to store the cluster labels(1,2,3) assigned to movies along with their similarity scores and ratings

    df_recommend = dataframe with information about the top 50 movies recommended from each cluster

    '''

    

    feature_set,movies_set = get_feature_set(df1,df2,df3,title)

    cluster_algo = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)

    cluster = cluster_algo.fit(feature_set)

    movies_set['cluster'] = cluster.labels_

    index = movies_set.index.values.tolist()

    idx = movies.index[movies.title == title].values.astype(int)[0]

    movies_set.loc[movies_set.index.isin(index),'wr'] = movies.loc[movies.index.isin(index),'wr']

    movies_set.loc[movies_set.index.isin(index),'p_score'] = score(idx,index,'plot')

    movies_set.loc[movies_set.index.isin(index),'g_score'] =score(idx,index,'genre')

    movies_set.loc[movies_set.index.isin(index),'c_score'] = score(idx,index,'cast')

    movies_set.loc[movies_set.index.isin(index),'d_score'] = score(idx,index,'director')

    target_cluster = movies_set.loc[movies_set.title == title, 'cluster'].values[0]

    if(target_cluster!=0):

        movies_set.loc[movies_set.cluster==target_cluster,'cluster'] = 100

        movies_set.loc[movies_set.cluster==0,'cluster'] = target_cluster

        movies_set.loc[movies_set.cluster==100,'cluster'] = 0

    if(cluster_num==1):

        recommend_1 = movies_set[movies_set.cluster == 0] 

        df_recommend = pd.DataFrame(recommend_1.sort_values(['g_score','p_score','c_score','d_score', 'wr'],ascending=[False,False,False,False,False])[1:50].title)

    if(cluster_num==2):

        recommend_1 = movies_set[movies_set.cluster == 1] 

        df_recommend = pd.DataFrame(recommend_1.sort_values(['g_score','p_score','c_score','d_score', 'wr'],ascending=[False,False,False,False,False])[1:50].title)

    if(cluster_num==3):

        recommend_1 = movies_set[movies_set.cluster == 2] 

        df_recommend = pd.DataFrame(recommend_1.sort_values(['g_score','p_score','c_score','d_score', 'wr'],ascending=[False,False,False,False,False])[1:50].title)

    df_recommend.loc[df_recommend.index.isin(index),'genres'] = movies.loc[movies.index.isin(index),'genre_pure']

    df_recommend.loc[df_recommend.index.isin(index),'title'] = movies.loc[movies.index.isin(index),'title']

    df_recommend.loc[df_recommend.index.isin(index),'director'] = movies.loc[movies.index.isin(index),'director']

    df_recommend.loc[df_recommend.index.isin(index),'cast'] = movies.loc[movies.index.isin(index),'cast']

    df_recommend.loc[df_recommend.index.isin(index),'ratings'] = movies.loc[movies.index.isin(index),'wr']

    df_recommend.loc[df_recommend.index.isin(index),'adult'] = movies.loc[movies.index.isin(index),'adult']

    df_recommend['ratings'] = df_recommend['ratings'].round(decimals=2)

    return df_recommend
def cluster_class(title,cl_num):

    '''

    converts each column of the recommendation dataframe into list

    '''

    df = get_recommendations(title,cluster_num=cl_num)

    cast = ' '.join(df.cast.tolist())

    genre = ' '.join(df.genres.tolist())

    director = ' '.join(df.director.tolist())

    ratings = df.ratings.tolist()

    return df,cast,director,genre,ratings,cl_num
class recommended_cluster:

    '''

    movies   : A dataframe of movies with other information within a cluster

    cast     : A list of cast for the movies within a cluster

    director : A list of directors for the movies within a cluster

    genre    : A list of genres of the movies within a cluster

    ratings  : A list of ratings of the movies within a cluster

    

    '''

   

    def __init__(self,cluster_tuple):

        self.movies = cluster_tuple[0]

        self.cast = cluster_tuple[1]

        self.director = cluster_tuple[2]

        self.genre = cluster_tuple[3]

        self.ratings = cluster_tuple[4]

        self.cl_num = cluster_tuple[5]

        

    def recommended_movie(self):

        '''

        generates a table containing top 10 recommendations from each cluster along with their corresponding ratings

        '''

        df = self.movies[0:10]

        fig = go.Figure(data=[go.Table(header = dict(values = ['Title','Rating'],

                                                     font = dict(size=15),

                                                     align = "center"),

                                       cells = dict(values = [df.title,df.ratings],

                                                    align = "center")

                                      )

                             ]

                       )

        fig.show()    

    

    

    def cast_cloud(self):

        '''

        generates a wordcloud of casts present in a cluster

        '''

        wordcloud = WordCloud(random_state=1,

                              background_color='black', colormap='Blues_r',

                              collocations=False, stopwords = STOPWORDS).generate(self.cast)

        return wordcloud



    def director_cloud(self):

        '''

        generates a wordcloud of directors present in a cluster

        '''

        wordcloud = WordCloud(random_state=1,

                              background_color='black', colormap='Blues_r',

                              collocations=False, stopwords = STOPWORDS).generate(self.director)

        return wordcloud



        

    def genre_cloud(self):

        '''

        generates a wordcloud of genres present in a cluster

        '''

        wordcloud = WordCloud(random_state=1,

                              background_color='black', colormap='Blues_r',

                              collocations=False, stopwords = STOPWORDS).generate(self.genre)

        return wordcloud



    

    def get_wordcloud(self):

        '''

        plots the cast, genre and director wordclouds in the from of subplots 

        '''

        fig = plt.figure(figsize=(30,10))

        ax1 = fig.add_subplot(131)

        ax2 = fig.add_subplot(132)

        ax3 = fig.add_subplot(133)



        font = FontProperties()

        font.set_family('serif')

        font.set_name('Times New Roman')

        font.set_size(40)



        def nulltick(ax):

            '''

            removes the ticks from x and y axis of the wordcloud plots

            '''

            ax.xaxis.set_major_locator(ticker.NullLocator())

            ax.xaxis.set_minor_locator(ticker.NullLocator())

            ax.yaxis.set_major_locator(ticker.NullLocator())

            ax.yaxis.set_minor_locator(ticker.NullLocator())

        

        ax1.imshow(self.genre_cloud(), interpolation='bilinear')

        nulltick(ax1)

        ax1.set_xlabel("GENRE WORDCLOUD",fontproperties = font)

        ax2.imshow(self.cast_cloud(), interpolation='bilinear')

        nulltick(ax2)

        ax2.set_xlabel("CAST WORDCLOUD",fontproperties = font)

        ax3.imshow(self.director_cloud(), interpolation='bilinear')

        nulltick(ax3)

        ax3.set_xlabel("DIRECTOR WORDCLOUD",fontproperties = font)

        

        fig.suptitle('CLUSTER '+str(self.cl_num), fontsize=60)

        fig.tight_layout()

        plt.show()

    

    def pie_data(self):

        '''

        returns a dataframe with frequency of occurence of genres in a cluster

        '''

        wordlist = self.genre.split()

        wordfreq = [wordlist.count(p) for p in wordlist]

        dictionary = dict(list(zip(wordlist,wordfreq)))

        lst = list(dictionary.items())

        df = pd.DataFrame(lst)

        df.columns =['Genre','Frequency']

        return df



    def get_piechart(self):

        '''

        plots the frequency of occurence of genres in a cluster in the form of a piechart

        '''

        df = self.pie_data()

        values = df['Frequency']

        labels = df['Genre']

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

        fig.show()

        

        

    def get_ratings(self):

        '''

        calculates the minimum, maximum and average rating of a cluster

        '''

        minima = min(self.ratings)

        maxima = max(self.ratings)

        avg = sum(self.ratings)/len(self.ratings)

        return round(minima,2),round(maxima,2),round(avg,2)

    

    def ratings_chart(self):

        '''

        plots the minimum, maximum and average rating of a cluster in the form of a bar chart 

        '''

        minima,maxima,avg = self.get_ratings()

        fig = go.Figure(data=[

            go.Bar(name='Minimum Rating', x=['CLUSTER '+str(self.cl_num)], y=[minima], text=minima,textposition='outside',width = [0.2,0.2,0.2],marker_color = 'indianred'),

            go.Bar(name='Average Rating', x=['CLUSTER '+str(self.cl_num)], y=[avg],text=avg,textposition='outside', width = [0.2,0.2,0.2],marker_color = 'blue'),

            go.Bar(name='Maximum Rating', x=['CLUSTER '+str(self.cl_num)], y=[maxima],text=maxima,textposition='outside', width = [0.2,0.2,0.2], marker_color = 'green')

        ])

        fig.update_yaxes(range=[1, 10],dtick=1)

        fig.show()

   
def Dashboard(title):

    for i in [1,2,3]:

        cluster = recommended_cluster(cluster_class(title,i))

        cluster.get_wordcloud()

        cluster.recommended_movie()

        cluster.get_piechart()

        cluster.ratings_chart()
Dashboard("Spider-Man")