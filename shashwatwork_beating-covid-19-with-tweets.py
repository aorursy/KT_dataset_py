#For uploading and accessing the data

import pandas as pd

import numpy as np



#For visualizations

import matplotlib.pyplot as plt

import seaborn as sns

!pip install dexplot -q

!pip install pycaret -q

!pip install stylecloud -q

# for visualizations

plt.style.use('ggplot')



# for interactive visualizations

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected = True)

import plotly.figure_factory as ff

from sklearn.preprocessing import StandardScaler



from pandas_profiling import ProfileReport



import dexplot as dxp





from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer



# Nltk for tokenize and stopwords

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

import requests

from PIL import Image

from io import BytesIO

from wordcloud import WordCloud, STOPWORDS

import stylecloud

from wordcloud import ImageColorGenerator

RANDOM_SEED = 42

df = pd.read_csv('/kaggle/input/covid19-tweets/covid19_tweets.csv')

df.head()
def missing_value_of_data(data):

    total=data.isnull().sum().sort_values(ascending=False)

    percentage=round(total/data.shape[0]*100,2)

    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])



def duplicated_values_data(data):

    dup=[]

    columns=data.columns

    for i in data.columns:

        dup.append(sum(data[i].duplicated()))

    return pd.concat([pd.Series(columns),pd.Series(dup)],axis=1,keys=['Columns','Duplicate count'])



def unique_values_in_column(data,feature):

    unique_val=pd.Series(data.loc[:,feature].unique())

    return pd.concat([unique_val],axis=1,keys=['Unique Values'])



def count_values_in_column(data,feature):

    total=data.loc[:,feature].value_counts(dropna=False)

    percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)

    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])



def ngrams_top(corpus,ngram_range,n=None):

    """

    List the top n words in a vocabulary according to occurrence in a text corpus.

    """

    vec = CountVectorizer(stop_words = 'english',ngram_range=ngram_range).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    total_list=words_freq[:n]

    df=pd.DataFrame(total_list,columns=['text','count'])

    return df



missing_value_of_data(df)
duplicated_values_data(df)
count_values_in_column(df,'user_location')
count_values_in_column(df,'hashtags')
dd_11 = ngrams_top(df['text'],(1,1),n=10)

dxp.bar(x='text', y='count', data=dd_11,figsize=(10,5),cmap='dark12_r',title='Count Plot for Most Frequent words in Tweets')
dd_22 = ngrams_top(df['text'],(2,2),n=10)

dxp.bar(x='text', y='count', data=dd_22,figsize=(10,5),cmap='dark12_r',title='Count Plot for Most Frequent words in Tweets')
dd_33 = ngrams_top(df['text'],(3,3),n=10)

dxp.bar(x='text', y='count', data=dd_33,figsize=(10,5),cmap='dark12_r',title='Count Plot for Most Frequent words in Tweets')
Top10_source = pd.DataFrame(df['source'].value_counts().sort_values(ascending=False)[:10]).reset_index()

Top10_source.columns = ['Source','Count']
dxp.bar(x='Source', y='Count', data=Top10_source,figsize=(10,5),cmap='viridis',title='Source for Tweets in COVID-19')
best_10_regions = pd.DataFrame(df['user_location'].value_counts().sort_values(ascending=False)[:15]).reset_index()

best_10_regions.columns = ['user_location','Count']
dxp.bar(x='user_location', y='Count', data=best_10_regions,figsize=(15,5),cmap='viridis',title='Geographical Location for Tweets')
Top10_user = pd.DataFrame(df['user_name'].value_counts().sort_values(ascending=False)[:10]).reset_index()

Top10_user.columns = ['user_name','count']
dxp.bar(x='user_name', y='count', data=Top10_user,figsize=(15,5),cmap='viridis',title='Users Dominating in Tweets in COVID-19')
dxp.bar(x='user_verified', y='user_followers', data=df.head(100),figsize=(10,10),split='source',aggfunc='mean',title='Relationship betwwen fake and real users')
hashtags = df['hashtags'].dropna().tolist()

unique_hashtags=(" ").join(hashtags)

stylecloud.gen_stylecloud(text = unique_hashtags,

                          icon_name='fas fa-first-aid',

                          palette='colorbrewer.diverging.Spectral_11',

                          background_color='black',

                          gradient='horizontal')

from IPython.display import Image 



Image("./stylecloud.png",width = 600, height = 600)
hashtags = df['text'].dropna().tolist()

unique_hashtags=(" ").join(hashtags)

stylecloud.gen_stylecloud(text = unique_hashtags,

                          icon_name='far fa-comment',

                          background_color='white',

                          gradient='horizontal')

from IPython.display import Image 



Image("./stylecloud.png",width = 600, height = 600)
df_plot = df[['user_created','user_followers','user_favourites','user_friends']]

df_plot['user_created'] = pd.to_datetime(df_plot.user_created)

df_plot['user_created'] = df_plot['user_created'].dt.strftime('%m/%d/%Y')

df_plot = df_plot.sort_values('user_created')
dxp.line(x='user_created',y = 'user_followers',aggfunc='mean',data=df_plot.head(100),figsize=(10,5),cmap='viridis',title='Followers Timeline')
dxp.line(x='user_created',y = 'user_favourites',aggfunc='mean',data=df_plot.head(100),figsize=(10,5),cmap='viridis_r',title='User Favourite Timeline')
dxp.line(x='user_created',y = 'user_friends',aggfunc='mean',data=df_plot.head(100),figsize=(10,5),cmap='plotly3',title='User Friends Timeline')
from pycaret.nlp import *
nlp1 = setup(df, target = 'text', session_id=RANDOM_SEED, experiment_name='covid')
models()
#Latent Dirichlet Allocation

lda = create_model('lda',multi_core=True)
#Non-Negative Matrix Factorization

nmf = create_model('nmf', num_topics = 6)
#Latent Semantic Indexing

lsi = create_model('lsi', num_topics = 6)
#Hierarchical Dirichlet Process

hdp = create_model('hdp', num_topics = 6)
#Random Projections

rp = create_model('rp',num_topics = 6)
lda_results = assign_model(lda)

lda_results.head()
nmf_results = assign_model(nmf)

nmf_results.head()
hdp_results = assign_model(hdp)

hdp_results.head()
plot_model(lda)
plot_model(lda, plot = 'distribution')
plot_model(lda, plot = 'bigram')
plot_model(lda, plot = 'trigram')
plot_model(lda, plot = 'tsne')
plot_model(lda, plot = 'topic_distribution')
plot_model(lda, plot = 'wordcloud')
plot_model(lda, plot = 'umap')
evaluate_model(lda)