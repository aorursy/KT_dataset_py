import sklearn

import matplotlib.pyplot as plt

%matplotlib inline

import pandas

from sklearn.cross_validation import train_test_split

import numpy

from wordcloud import WordCloud,STOPWORDS
df = pandas.read_csv('../input/techcrunch_posts.csv')

print(df.shape)
df.head()
df['authors']=df['authors'].apply(lambda x: str(x).split(','))

df['tags']=df['tags'].apply(lambda x:['No tag'] if str(x)=='NaN' else str(x).split(','))

df['topics']=df['topics'].apply(lambda x: str(x).split(','))
df['content']=df['content'].fillna(0)

df=df[df['content']!=0]

df=df.reset_index(drop=True)
import re

import nltk

from nltk.corpus import stopwords
def to_words(content):

    letters_only = re.sub("[^a-zA-Z]", " ", content) 

    words = letters_only.lower().split()                             

    stops = set(stopwords.words("english"))                  

    meaningful_words = [w for w in words if not w in stops] 

    return( " ".join( meaningful_words )) 
clean_content=[]

for each in df['content']:

    clean_content.append(to_words(each))
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer()

features=tfidf.fit_transform(clean_content)
from sklearn.neighbors import NearestNeighbors

knn=NearestNeighbors(n_neighbors=30,algorithm='brute',metric='cosine')

knn_fit=knn.fit(features)
def recommend_to_user(author):

    ## Find All Authors##

    indexes=[]

    for i in range(len(df)):

        if author in df['authors'][i]:

            indexes.append(i)

    tmp_df=df.iloc[indexes,:]

    author_content=[]

    for each in tmp_df['content']:

        author_content.append(to_words(each))

    wordcloud = WordCloud(background_color='black',

                      width=3000,

                      height=2500

                     ).generate(author_content[0])

    ## Find Nearest Neighbors based on the latest aritcles the author published on the website

    Neighbors = knn_fit.kneighbors(features[indexes[0]])[1].tolist()[0][2:]

    ## Get rid of all articles that is authored/co-authored by the author and find the articles

    All_article = df.iloc[Neighbors,:]

    All_article = All_article.reset_index(drop=True)

    kept_index = []

    for j in range(len(All_article)):

        if author in All_article['authors'][j]:

            pass

        else:

            kept_index.append(j)

    Final_frame = All_article.iloc[kept_index,:]

    Final_frame=Final_frame.reset_index(drop=True)

    Selected_articles = Final_frame.iloc[0:5,:]

    

    ## Print out result directly ##

    print('==='*30)

    print('The article(s) of '+author+' is always featured by the following words')

    plt.figure(1,figsize=(8,8))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()

    print("==="*30)

    print('The top five articles recommended to '+author+' are:')

    for k in range(len(Selected_articles)):

        print(Selected_articles['title'][k]+', authored by '+Selected_articles['authors'][k][0]+' ,article can be visted at \n '+Selected_articles['url'][k])
recommend_to_user('Bastiaan Janmaat')
recommend_to_user('Matthew Lynley')