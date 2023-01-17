import pandas as pd
import numpy as np
import csv
import collections
from collections import Counter
netflix=pd.read_csv('../input/netflix-shows/netflix_titles.csv')
netflix.head()
len(netflix[netflix['type']=='Movie'])
len(netflix[netflix['type']=='TV Show'])
list(netflix[netflix['date_added'].isna()]['title'])
len(list(netflix[netflix['date_added'].isna()]['title']))
netflix=netflix[netflix['date_added'].notna()]
netflix.head()
netflix['director']=netflix['director'].fillna("")
netflix['rating']=netflix['rating'].fillna("")
netflix['date_added']=pd.to_datetime(netflix['date_added'])
netflix['year_added']=netflix['date_added'].dt.year
netflix.head()
from collections import Counter
count_countries=Counter(",".join(netflix['country'].dropna()).split(","))
print(count_countries)
top_10_countries=count_countries.most_common(10)
top_10_countries
country_name=[]
country_count=[]
for i in range(10):
    country_name.append(top_10_countries[i][0])
    
print(country_name)
for i in range(10):
    country_count.append(top_10_countries[i][1])
    
print(country_count)
import matplotlib.pyplot as plt
y_pos=np.arange(len(country_name))
plt.barh(y_pos, country_count, align='center', alpha=0.5)
plt.yticks(y_pos, country_name)
plt.xlabel('Count')
plt.title('Top 10 Countries')
plt.show()
count_genre=Counter(",".join(netflix['listed_in'].dropna()).split(","))
print(count_genre)
top_10_genre=count_genre.most_common(10)
print(top_10_genre)
genre_name=[]
genre_count=[]
for i in range(10):
    genre_name.append(top_10_genre[i][0])
for i in range(10):
    genre_count.append(top_10_genre[i][1])
    
    
print(genre_count)
print(genre_name)
y_pos=np.arange(len(genre_name))
plt.barh(y_pos, genre_count, align='center', alpha=0.5)
plt.yticks(y_pos, genre_name)
plt.xlabel('Count')
plt.title('Top 10 Genre')
plt.show()
def by_country(df,country):
    drop_country_na=df[df['country'].notna()]
    return drop_country_na[drop_country_na['country'].str.contains(country)]


def top_genre_by_country(df,country):
    genre_counter = Counter(", ".join(by_country(netflix, country)['listed_in']).split(", "))
    genre_counter1=genre_counter.most_common(10)
    genre_name=[]
    genre_count=[]
    for i in range(10):
        genre_name.append(genre_counter1[i][0])
        genre_count.append(genre_counter1[i][1])
    return genre_name, genre_count
countries=['United States', 'France', 'Japan', 'South Korea']
for i in np.arange(len(countries)):
    genre=[]
    count=[]
    genre,count=top_genre_by_country(netflix,countries[i])
    
    y_pos=np.arange(len(genre))
    plt.barh(y_pos, count, align='center', alpha=0.5)
    plt.yticks(y_pos, genre)
    plt.xlabel('Count')
    plt.title(countries[i])
    plt.show()
    
    
shows_by_year=netflix[netflix['type']=='TV Show'].groupby('year_added').size()
shows_by_year.head()
movies_by_year=netflix[netflix['type']=='Movie'].groupby('year_added').size()
movies_by_year.head()
movies_by_year=movies_by_year.to_frame()
movies_by_year=movies_by_year.reset_index('year_added')
movies_by_year.head()
shows_by_year=shows_by_year.to_frame()
shows_by_year=shows_by_year.reset_index('year_added')
shows_by_year.head()
movies_by_year.rename( columns={0:'movies_no'}, inplace=True )
shows_by_year.rename( columns={0:'shows_no'}, inplace=True )


movies_by_year.head()
ax = plt.gca()

movies_by_year.plot(kind='line',x='year_added',y='movies_no',ax=ax)
shows_by_year.plot(kind='line',x='year_added',y='shows_no',ax=ax)

plt.show()
import nltk
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
netflix=netflix[netflix['title'].notna()]
netflix['aggregated_text']=netflix['description'].str.lower()+""+netflix['listed_in'].str.lower()+""+netflix['rating'].str.lower()+netflix['director'].str.lower()
corpus=list(netflix['aggregated_text'].str.split())
stopwords_list=set(stopwords.words("english"))
index=list(range(0,len(corpus)))
clean_corpus=[]

for sentence in corpus:
    s=[]
    for word in sentence:
        clean_word=re.sub(r'[^\w\s]','', word)
        if clean_word not in stopwords_list:
            s.append(clean_word)
        clean_corpus.append(" ".join(s))    
    
tfidf_vectorizer = TfidfVectorizer().fit_transform(clean_corpus)
def get_recommendation(show_list, vectorizer):

    title, scores, genre = [], [], []
    for show_name in show_list:
        show_index = netflix[netflix['title'] == show_name].index[0]
        cosine_similarities = linear_kernel(vectorizer[show_index], vectorizer).flatten()
        similar_show_index = cosine_similarities.argsort()[:-7:-1][1:]
        title += [netflix['title'][i] for i in similar_show_index]
        genre += [netflix['listed_in'][i] for i in similar_show_index]
        scores += list(cosine_similarities[similar_show_index])

    df = pd.DataFrame(data = {'Title': title, 
                                 'Genre': genre,
                                 'Cosine_similarity': scores})

    df = df[~df['Title'].isin(show_list)].sort_values('Cosine_similarity', ascending = False)
    df['Title'] = df['Title'].drop_duplicates()
    top_five_list = df[df['Title'].notna()].iloc[0:5, :]
    
    return top_five_list
watched_shows = ['Narcos', 'The Vampire Diaries', 'Transformers Prime']
get_recommendation(watched_shows, tfidf_vectorizer)
