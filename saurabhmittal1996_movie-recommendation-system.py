import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
movie = pd.read_csv('../input/movielens-20m-dataset/movie.csv',index_col=False)
rating = pd.read_csv('../input/movielens-20m-dataset/rating.csv')
movie.head()
rating.head()
movie.shape, rating.shape
rating.describe(include='all').T
movie.describe(include='all').T
data = pd.merge(movie, rating, on='movieId')
data.head()
data.shape
data.nunique()
data.head()
avg_rating = data.groupby('title')['rating'].mean()
avg_rating
avg_rating.sort_values(ascending=False)
total_rating = data.groupby('title')['rating'].count()
total_rating
df = pd.DataFrame()
df['Average_rating'] = avg_rating
df['Total_Rating'] = total_rating
df.tail()
df.shape
avg_rating.hist(bins=25, grid=False, edgecolor='b', label ='All genres', figsize=(20,8))
plt.legend(loc=(1.05,0), ncol=2)
plt.xlim(0,5)
plt.xlabel('Movie rating')
plt.title('Movie rating histograms')
plt.show()
user_rating = rating[['userId','rating']].groupby('userId').mean()

# Plot histogram
user_rating.plot(kind='hist', bins=50, grid=0, edgecolor='black', figsize=(20,8))

plt.xlim(0,5)
plt.legend()
plt.xlabel ('Average movie rating')
plt.ylabel ('Normalized frequency')
plt.title ('Average ratings per user')
plt.show()
# Histogram of ratings counts.

user_rating = rating[['userId', 'movieId']].groupby('userId').count()
user_rating.columns=['num_ratings']

plt.figure(figsize=(25,8))
plt.hist(user_rating.num_ratings, bins=100, edgecolor='black', log=True)
plt.title('Ratings per user')
plt.xlabel('Number of ratings given')
plt.ylabel('Number of userIds')
plt.xlim(0,)
plt.xticks(np.arange(0,10000,500))
plt.show()
plt.figure(figsize=(25,15))
sns.jointplot(df.Average_rating, df.Total_Rating,)
plt.xlabel('Average Rating')
plt.ylabel('Total Rating')
plt.xlim(0,)
plt.show()
data.shape
new_data = data.iloc[0:15000000]
new_data.shape
#sorting values according to num of rating column

movie_title = new_data.pivot_table(index='userId', columns = 'title', values='rating')
movie_title.head()
movie_title.shape
def movie_recommendation():
    #Taking Movie as a Input on whose basic User want recommendation
    movie = input("Please Enter Movie Name : ")
    
    #How many Recommendation should be shown to user
    n = int(input("How many recommendation you need : "))

    #getting its rating from our data
    user_rating = movie_title[movie]
  
    #Finding Movies with similar rating
    similar_movie = movie_title.corrwith(user_rating)
    
    similar_movie = pd.DataFrame(similar_movie, columns =['Correlation']) 
    
    similar_movie = similar_movie.join(df['Total_Rating'])

    #Sorting related 
    rec_ = similar_movie.sort_values(by = ['Correlation', 'Total_Rating'], ascending=False)

    rec = pd.DataFrame(rec_)
    rec.drop_duplicates()
    print()
    print()
    print("Following are the {} Recommended Movies".format(n))

    return rec.head(n)
movie_recommendation()
genre_labels = set()
for s in movie['genres'].str.split('|').values:
    genre_labels = genre_labels.union(set(s))
    
genre_labels = list(genre_labels)
genre_labels
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
def count_word(df, ref_col, liste):
    keyword_count = dict()
    
    for s in liste: 
        keyword_count[s] = 0
    
    for liste_keywords in df[ref_col].str.split('|'):
        if type(liste_keywords) == float and pd.isnull(liste_keywords): 
            continue
        
        for s in liste_keywords: 
            if pd.notnull(s): 
                keyword_count[s] += 1
    
    # convert the dictionary in a list to sort the keywords  by frequency
    keyword_occurences = []
    
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    
    return keyword_occurences, keyword_count
keyword_occurences, dum = count_word(movie, 'genres', genre_labels)
keyword_occurences
keyword_occurences[0:50]
words = dict()
for s in keyword_occurences:
    words[s[0]] = s[1]
# instantiate a word cloud object
word_cloud = WordCloud(
    background_color='white',
    max_words=2000,
    stopwords=stopwords
)

# generate the word cloud
word_cloud.generate_from_frequencies(words)
plt.figure(figsize=(20,15))
plt.imshow(word_cloud, interpolation="bilinear")
plt.axis('off')
plt.show()
fig = plt.figure(1, figsize=(18,13))
ax2 = fig.add_subplot(2,1,2)
y_axis = [i[1] for i in keyword_occurences]
x_axis = [k for k,i in enumerate(keyword_occurences)]
x_label = [i[0] for i in keyword_occurences]
plt.xticks(rotation=85, fontsize = 15)
plt.yticks(fontsize = 15)
plt.xticks(x_axis, x_label)
plt.ylabel("No. of occurences", fontsize = 24, labelpad = 0)
ax2.bar(x_axis, y_axis, align = 'center', color='r')
plt.title("Popularity of Genres",color='g')
plt.show()
for i in genre_labels:
    movie[i] = movie.apply(lambda _:int(i in _.genres), axis = 1)
movie.head()
movie.info()
movie['movieId'] = movie['movieId'].astype(object)
avg_movieid_rating = pd.DataFrame(rating.groupby('movieId')['rating'].agg(['mean','count']))
avg_movieid_rating.head()
movies = pd.merge(avg_movieid_rating,movie,on='movieId')
movies.head()
tags = pd.read_csv("../input/movielens-20m-dataset/tag.csv")
tags.head()
tags.shape
tags.nunique()
tags.drop(['timestamp'], axis=1, inplace=True)
tag_labels = [i for i in tags.tag.unique()]
tag_labels
tag_keyword_occurences, tag_dum = count_word(tags, 'tag', tag_labels)
tag_keyword_occurences
tag_words = dict()
tag_trunc_occurences = tag_keyword_occurences[0:50]
for s in tag_trunc_occurences:
    tag_words[s[0]] = s[1]
# instantiate a word cloud object
word_cloud = WordCloud(
    background_color='white',
    max_words=2000,
    stopwords=stopwords
)

# generate the word cloud
word_cloud.generate_from_frequencies(tag_words)
plt.figure(figsize=(20,15))
plt.imshow(word_cloud, interpolation="bilinear")
plt.axis('off')
plt.show()
fig = plt.figure(1, figsize=(25,22))
ax2 = fig.add_subplot(2,1,2)
tag_trunc_occurences = tag_keyword_occurences[0:20]

y_axis = [i[1] for i in tag_trunc_occurences]
x_axis = [k for k,i in enumerate(tag_trunc_occurences)]
x_label = [i[0] for i in tag_trunc_occurences]
plt.xticks(rotation=90, fontsize = 15)
plt.yticks(fontsize = 15)
plt.xticks(x_axis, x_label)
plt.ylabel("No. of occurences", fontsize = 24, labelpad = 0)
ax2.bar(x_axis, y_axis, align = 'center', color='m')
plt.title("Popularity of Tags",color='g')
plt.show()
movies = pd.merge(movies,tags,on='movieId')
movies.head()
movies.shape
def movie_by_genre():
    genre = input("Please enter genre : ")
    n = int(input("Please enter Number of Recommendation you want : "))
    
    df = pd.DataFrame(movies.loc[(movies[genre]==1)].sort_values(['mean'], ascending=False)[['title', 'genres', 'count','mean']])
    df = df.drop_duplicates()
    
    return df[:n]
movie_by_genre()
def movie_by_tag():
    tag = input("Please enter tag : ")
    n = int(input("Please enter Number of Recommendation you want : "))
    
    df = pd.DataFrame(movies.loc[(movies['tag']==tag)].sort_values(['mean'], ascending=False)[['tag', 'title','count','mean']])
    df = df.drop_duplicates()
    
    return df[:n]
movie_by_tag()
def movie_by_tag_genre():
    genre = input("Please enter Genre : ")
    tag = input("Please enter tag : ")
    n = int(input("Please enter Number of Recommendation you want : "))
    
    df = pd.DataFrame(movies.loc[(movies['tag']==tag) & (movies[genre]==1)].sort_values(['mean'], ascending=False)[['tag', 'title', 'genres', 'count','mean']])
    df = df.drop_duplicates()
    
    return df[:n]
movie_by_tag_genre()
