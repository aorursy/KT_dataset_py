!pip install rake_nltk
import pandas as pd
import numpy as np

from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer #tokenizes a collection of words extracted from a text doc
from ast import literal_eval #This evaluates whether an expresion is a Python datatype or not
data = pd.read_csv('../input/imdb-extensive-dataset/IMDb movies.csv')
print(data.shape)
data.head()
#There are many null values
data.isnull().sum()
#Lets convert all Null values into 'missing value'
data = data.fillna('missing value')
#Recommend movies based on a director (Pls give full names)
#rec_director = input('Enter director you want to be recommended movies of: ')
rec_director = 'Christopher Nolan'
data[data['director'] == rec_director]

#Recommend movies based on a writer (Pls give full names)
#rec_writer = input('Enter writer you want to be recommended movies of: ')
#data[data['writer'] == rec_writer]
#rec_actor = input('Enter actor you want to be recommended movies of: ')
rec_actor = 'Ryan Gosling'
rec_actor = data[data['actors'].str.contains(rec_actor)] 
rec_actor
data.columns
#Extract relevant columns that would influence a movie's rating based on the content.

#Due to memory issue using just 3k data. You can try this code on Google Colabs for better performance
data1 = data[['title','genre','director','actors','description']].head(3000)
data1.head()
data1.isnull().sum()
#Impute all missing values
data1 = data1.fillna('missing value')
#Convert all columns into lower case
data1 = data1.applymap(lambda x: x.lower() if type(x) == str else x)
data1.head()
#Use genre as a list of words
data1['genre'] = data1['genre'].map(lambda x: x.split(','))
data1['genre']
#Similarily lets separate names into first and last name with commas
data1[['director','actors']] = data1[['director','actors']].applymap(lambda x: x.split(',')) #apply map used for more than 1 column, map for 1 column
data1[['director','actors']].head()
#Combine director, actor names into 1 word respectively this will be used for text extraction

for index,row in data1.iterrows():
    row['actors'] = [x.replace(' ','') for x in row['actors']]
    row['director'] = [x.replace(' ','') for x in row['director']]
data1.head()
#Create a empty list Keywords
data1['keywords'] = ''
#Loop across all rows to extract all keywords from description
for index, row in data1.iterrows():
    description = row['description']
    
    #instantiating Rake by default it uses English stopwords from NLTK and discards all punctuation chars
    r = Rake()
    
    #extract words by passing the text
    r.extract_keywords_from_text(description)
    
    #get the dictionary with key words and their scores
    keyword_dict_scores = r.get_word_degrees()
    
    #assign keywords to new columns
    row['keywords'] = list(keyword_dict_scores.keys())
    
#drop description
data1.set_index('title', inplace = True)
data1.head()
data1['bow'] = ''
columns = data1.columns
for index, row in data1.iterrows():
    words = ''
    for col in columns:
        words = words + ' '.join(row[col])+ ' '
        row['bow'] = words
        

#Use below code if you do not want to include director name into bow
    #for col in columns:
        #if col != 'director':
            #words = words + ' '.join(row[col])+ ' '
        #else:
            #words = words + row[col]+ ' '
        #row['bow'] = words

    
#df1.drop(columns = [col for col in df1.columns if col!= 'bag_of_words'], inplace = True)
data1.head()
#instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(data1['bow'])

#create a Series for movie titles so they are associated to an ordered numerical list, we will use this later to match index
indices = pd.Series(data1.index)
indices[:5]
#Shape count_matrix
count_matrix
type(count_matrix)
#Convert sparse count_matrix to dense vector
c = count_matrix.todense()
c
#Print count_matrix for 0th row
print(count_matrix[0,:]) #This shows all words and their frequency in bow of 0th row
#Gives vocabulary of all words in 'bow' and their counts
count.vocabulary_
#generating the cosine similarity matrix

cosine_sim = cosine_similarity(count_matrix, count_matrix)
cosine_sim
#Lets build a function that takes in movie and recommends top n movies

def recommendations(title,n,cosine_sim = cosine_sim):
    recommended_movies = []
    
    #get index of the movie that matches the title
    idx = indices[indices == title].index[0]
    
    #find highest cosine_sim this title shares with other titles extracted earlier and save it in a Series
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    
    #get indexes of the 'n' most similar movies
    top_n_indexes = list(score_series.iloc[1:n+1].index)
    print(top_n_indexes)
    
    #populating the list with titles of n matching movie
    for i in top_n_indexes:
        recommended_movies.append(list(data1.index)[i])
    return recommended_movies
#movie = input("Enter the movie name you wished to be recommended similar movies: ").lower()
movie = 'cleopatra'
#n = int(input("How many movies do you want to be recommended: "))
n = 10
movie
recommendations(movie, n)
indices[indices == movie].index[0]
pd.Series(cosine_sim[indices[indices == movie].index[0]])