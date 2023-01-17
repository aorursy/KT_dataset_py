import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import warnings

warnings.filterwarnings("ignore")
data = pd.read_excel('/kaggle/input/movie-datasets/Hollywood_Movie_Dataset.xlsx')
data.head()
data.columns
# We need to convert certain features (useful for extracting the contents) to string
data[['keywords','cast','genres','director']] = data[['keywords','cast','genres','director']].astype(str)
columns = ['keywords','cast','genres','director']

def combination(n):

    return n['keywords']+" "+n['cast']+" "+n['genres']+" "+n['director']
for columns in columns:

    data[columns] = data[columns].fillna('')
data['combination'] = data.apply(combination,axis=1)
# We can use CountVectorizer() object for getting the count matrix form the combined text

count_vector = CountVectorizer()

count_matrix = count_vector.fit_transform(data['combination']) 
similarity = cosine_similarity(count_matrix)
# We need the title and the index to match for the following loop to rank movies

def index(index):

    return data[data.index == index]["title"].values[0]

def title(title):

    return data[data.title == title]["index"].values[0]
user_input = "Skyfall"

index_match = title(user_input)

similar_movies = list(enumerate(similarity[index_match]))

final_list = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]

i=0

print("Top 10 movies similar to '"+user_input+"' are:\n")

for x in final_list:

    print(index(x[0]))

    i=i+1

    if i>10:

        break