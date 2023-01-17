import numpy as np

import pandas as pd
df = pd.read_csv('../input/booksrecommendedbyworldfamouspersonnalities/books_clean.csv')
df.info()
df.head()
# Checking for na values



print(df['category'].isna().sum())

print(df['publication_date'].isna().sum())
#Dropping all rows with na values



df = df.dropna()

df.shape
# Conversion to lowercase

# Eliminating the space b/w first and last name as to avoid the recommender to bias towards similar occurences of a word in different people names as this would affect the similarity



df['recommender'] = df['recommender'].str.lower().str.replace(" ","").str.replace("|",' ')

df['recommender'].head()
# Conversion of author names to lowercase

df['author'] = df['author'].str.lower()

df['author'].head()
# Conversion of category to lowercase

df['category'] = df['category'].str.lower()

df['category'].head()
# Total no of categories

df['category'].value_counts().count()
# Find duplicated rows



df[df.duplicated(['title'], keep = False)] 
#Our Data after now looks like

df.head()
# Creating a function that creates our content that will be used to calculate the similarity between two books

#this content will act as our meta data that will help us determine 



def create_content(x):

    return ''.join(x['title']) + " " + ''.join(x['recommender']) + " " + ''.join(x['author']) +" "+''.join(x['category'])

df['content'] = df.apply(create_content, axis=1)
# Not using tf-idf vectorizer as it would downweight more repetitive words, but we do not need that 

# as our similarity will be reduced



# Import CountVectorizer and create the count matrix



from sklearn.feature_extraction.text import CountVectorizer



count = CountVectorizer(stop_words='english')

count_matrix = count.fit_transform(df['content'])



count_matrix
# Compute the Cosine Similarity matrix based on the count_matrix



from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(count_matrix, count_matrix)
#Construct a reverse map of indices and movie titles



indices = pd.Series(df.index, index=df['title']).drop_duplicates()



def get_recommendations(title, cosine_sim = cosine_sim):

    # Get the index of the movie that matches the title

    id_ = indices[title]



    # Get the pairwsie similarity scores of all movies with that movie

    sim_scores = list(enumerate(cosine_sim[id_]))



    # Sort the movies based on the similarity scores

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



    # Get the scores of the 10 most similar movies

    sim_scores = sim_scores[1:11]



    # Get the movie indices

    book_indices = [i[0] for i in sim_scores]



    # Return the top 10 most similar movies

    data = df['title'].iloc[book_indices]

    data.index = np.arange(1, len(data)+1)

    data = pd.DataFrame(data)



    return data

    

get_recommendations("QED")
#Construct a reverse map of indices and movie titles



indices = pd.Series(df.index, index=df['title']).drop_duplicates()



def get_recommendations(title, cosine_sim = cosine_sim):

    try:

        # Get the index of the movie that matches the title

        id_ = indices[title]



        # Get the pairwsie similarity scores of all movies with that movie

        sim_scores = list(enumerate(cosine_sim[id_]))



        # Sort the movies based on the similarity scores

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



        # Get the scores of the 10 most similar movies

        sim_scores = sim_scores[1:11]



        # Get the movie indices

        book_indices = [i[0] for i in sim_scores]



        # Return the top 10 most similar movies

        data = df['title'].iloc[book_indices]

        data.index = np.arange(1, len(data)+1)

        data = pd.DataFrame(data)



        return data

    

    except:

        

        data = df.sort_values(['recommender_count', 'publication_date'], ascending=[ False, False])

        data = data['title'].head(10)

        data.index = np.arange(1, len(data)+1)

        data = pd.DataFrame(data)

        

        return data
get_recommendations("some_random_text")