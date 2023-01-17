import pandas as pd 

import numpy as np 

import pickle 



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity



df_movies= pd.read_csv("../input/imdb-extensive-dataset/IMDb movies.csv")
df_movies.head()
df_movies.dtypes
df_movies= df_movies.drop(["original_title", "duration", "metascore", "worlwide_gross_income", "date_published", "production_company", "budget", "usa_gross_income", "reviews_from_users", "reviews_from_critics"], axis=1)
df_movies= df_movies[df_movies.year > 1990]

df_movies= df_movies[df_movies.avg_vote > 6.5]
quant= df_movies["votes"].quantile(0.8)



print("Number of votes: " +str(quant))



mean= df_movies["avg_vote"].mean()



print("Average vote: " +str(mean))
movies= df_movies.copy().loc[df_movies["votes"] >= quant]



movies.shape
def Wrate(df, m=mean,  q=quant):

    v=df["votes"]

    R= df["avg_vote"]

    

    return (v/(v+q)* R) + (q/(q+v)* m)
movies["score"]= movies.apply(Wrate, axis=1)
movies= movies.sort_values("score", ascending= False)



movies["title"].head()
def clean_data(x):

    if isinstance(x, list):

        return [str.lower(i.replace(" ", "")) for i in x]

    

    else: 

        

        if isinstance(x, str):

            return str.lower(x.replace(" ", ""))

        

        else: return ""
features=["actors", "director", "genre", "description"]



for f in features:

    df_movies[f]= df_movies[f].apply(clean_data)
def create_col(x):

    return " ".join(x["actors"]) + " " + x["director"] + " " + " ".join(x["genre"])+ " "+ " ".join(x["description"])





df_movies["features"] = df_movies.apply(create_col, axis=1)
df_movies.dtypes
tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(df_movies["features"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim.shape
pickle.dump(cosine_sim, open("cos.pkl", "wb"))
def get_recom(title, cosine_sim=cosine_sim):

    # Get the index of the movie that matches the title

    idx = indices[title]



    # Get the pairwise similarity scores of all movies with that movie

    sim_scores = list(enumerate(cosine_sim[idx]))



    # Sort the movies

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



    sim_scores = sim_scores[1:10]



    # Get the movie indices

    movie_indices = [i[0] for i in sim_scores]



    return df_movies["title"].iloc[movie_indices]
df_movies = df_movies.reset_index()

indices = pd.Series(df_movies.index, index=df_movies["title"])
df_movies.to_csv("Mov.csv")
get_recom("JFK", cosine_sim)