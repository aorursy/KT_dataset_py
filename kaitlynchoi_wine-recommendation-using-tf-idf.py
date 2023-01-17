import numpy as np 

import pandas as pd 



# Importe two csv files

original_data_1=pd.read_csv("/kaggle/input/wine-reviews/winemag-data_first150k.csv")

original_data_2=pd.read_csv("/kaggle/input/wine-reviews/winemag-data-130k-v2.csv")
original_data_1.head()
original_data_2.head()
# Create a single DataFrame that contains variety and description only. Delete any rows that are duplicated or contain missing data.

variety_description= original_data_1[["variety", "description"]].append(original_data_2[["variety", "description"]])

variety_description=variety_description.drop_duplicates().dropna()

variety_description.head()
# How many grape varieties are there in this DataFrame?

len(variety_description["variety"].unique().tolist())
variety_description.shape
# Create and display the chart showing the number of reviews per grape variety for the top 30 wines

variety_description["variety"].value_counts().iloc[:30].plot.bar()
# Count the number of reviews per grape variety. This returns a series.

variety_rev_number=variety_description["variety"].value_counts()



# Convert the Series to Dataframe

df_rev_number=pd.DataFrame({'variety':variety_rev_number.index, 'rev_number':variety_rev_number.values})

df_rev_number[(df_rev_number["rev_number"]>1)].shape
# Create a ist of grape varieties that have more than one review

variety_multi_reviews=df_rev_number[(df_rev_number["rev_number"]>1)]["variety"].tolist()



# Create a ist of grape varieties that have only one review

variety_one_review=df_rev_number[(df_rev_number["rev_number"]==1)]["variety"].tolist()
# This demo is modified from https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/



from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer



docs=["there was a king named Matati", 

      "He had a lot of gold", 

      "He loved the gold most", 

      "The beginning of the Matati gold story"]



cv=CountVectorizer()



word_count_vect=cv.fit_transform(docs)



# Display the result of CountVectorizer output (Reference: https://gist.github.com/larsmans/3745866)



print("the result of CountVectorizer") 

print(pd.DataFrame(word_count_vect.A, columns=cv.get_feature_names()).to_string())





# Use TfidfTransformer to compute the IDF values

tfidf_trans=TfidfTransformer(smooth_idf=True, use_idf=True)

tfidf_trans.fit(word_count_vect)



# Display the IDF value for each term in the text

df_idf=pd.DataFrame(tfidf_trans.idf_, index=cv.get_feature_names(), columns=["idf_values"])

df_idf.sort_values(by=['idf_values'])
variety_description=variety_description.set_index("variety")
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer



variety_description_2=pd.DataFrame(columns=["variety","description"])



# Define a CountVectorizer object

    # stop_words="english": Remove all the uninformative words such as 'and', 'the' from analysis

    # ngram=range(1,2): means unigrams and bigrams

cv=CountVectorizer(stop_words="english", ngram_range=(2,2))



# Define a TfidfTransformer object

tfidf_transformer=TfidfTransformer(smooth_idf=True, use_idf=True)



for grape in variety_multi_reviews:



    df=variety_description.loc[[grape]]



    # Generate word counts for the words used in the reviews of a specific grape variety

    word_count_vector=cv.fit_transform(df["description"])



    # Compute the IDF values

    tfidf_transformer.fit(word_count_vector)



    # Obtain top 100 common words (meaning low IDF values) used in the reviews. Put the IDF values in a DataFrame

    df_idf=pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])

    df_idf.sort_values(by=["idf_weights"], inplace=True)



    # Collect top 100 common words in a list

    common_words=df_idf.iloc[:100].index.tolist()

   

    # Convert the list to a string and create a dataframe

    common_words_str=", ".join(elem for elem in common_words)

    new_row= {"variety":grape, "description":common_words_str}



    # Add the variety and its common review words to a new dataframe

    variety_description_2=variety_description_2.append(new_row, ignore_index=True)



variety_description_2=variety_description_2.set_index("variety")

variety_description_2=variety_description_2.append(variety_description.loc[variety_one_review])

variety_description_2
variety_description_2.shape
# Load a relevant library

from sklearn.feature_extraction.text import TfidfVectorizer



# Define a TfidVectorizer object. Remove all the uninformative words such as 'and,' 'the,' and 'him' from analysis. Bigrams only (ngram_range=(2,2)).

tfidf=TfidfVectorizer(stop_words="english", ngram_range=(2,2))



# Count the words in each description, calculate idf, and multiply idf by tf.

tfidf_matrix=tfidf.fit_transform(variety_description_2["description"])



# Resulting matrix should be # of descriptions (row) x # of bigrams (column)

tfidf_matrix.shape
# Since we used TfidfVectorizer to convert the text into a matrix, we can use linear_kernel to get cosine similarity, instead of sklearn's cosine_similarity

# Load linear_kernel

from sklearn.metrics.pairwise import linear_kernel



# Compute the cosine similarity

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# Create a Series, where the index is the grape variety and the element is the index of the wine in the dataset.

variety_description_2=variety_description_2.reset_index()

indices = pd.Series(variety_description_2.index, index=variety_description_2['variety'])
# Make a function that takes in the grape variety as an input and produces a DataFrame of three similar varieties and key words of their reviews



def what_should_I_drink_next(grape, cosine_sim=cosine_sim):

    # Get the index of the input wine

    idx = indices[grape]



    # Get the pairwise similarity scores between the input wine and all the wines

    sim_scores = list(enumerate(cosine_sim[idx]))



    # Sort the wines based on the similarity scores

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



    # Select the top three similarity scores

    sim_scores = sim_scores[1:4]



    # Get the grape variety indices

    wine_idx_list = [i[0] for i in sim_scores]

     

    # Create the output dataframe

    df=pd.DataFrame(columns=["similar wines", "Top 6 common words in wine reviews"])

     

    for wine_idx in wine_idx_list:

     

        g_variety=variety_description_2.iloc[wine_idx]["variety"]

    

        # Get top 6 common words in the review

        des=variety_description_2.iloc[wine_idx]["description"]

        

        if g_variety in variety_multi_reviews:     # If the wine has more than one reviews

            des_split=des.split(", ")

            key_words_list=des_split[:6]

            key_words_str=", ".join(key_words_list)

        

        else:

            key_words_str = des

            

        new_row={"similar wines": g_variety, "Top 6 common words in wine reviews": key_words_str}

        df=df.append(new_row, ignore_index=True)

    

    df.set_index("similar wines") 

    

    # Widen the column width so that all common words could be displayed

    pd.set_option('max_colwidth', 500)

   

    return df  
what_should_I_drink_next("Pinot Noir")
what_should_I_drink_next("Shiraz")