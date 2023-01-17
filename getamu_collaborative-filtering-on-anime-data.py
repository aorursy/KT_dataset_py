# Import relevant libraries 





import pandas as pd

import numpy as np

import scipy as sp

from sklearn.metrics.pairwise import cosine_similarity

import operator

%matplotlib inline
print("Start")

anime = pd.read_csv('../input/anime.csv')

rating = pd.read_csv('../input/rating.csv')
rating.rating.replace({-1: np.nan}, regex=True, inplace = True)

rating.head()
# For this analysis I'm only interest in finding recommendations for the TV category



anime_tv = anime[anime['type']=='TV']

anime_tv.head()
# Join the two dataframes on the anime_id columns



merged = rating.merge(anime_tv, left_on = 'anime_id', right_on = 'anime_id', suffixes= ['_user', ''])

merged.rename(columns = {'rating_user':'user_rating'}, inplace = True)
# For computing reasons I'm limiting the dataframe length to 10,000 users



merged=merged[['user_id', 'name', 'user_rating']]

merged_sub= merged[merged.user_id <= 10000]

merged_sub.head()
piv = merged_sub.pivot_table(index=['user_id'], columns=['name'], values='user_rating')
print(piv.shape)

piv.head()
# Note: As we are subtracting the mean from each rating to standardize

# all users with only one rating or who had rated everything the same will be dropped



# Normalize the values

piv_norm = piv.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)





# Drop all columns containing only zeros representing users who did not rate

piv_norm.fillna(0, inplace=True)

piv_norm = piv_norm.T

piv_norm = piv_norm.loc[:, (piv_norm != 0).any(axis=0)]
piv_norm.head()
# Our data needs to be in a sparse matrix format to be read by the following functions



piv_sparse = sp.sparse.csr_matrix(piv_norm.values)
item_similarity = cosine_similarity(piv_sparse)

user_similarity = cosine_similarity(piv_sparse.T)
# Inserting the similarity matricies into dataframe objects



item_sim_df = pd.DataFrame(item_similarity, index = piv_norm.index, columns = piv_norm.index)

user_sim_df = pd.DataFrame(user_similarity, index = piv_norm.columns, columns = piv_norm.columns)
# This function will return the top 10 shows with the highest cosine similarity value



def top_animes(anime_name):

    count = 1

    print('Similar shows to {} include:\n'.format(anime_name))

    for item in item_sim_df.sort_values(by = anime_name, ascending = False).index[1:11]:

        print('No. {}: {}'.format(count, item))

        count +=1  
# This function will return the top 5 users with the highest similarity value 



def top_users(user):

    

    if user not in piv_norm.columns:

        return('No data available on user {}'.format(user))

    

    print('Most Similar Users:\n')

    sim_values = user_sim_df.sort_values(by=user, ascending=False).loc[:,user].tolist()[1:11]

    sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:11]

    zipped = zip(sim_users, sim_values,)

    for user, sim in zipped:

        print('User #{0}, Similarity value: {1:.2f}'.format(user, sim)) 
# This function constructs a list of lists containing the highest rated shows per similar user

# and returns the name of the show along with the frequency it appears in the list



def similar_user_recs(user):

    

    if user not in piv_norm.columns:

        return('No data available on user {}'.format(user))

    

    sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:11]

    best = []

    most_common = {}

    

    for i in sim_users:

        max_score = piv_norm.loc[:, i].max()

        best.append(piv_norm[piv_norm.loc[:, i]==max_score].index.tolist())

    for i in range(len(best)):

        for j in best[i]:

            if j in most_common:

                most_common[j] += 1

            else:

                most_common[j] = 1

    sorted_list = sorted(most_common.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_list[:5]    
# This function calculates the weighted average of similar users

# to determine a potential rating for an input user and show



def predicted_rating(anime_name, user):

    sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:1000]

    user_values = user_sim_df.sort_values(by=user, ascending=False).loc[:,user].tolist()[1:1000]

    rating_list = []

    weight_list = []

    for j, i in enumerate(sim_users):

        rating = piv.loc[i, anime_name]

        similarity = user_values[j]

        if np.isnan(rating):

            continue

        elif not np.isnan(rating):

            rating_list.append(rating*similarity)

            weight_list.append(similarity)

    return sum(rating_list)/sum(weight_list)    
top_animes('Cowboy Bebop')
top_users(3)
similar_user_recs(3)
predicted_rating('Cowboy Bebop', 3)
# Creates a list of every show watched by user 3



watched = piv.T[piv.loc[3,:]>0].index.tolist()
# Make a list of the squared errors between actual and predicted value



errors = []

for i in watched:

    actual=piv.loc[3, i]

    predicted = predicted_rating(i, 3)

    errors.append((actual-predicted)**2)
# This is the average squared error for user 3

np.mean(errors)
import numpy



#Function to find latent factor matrix.



def matrix_factorization(user_item, factor1, factor2, num, steps=300, alpha=0.002, beta=0.2):

    factor2 = factor2.T

    for step in range(steps):

        for i in range(len(user_item)):

            for j in range(len(user_item[i])):

                if user_item[i][j] > 0:

                    eij = user_item[i][j] - numpy.dot(factor1[i,:],factor2[:,j])

                    for k in range(K):

                        factor1[i][k] = factor1[i][k] + alpha * (2 * eij * factor2[k][j] - beta * factor1[i][k])

                        factor2[k][j] = factor2[k][j] + alpha * (2 * eij * factor1[i][k] - beta * factor2[k][j])

        eR = numpy.dot(factor1,factor2)

        e = 0

        for i in range(len(user_item)):

            for j in range(len(user_item[i])):

                if user_item[i][j] > 0:

                    e = e + pow(user_item[i][j] - numpy.dot(factor1[i,:],factor2[:,j]), 2)

                    for k in range(K):

                        e = e + (beta/2) * (pow(factor1[i][k],2) + pow(factor2[k][j],2))

        if e < 0.01:

            break

    return factor1, factor2.T

user_item = numpy.array(piv)



K = 2



#Initializing the latent factor matrix with random values.



rand_factor1 = numpy.random.rand(len(rating.user_id),K)

rand_factor2 = numpy.random.rand(len(rating.anime_id),K)



#Calling the function for finding factors of user X movies matrix.

factor1, factor2 = matrix_factorization(user_item, rand_factor1, rand_factor2, K)



#Finding the prediction matrix for all user X items

predicted = numpy.dot(factor1, factor2.T)