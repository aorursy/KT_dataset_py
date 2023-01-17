# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
"""below using https://github.com/nihal223/Amazon-Product-Recommendation-System/blob/master/Recommender%20Implement.ipynb



"""


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pandas import DataFrame 

import nltk



from sklearn.neighbors import NearestNeighbors

from sklearn.linear_model import LogisticRegression

from sklearn import neighbors

from scipy.spatial.distance import cosine

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectKBest

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



import re

import string

from sklearn.metrics import mean_squared_error
import pandas as pd

df = pd.read_csv("../input/blackfriday/BlackFriday.csv")
df = df.sort_values(by='Product_ID')

df = df.reset_index(drop=True)

count = df.groupby("Product_ID", as_index=False).mean()

count_users = df.groupby("User_ID", as_index=False).count()



df_clean = df[['Product_ID', 'User_ID', 'Rating']]

print(df_clean)

n_users = df_clean.User_ID.unique().shape[0]

n_items = df_clean.Product_ID.unique().shape[0]

print ('Number of product = ' + str(n_items)+' | Number of users = ' + str(n_users))
items_df = count[['Product_ID']]

items_df

users_df = count_users[['User_ID']]

users_df

users_list = users_df.values

users_list
from sklearn.model_selection import train_test_split

trainSet, testSet = train_test_split(df_clean, test_size=0.2,stratify=df_clean['User_ID'])



print(len(testSet))

print(len(trainSet))



print(testSet.shape)

print(trainSet.shape)
mean_items = trainSet.groupby(['Product_ID']).mean()



mean_users = trainSet.groupby(['User_ID']).mean()



mean_overall = trainSet["Rating"].mean()



print(mean_items,mean_users,mean_overall)
trainSet
train_data_matrix_na = trainSet.pivot(index='Product_ID', columns='User_ID', values='Rating')

train_data_matrix = train_data_matrix_na

train_data_matrix
train_values = train_data_matrix.values

train_sum = train_data_matrix.sum(axis = 1)

train_count = train_data_matrix.count(axis =1).astype('float64')

train_normal = train_sum.div(train_count,axis=0)



train_data_matrix = train_data_matrix.sub(train_normal,axis=0).fillna(0)

train_data_matrix
test_data_matrix_na = testSet.pivot(index='Product_ID', columns='User_ID', values='Rating')

test_data_matrix = test_data_matrix_na.fillna(0)

test_data_matrix
from scipy.spatial.distance import correlation

neighbor = NearestNeighbors(n_neighbors=4, algorithm='auto', metric = 'correlation').fit(train_data_matrix)

distances, indices = neighbor.kneighbors(train_data_matrix)

print(distances)

print("__________________________")

print(indices)
reference_index = train_data_matrix.index.values

reference_index
for line in trainSet.values[:5]:

    print(line)
pred_rating = {}



for line in testSet.values[:500]:

    myuser = line[1]

    myitem = line[0]

    myrate = line[2]

# myuser = 'AMCAID3LTHKEC'   #testSet[1:4]['reviewerID'].iat[0]

# myitem = 'B00007LVCN'  #testSet[5:6]['asin'].iat[0]



    its_location, =np.where(reference_index == myitem)

    its_location = its_location[0]



    myitem_distances = distances[its_location:its_location+1][0]

    myitem_indices = indices[its_location:its_location+1][0]

    

    

    bi = mean_items.loc[myitem]['Rating'] - mean_overall

    bx = mean_users.loc[myuser]['Rating'] - mean_overall

    bxi = mean_overall + bi + bx



#     print(its_location)

#     print(myitem_distances)

#     print(myitem_indices)



#     print("____________________________________")



    item_occurences_in_train = []

    for i in range(4):

        item_occurence = trainSet.loc[(trainSet['Product_ID'] == reference_index[myitem_indices[i]]) & (trainSet['User_ID'] == myuser)]

        bi_n = mean_items.loc[reference_index[myitem_indices[i]]]['Rating'] - mean_overall

        bix_n = mean_overall + bi_n + bx

        item_occurences_in_train.append((item_occurence, myitem_distances[i], bix_n))



#     print(item_occurences_in_train)



#     print("____________________________________")



    

#     print(bxi)



#     print("____________________________________")



    sim_rating = 0

#     divide_rating = 0

    # pred_rating((myuser,myitem)) = bxi + sim_rating



    for line in item_occurences_in_train:

#         print(line[0]['overall'].values)

        if not line[0].empty:

#             print(line[2])

            sim_rating = sim_rating + (line[0]['Rating'].values-line[2])*(line[1])

            sim_rating=sim_rating[0]

#             print(sim_rating)

    #         divide_rating += (1-line[1]) 

#     print(sim_rating)



#     print("____________________________________")



    pred = bxi + sim_rating

    pred_rating[(myitem, myuser, myrate)] = pred



y_true = []

y_pred = []

for key in pred_rating:

    y_true.append(key[2])

    y_pred.append(pred_rating[key])

#     print(key," __ ", pred_rating[key])

# y_pred.append(pred_rating.values())

# print(y_pred[0])



mse = mean_squared_error(y_true, y_pred, multioutput='uniform_average')

print(mse)



df_clean_matrix = df_clean.pivot(index='Product_ID', columns='User_ID', values='Rating').fillna(0)

df_clean_matrix = df_clean_matrix.T

R = (df_clean_matrix).as_matrix()

R
user_ratings_mean = np.mean(R, axis = 1)

R_demeaned = R - user_ratings_mean.reshape(-1, 1)

R_demeaned


from scipy.sparse.linalg import svds

U, sigma, Vt = svds(R_demeaned)
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

preds_df = pd.DataFrame(all_user_predicted_ratings, columns = df_clean_matrix.columns)

preds_df['User_ID'] = users_df

preds_df.set_index('User_ID', inplace=True)

preds_df
def recommend_method(predictions_df, movies_df, original_ratings_df, num_recommendations=5):

    

    # Get and sort the user's predictions

    sorted_user_predictions = predictions_df.loc[1006038].sort_values(ascending=False)

    

    # Get the user's data and merge in the movie information.

    user_data = original_ratings_df[original_ratings_df.User_ID == 1006038]

    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'Product_ID', right_on = 'Product_ID').

                     sort_values(['Rating'], ascending=False)

                 )



    print ('User {0} has already rated {1} items.'.format(1006038, user_full.shape[0]))

    print ('Recommending the highest {0} predicted ratings for items not already rated.'.format(num_recommendations))

    

    # Recommend the highest predicted rating movies that the user hasn't seen yet.

    recommendations = (movies_df[~movies_df['Product_ID'].isin(user_full['Product_ID'])].

         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',

               left_on = 'Product_ID',

               right_on = 'Product_ID').

         rename(columns = {1006038: 'Predictions'}).

         sort_values('Predictions', ascending = False).

                       iloc[:num_recommendations, :-1]

                      )



    return (sorted_user_predictions,recommendations)


already_rated,recommendations = recommend_method(preds_df, items_df, df_clean, 10)
already_rated
recommendations
####################################################################3



"""The below using Knearest neighbours



https://github.com/krishnaik06/Recommendation_complete_tutorial/blob/master/KNN%20Movie%20Recommendation/KNNRecommendation.ipynb"""

import pandas as pd

import numpy as np
from scipy.sparse import csr_matrix



train_data_df_matrix = csr_matrix(train_data_matrix.values)



from sklearn.neighbors import NearestNeighbors





model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')

model_knn.fit(train_data_df_matrix)
train_data_matrix.shape


query_index = np.random.choice(train_data_matrix.shape[0])

print(query_index)

distances, indices = model_knn.kneighbors(train_data_matrix.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)

train_data_matrix.head()


for i in range(0, len(distances.flatten())):

    if i == 0:

        print('Recommendations for {0}:\n'.format(train_data_matrix.index[query_index]))

    else:

        print('{0}: {1}, with distance of {2}:'.format(i, train_data_matrix.index[indices.flatten()[i]], distances.flatten()[i]))