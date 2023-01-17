from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all' #this helps to full output and not only the last lines of putput



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import scipy.sparse



import warnings

warnings.simplefilter('ignore')
data = pd.read_csv('../input/restaurant-data-with-consumer-ratings/rating_final.csv')

data.head()
data.info()
#Summary statistics

data.describe(include = 'all').transpose()
#No.of unique users, restaurants, no. of ratings, food_ratings, service_ratings

print('Unique users: ', data['userID'].nunique())

print('Unique restaurant: ', data['placeID'].nunique())

print('Total no.of ratings given: ', data['rating'].count())

print('Total no.of food ratings given: ', data['food_rating'].count())

print('Total no.of service ratings given: ', data['service_rating'].count())
# How many times has a user rated

most_rated_users = data['userID'].value_counts()

most_rated_users
#How many times has a restaurant been rated

most_rated_restaurants = data['placeID'].value_counts()

most_rated_restaurants
#What's the rating distribution

plt.figure(figsize = (8,5))

sns.countplot(data['rating'])
#What's the food rating distribution

plt.figure(figsize = (8,5))

sns.countplot(data['food_rating'])
#What's the service rating distribution

plt.figure(figsize = (8,5))

sns.countplot(data['service_rating'])
#How many users have rated more than n places ?

n = 3

user_counts = most_rated_users[most_rated_users > n]

len(user_counts)

user_counts
#No. of ratings given

user_counts.sum()
#Retrieve all ratings given by the above users from the full data

data_final = data[data['userID'].isin(user_counts.index)]

data_final
final_ratings_matrix = data_final.pivot(index = 'userID', columns = 'placeID', values = 'rating').fillna(0)

final_ratings_matrix.head()
#Lets calculate the density of the matrix. This is to see how many possible ratings could be given and exactly how many ratings were given 



#No. of ratings given

given_num_of_ratings = np.count_nonzero(final_ratings_matrix)

print('given_num_of_ratings: ', given_num_of_ratings)



#Total no. of ratings that could have been given 

possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]

print('possible_num_of_ratings: ', possible_num_of_ratings)



#Calculate matrix density

density = (given_num_of_ratings / possible_num_of_ratings) * 100

print('density: {:4.2f}%'.format(density))
#No. of users who have rated a resto

data_grouped = data.groupby('placeID').agg({'userID':'count'}).reset_index()

data_grouped.rename(columns = {'userID': 'score'}, inplace = True )

data_sort = data_grouped.sort_values(['score','placeID'], ascending = False)

data_sort.head()
#Let's rank them based on scores

data_sort['Rank'] = data_sort['score'].rank(ascending = 0, method = 'first')

pop_recom = data_sort

pop_recom.head()
print('Here are the most popular restaurants')

pop_recom[['placeID','score','Rank']].head()
#Transform the data into a pivot table -> Format required for colab model

pivot_data = data_final.pivot(index = 'userID', columns = 'placeID', values = 'rating').fillna(0)

pivot_data.shape

pivot_data.head()
#Create a user_index column to count the no. of users -> Change naming convention of user by using counter

pivot_data['user_index'] = np.arange(0, pivot_data.shape[0],1)

pivot_data.head()
pivot_data.set_index(['user_index'], inplace = True)

pivot_data.head()
#Applying SVD method on a large sparse matrix -> To predict ratings for all resto that weren't rated by a user



from scipy.sparse.linalg import svds



#SVD

U,s, VT = svds(pivot_data, k = 10)



#Construct diagonal array in SVD

sigma = np.diag(s)



#Applying SVD would output 3 parameters namely

print("U = ",U) #Orthogonal matrix

print('************************************************')

print("S = ",s) #Singular values

print('************************************************')

print("VT = ", VT) #Transpose of Orthogonal matrix
#Predict ratings for all restaurants not rated by a user using SVD

all_user_predicted_ratings = np.dot(np.dot(U,sigma), VT)



#Predicted ratings

pred_data = pd.DataFrame(all_user_predicted_ratings, columns = pivot_data.columns)

pred_data.head()
#Recommend places with the highest predicted ratings



def recommend_places(userID, pivot_data, pred_data, num_recommendations):

    user_index  = userID-1 #index starts at 0



    sorted_user_ratings = pivot_data.iloc[user_index].sort_values(ascending = False) #sort user ratings



    sorted_user_predictions = pred_data.iloc[user_index].sort_values(ascending = False)#sorted_user_predictions

    

    temp = pd.concat([sorted_user_ratings, sorted_user_predictions], axis = 1)

    temp.index.name = 'Recommended Places'

    temp.columns = ['user_ratings', 'user_predictions']

    

    temp = temp.loc[temp.user_ratings == 0]

    temp = temp.sort_values('user_predictions', ascending = False)

    print('\n Below are the recommended places for user(user_id = {}):\n'. format(userID))

    print(temp.head(num_recommendations))
#Recommend places based on userID, past ratings, predicted ratings, num of places 



userID = 12

num_recommedations = 5

recommend_places(userID, pivot_data, pred_data, num_recommedations)
#Actual ratings given by the users

final_ratings_matrix.head()



#Average actual rating for each place



final_ratings_matrix.mean().head()
#Predicted ratings for a place

pred_data.head()



#Average predicted rating for each place

pred_data.mean().head()
#Calculate RMSE



rmse_data = pd.concat([final_ratings_matrix.mean(), pred_data.mean()], axis = 1)

rmse_data.columns = ['Avg_actual_ratings','Avg_predicted_ratings']

print(rmse_data.shape)

rmse_data['place_index'] = np.arange(0, rmse_data.shape[0],1)

rmse_data.head()
RMSE = round((((rmse_data.Avg_actual_ratings - rmse_data.Avg_predicted_ratings) ** 2).mean() ** 0.5),5)

print('\n RMSE SVD Model = {}\n'.format(RMSE))