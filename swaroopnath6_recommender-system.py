# Imports

import sys

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import time

from sklearn.metrics import mean_squared_error

from sklearn.neighbors import BallTree
dataset = pd.read_csv('../input/amazon-ratings/ratings_Beauty.csv')
print(dataset.info())

dataset.head(4)
class Recommender:

    def __init__(self, strategy = 'user', neighbours = 10):

        self.strategy = strategy

        self.num_neighbours = neighbours

        if strategy is 'user':

            self.columns = ['User_' + str(index) for index in range(1, self.num_neighbours + 1)]

        elif strategy is 'item':

            self.columns = ['Item_' + str(index) for index in range(1, self.num_neighbours + 1)]

        

    def fit(self, matrix):

        if self.strategy is 'user':

            # User - User based collaborative filtering

            start_time = time.time()

            self.user_item_matrix = matrix

            self.mapper_indices = matrix.index

            self.user_tree = BallTree(matrix, leaf_size = self.num_neighbours * 2)

            time_taken = time.time() - start_time

            print('Model built in {} seconds'.format(time_taken))

            return self

        

        elif self.strategy is 'item':

            # Item - Item based collaborative filtering

            start_time = time.time()

            matrix = matrix.T

            self.item_user_matrix = matrix

            self.mapper_indices = matrix.index

            self.item_tree = BallTree(matrix, leaf_size = self.num_neighbours * 2)

            time_taken = time.time() - start_time

            print('Model built in {} seconds'.format(time_taken))

            return self

            

                    

    def predict(self, X_test):

        if self.strategy is 'user':

            y_pred = pd.Series(index = X_test.index)

            

            for index in tqdm(X_test.index, desc = 'Predicting Ratings'):

                row = X_test.loc[index]

                target_user = row['UserId']

                target_product = row['ProductId']

                

                if target_user not in self.user_item_matrix.index:

                    y_pred[index] = 0

                    continue

                

                user_attributes = self.user_item_matrix.loc[target_user]

                _, neighbour_indices = self.user_tree.query(user_attributes.values.reshape(1, -1), k = self.num_neighbours)

                

                rating = 0

                for neighbour_index in neighbour_indices:

                    user = self.mapper_indices[neighbour_index]

                    if target_product in self.user_item_matrix.loc[user].index:

                        rating += self.user_item_matrix.loc[user, target_product]

                    else:

                        rating += 0

                avg_rating = rating/self.num_neighbours

                y_pred.loc[index] = avg_rating

                

            return y_pred.values

        

        elif self.strategy is 'item':

            y_pred = pd.Series(index = X_test.index)

            

            for index in tqdm(X_test.index, desc = 'Predicting Ratings'):

                row = X_test.loc[index]

                target_user = row['UserId']

                target_product = row['ProductId']

                

                if target_product not in self.item_user_matrix.index:

                    y_pred[index] = 0

                    continue

                

                item_attributes = self.item_user_matrix.loc[target_product]

                _, neighbour_indices = self.item_tree.query(item_attributes.values.reshape(1, -1), k = self.num_neighbours)

                

                rating = 0

                for neighbour_index in neighbour_indices:

                    product = self.mapper_indices[neighbour_index]

                    if target_user in self.item_user_matrix.loc[product].index:

                        rating += self.item_user_matrix.loc[product, target_user]

                    else:

                        rating += 0

                avg_rating = rating/self.num_neighbours

                y_pred.loc[index] = avg_rating

                

            return y_pred.values

        

    def recommend_items(self, id, num_recommendations = 10):

        if self.strategy is 'user':

            user_id = id

            

            if user_id not in self.user_item_matrix.index:

                # New user - We will be looking at this case later on

                return None

            

            user_attributes = self.user_item_matrix.loc[user_id]

            distances, neighbour_indices = self.user_tree.query(user_attributes.values.reshape(1, -1), k = self.num_neighbours + 1)

            distances = distances[0]

            neighbour_indices = neighbour_indices[0]

            

            # We will be scoring each product by the user's distance from the target user and the 

            # rating given by the user to the item.

            recommendations = pd.DataFrame(columns = ['ProductId', 'Recommendability'])

            

            for index, neighbour_index in enumerate(neighbour_indices):

                user = self.mapper_indices[neighbour_index]

                user_similarity = 1 - distances[index]

                products_with_ratings = self.user_item_matrix.loc[user]

                

                for product_id in products_with_ratings.index:

                    recommendability = user_similarity * products_with_ratings.loc[product_id]

                    recommendation = {'ProductId': product_id, 'Recommendability': recommendability}

                    recommendations = recommendations.append(recommendation, ignore_index = True)

            

            recommendations.sort_values(by = 'Recommendability', ascending = False, inplace = True)

            recommendations = recommendations[~recommendations.duplicated('ProductId')]

            

            max_recommendations = min(num_recommendations, recommendations.shape[0])

            return recommendations.iloc[:max_recommendations, :-1]

        

        elif self.strategy is 'item':

            product_id = id

            

            if product_id not in self.item_user_matrix.index:

                # New product - We will be looking at this case later on

                return None

            

            product_attributes = self.item_user_matrix.loc[product_id]

            distances, neighbour_indices = self.item_tree.query(product_attributes.values.reshape(1, -1), k = num_recommendations)

            distances = distances[0]

            neighbour_indices = neighbour_indices[0]

            

            recommendations = pd.DataFrame(columns = ['ProductId', 'Recommendability'])

            

            for index, neighbour_index in enumerate(neighbour_indices):

                product_id = self.mapper_indices[neighbour_index]

                product_similarity = 1 - distances[index]

                

                recommendation = {'ProductId': product_id, 'Recommendability': product_similarity}

                recommendations = recommendations.append(recommendation, ignore_index = True)

            

            recommendations.sort_values(by = 'Recommendability', ascending = False, inplace = True)

            

            return recommendations.iloc[1:, :-1]
# Let's look at the ratings count of the products

gb_product = dataset.groupby('ProductId').size()

gb_product = gb_product.sort_values()
plt.plot(range(1, gb_product.shape[0] + 1), gb_product.values)

plt.show()

high_rated_products = gb_product[gb_product >= 500]

plt.plot(range(1, high_rated_products.shape[0] + 1), high_rated_products.values)

print(high_rated_products.shape[0])
data_complete = dataset.loc[dataset['ProductId'].isin(high_rated_products.index)]

data = data_complete.iloc[:, :-1]
data_train, data_test, _, _ = train_test_split(data, np.zeros(data.shape[0]), test_size = 0.2)
user_item_matrix_raw = pd.pivot_table(data_train, index = 'UserId', 

                                  columns = 'ProductId', values = 'Rating', aggfunc = np.sum)
print(user_item_matrix_raw.shape)

user_item_matrix_raw.head(4)
sparsity = np.isnan(user_item_matrix_raw.values).sum()/np.prod(user_item_matrix_raw.shape)

print('The sparsity of the matrix is: {}'.format(sparsity))
# Filling the NaN values with mean of the column

user_item_matrix = user_item_matrix_raw.fillna(user_item_matrix_raw.mean())

user_item_matrix.head(4)
user_item_matrix_rating = user_item_matrix.apply(lambda row: row - 3)
recommender = Recommender().fit(user_item_matrix)
recommender_rating = Recommender().fit(user_item_matrix_rating)
X_test = data_test.iloc[:, :-1]

y_test = data_test.iloc[:, -1]
# Predicting using the two recommender models for normal as well as overhead rating 

y_pred = recommender.predict(X_test)

y_pred_rating = recommender_rating.predict(X_test)



y_pred_rating += 3



rmse = np.sqrt(mean_squared_error(y_test, y_pred))

rmse_rating = np.sqrt(mean_squared_error(y_test, y_pred_rating))



print('RMSE using first perscpective: {}\nRMSE using second perspective: {}'.format(rmse, rmse_rating))
recommender_item = Recommender(strategy = 'item').fit(user_item_matrix)
recommender_item_rating = Recommender(strategy = 'item').fit(user_item_matrix_rating)
# Predicting using the two recommender models for normal as well as overhead rating 

y_item_pred = recommender_item.predict(X_test)

y_item_pred_rating = recommender_item_rating.predict(X_test)



y_item_pred_rating += 3



rmse = np.sqrt(mean_squared_error(y_test, y_item_pred))

rmse_rating = np.sqrt(mean_squared_error(y_test, y_item_pred_rating))



print('RMSE using first perscpective: {}\nRMSE using second perspective: {}'.format(rmse, rmse_rating))
recommender.recommend_items('ABQAIIBTTEKVM')
recommender_rating.recommend_items('ABQAIIBTTEKVM')
recommender_item.recommend_items('B004OHQR1Q')
recommender_item_rating.recommend_items('B004OHQR1Q')
user_item_matrix_baseline = user_item_matrix_raw.fillna(3) - 3

user_item_matrix_baseline.head(4)
recommender_baseline = Recommender().fit(user_item_matrix_baseline)
y_pred_baseline = recommender_baseline.predict(X_test)

y_pred_baseline += 3



rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))

print('RMSE using User-User CF: {}'.format(rmse_baseline))
recommender_item_baseline = Recommender(strategy = 'item').fit(user_item_matrix_baseline)
y_pred_item_baseline = recommender_item_baseline.predict(X_test)

y_pred_item_baseline += 3



rmse_item_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))

print('RMSE using Item-Item CF: {}'.format(rmse_item_baseline))
class ModelRecommender:

    def __init__(self, strategy = 'SVD', latent_factors = 5, num_epochs = 10, reg_param = 0.01):

        self.strategy = strategy

        self.latent_factors = latent_factors = 5

        self.num_epochs = 10

        self.reg_param = reg_param

        self.learning_rate = 0.0005

        

    def fit(self, matrix):

        m, n = matrix.shape

        self.P = pd.DataFrame(np.random.rand(m, self.latent_factors), index = matrix.index) # Users

        self.Q = pd.DataFrame(np.random.rand(n, self.latent_factors), index = matrix.columns) # Products

        

        users = list(matrix.index)

        products = list(matrix.columns)



        for epoch in tqdm(range(self.num_epochs), desc = 'Epoch'):

            for user, product in zip(users, products):

                error = matrix.loc[user, product] - self.predictions(self.P.loc[user].values, self.Q.loc[product].values)

                self.P.loc[user] += self.learning_rate * (error * self.Q.loc[product].values - self.reg_param * self.P.loc[user].values)

                self.Q.loc[product] += self.learning_rate * (error * self.P.loc[user].values - self.reg_param * self.Q.loc[product].values)

                

    def predictions(self, P, Q):

        return np.dot(P, Q.T)

    

    def predict(self, X_test):

        y_pred = pd.Series(index = X_test.index)

        

        for index, row in X_test.iterrows():

            user_id = row['UserId']

            product_id = row['ProductId']

            if user_id not in self.P.index:

                y_pred.loc[index] = 0

                continue

            if product_id not in self.Q.index:

                y_pred.loc[index] = 0

                continue

            pred = self.predictions(self.P.loc[user_id].values, self.Q.loc[product_id].values)

            y_pred.loc[index] = pred

        

        return y_pred.values

    

    def recommend(self, user_id, num_recommendations = 10):

        recommendations = pd.DataFrame(columns = ['ProductId', 'Recommendability'])

        

        for product_id in self.Q.index:

            recommendability = self.predictions(self.P.loc[user_id].values, self.Q.loc[product_id].values)

            recommendations = recommendations.append({'ProductId': product_id, 'Recommendability': recommendability}, ignore_index = True)

            

        recommendations.sort_values(by = 'Recommendability', ascending = False, inplace = True)

        

        max_recommendations = min(num_recommendations, self.Q.shape[0])

        return recommendations.iloc[:max_recommendations, 0]
# Filling NaN values with baseline rating

user_item_matrix = user_item_matrix_raw.fillna(3)

user_item_matrix.head(4)
user_item_matrix_rating = user_item_matrix - 3
recommender = ModelRecommender()

recommender.fit(user_item_matrix)
X_test = data_test.iloc[:, :-1]

y_test = data_test.iloc[:, -1]
y_pred = recommender.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))



print('RMSE using normal rating method: {}'.format(rmse))
recommender_rating = ModelRecommender()

recommender_rating.fit(user_item_matrix_rating)
y_pred_rating = recommender.predict(X_test)

y_pred_rating += 3

rmse_rating = np.sqrt(mean_squared_error(y_test, y_pred_rating))



print('RMSE using baseline rating method: {}'.format(rmse_rating))
recommender.recommend('ABQAIIBTTEKVM', 10)
recommender_rating.recommend('ABQAIIBTTEKVM', 10)