import os



import numpy as np

import pandas as pd
# For systems using OpenBLAS

os.environ['OPENBLAS_NUM_THREADS'] = '1'



# For systems using Intel MKL

# os.environ['MKL_NUM_THREADS'] = '1'



pd.set_option('mode.chained_assignment', None)
# Load the data

raw_data = pd.read_csv('../input/events.csv')

raw_data = raw_data.drop(labels=raw_data.columns[4], axis=1)

raw_data.columns = ['timestamp', 'user', 'event', 'item']
# Drop rows with users that have less than 10 interactions

user_value_counts = raw_data['user'].value_counts()

data = raw_data[raw_data['user'].isin(user_value_counts[user_value_counts >= 10].index)]
# Set all event (weight) values to 1.0

data.loc[:, 'event'] = 1.0
# Create numeric user_id and item_id columns

data['user'] = data['user'].astype('category')

data['item'] = data['item'].astype('category')

data['user_id'] = data['user'].cat.codes

data['item_id'] = data['item'].cat.codes
# Separate training and testing data

latest_timestamp = data['timestamp'].max()

day_in_ms = 86_400_000

last_day_timestamp = latest_timestamp - day_in_ms



data_test = data.copy()

data_test = data_test.loc[data_test.timestamp >= last_day_timestamp].copy()

data = data.loc[data.timestamp < last_day_timestamp].copy()
import scipy.sparse as sparse

import implicit
# The implicit library expects data as an item-user matrix so we create two matrices,

# one for fitting the model (item-user) and one for recommendations (user-item)

sparse_item_user = sparse.csr_matrix((data['event'].astype(float), (data['item_id'], data['user_id'])))

sparse_user_item = sparse.csr_matrix((data['event'].astype(float), (data['user_id'], data['item_id'])))
# Initialize the als model and fit it using the sparse item-user matrix

model = implicit.als.AlternatingLeastSquares(factors=128, regularization=0.1, iterations=20)
# Calculate the confidence by multiplying it by our alpha value

# Alpha = (sparse_item_user.shape[0] * sparse_item_user.shape[1] - sparse_item_user.nnz) / sum(sparse_item_user.data)

alpha_val = 3080

data_conf = (sparse_item_user * alpha_val).astype('double')
# Fit the model

model.fit(data_conf, show_progress=False)
def get_recommendations(user_ids, user_items, k=10):

    return {user_id: model.recommend(user_id, user_items, N=k, filter_already_liked_items=False)

            for user_id in user_ids}
def get_intersections(test_user_ids, test_data, recommendations):

    intersections = {}

    for test_user_id in test_user_ids:

        last_day_items = test_data.loc[test_data.user_id == test_user_id]['item_id'].values

        if recommendations[test_user_id][0][1] != 0.0:

            recommended, _ = list(zip(*recommendations[test_user_id]))

            intersections[test_user_id] = set(recommended).intersection(set(last_day_items))

    return intersections
def get_hit_rate(intersections):

    user_count = 0

    item_count = 0

    for intersection in intersections:

        intersection_len = len(intersections[intersection])

        if intersection_len > 0:

            user_count += 1

            item_count += intersection_len

    return user_count / len(intersections)
# Create a list of all test users

users = list(np.sort(data_test.user_id.unique()))
# Top 10 hit rate

recommendations_10 = get_recommendations(users, sparse_user_item, k=10)

intersections_10 = get_intersections(users, data_test, recommendations_10)

hit_rate_10 = get_hit_rate(intersections_10)

print(f'Top 10 hit rate: {hit_rate_10:%}')
# Top 50 hit rate

recommendations_50 = get_recommendations(users, sparse_user_item, k=50)

intersections_50 = get_intersections(users, data_test, recommendations_50)

hit_rate_50 = get_hit_rate(intersections_50)

print(f'Top 50 hit rate: {hit_rate_50:%}')
# Top 100 hit rate

recommendations_100 = get_recommendations(users, sparse_user_item, k=100)

intersections_100 = get_intersections(users, data_test, recommendations_100)

hit_rate_100 = get_hit_rate(intersections_100)

print(f'Top 100 hit rate: {hit_rate_100:%}')