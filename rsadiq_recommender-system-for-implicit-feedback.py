import pandas as pd

data = pd.read_csv('../input/data.csv')

data.head()
print('Number of rows and columns:',data.shape)
print('Number of user:', len(data.user_id.unique()))

print('Number of item:', len(data.item_id.unique()))

print('Number of category:', len(data.category_id.unique()))

print('Number of cusine:', len(data.cusine_id.unique()))

print('Number of restaurant:', len(data.restaurant_id.unique()))
print('Number of orders on different days')

data.dow.value_counts()
print('Number of items sold on different days:')

x_d =pd.pivot_table(data, values = 'item_count', index = 'dow', aggfunc = sum)

x_d.sort_values('item_count', ascending = False)
print('Number of orders on different hours of the day:')

data.hod.value_counts()
print('Number of items sold in different hours of the day:')

x = pd.pivot_table(data, values = 'item_count', index = 'hod', aggfunc = sum)

x.sort_values('item_count', ascending = False)
import matplotlib.pyplot as plt

import seaborn as sns

data.dow = data.dow.astype('str')

order = data.dow.value_counts().index

color = sns.color_palette()[9]

n_dow = data.dow.shape[0]

dow_counts = data.dow.value_counts()

fig = plt.figure(figsize = (20,6))

plt.subplot(1,2,1)

sns.countplot(data = data, y = 'dow', order = order, color = color)

for i in range(dow_counts.shape[0]):

    count = dow_counts[i]

    string = '{:0.1f}%'.format(100*count/n_dow)

    plt.text(count+1, i, string)

    plt.xlabel('Proportion')

    plt.title('Order (%) on different days of week')

    

data.hod = data.hod.astype('str')

order_hod = data.hod.value_counts().index

n_hod = data.hod.shape[0]

hod_counts = data.hod.value_counts()

plt.subplot(1,2,2)

sns.countplot(data = data, y = 'hod', order = order_hod, color = color)

for i in range(hod_counts.shape[0]):

    count_h = hod_counts[i]

    string_h = '{:.1f}%'.format(100*count_h/n_hod)

    plt.text(count_h+i, i, string_h)

    plt.xlabel('Proportion')

    plt.title('Order (%) on different hours of the day')
a = data['dow'].astype('str')

b = data['hod'].astype('str')

H_D = a + b

data.insert(4, 'H_D', H_D)

data.head()
pd.set_option('max_r', 15)

data.H_D.value_counts()
order_H = data.H_D.value_counts().index

n_H = data.H_D.shape[0]

H_counts = data.H_D.value_counts()

fig = plt.figure(figsize = (10,100))

sns.countplot(data = data, y = 'H_D', order = order_H, color = color)

for i in range(H_counts.shape[0]):

    count_H = H_counts[i]

    string_H = '{:.1f}%'.format(100*count_H/n_H)

    plt.text(count_H+i, i, string_H)

    plt.xlabel('Proportion')

    plt.title('Order (%) of different hours of different days')
pd.pivot_table(data, values = 'item_count', index = 'hod', columns = 'dow', aggfunc = sum)
print('Number of missing values:')

data.isnull().sum()
print('item_id dtype:',data.item_id.dtype)
data['userId'] = data['user_id'].astype('category').cat.codes

data['itemId'] = data['item_id'].astype('category').cat.codes
data.head()
from sklearn.model_selection import train_test_split

train, cros_val = train_test_split(data, test_size = 0.2, random_state = 1)
train, test = train_test_split(train, test_size = 0.25, random_state = 1)
print('Splitted dataset into train set, cross validation set and test set')

print('Train shape:', train.shape)

print('Test shape:', test.shape)

print('cros_val shape:',cros_val.shape)
import scipy.sparse as sparse

user_items = sparse.csr_matrix((train['item_count'].astype(float),(train['userId'], train['itemId'])))

item_users = sparse.csr_matrix((train['item_count'].astype(float),(train['itemId'], train['userId'])))
print(item_users)
import os

os.environ['MKL_NUM_THREADS'] = '1' #To avoid multithreading.

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import implicit

model = implicit.als.AlternatingLeastSquares(factors = 500, iterations = 10)

''''Parameters: (factors=100, regularization=0.01, dtype=<type 'numpy.float32'>, use_native=True, use_cg=True, 

use_gpu=False, iterations=15, calculate_training_loss=False, num_threads=0)''';
alpha = 40

train_conf = (item_users*alpha).astype('double')
model.fit(train_conf)
import csv

fields = 'userId', 'item_list'

filename = 'rec_train.csv'

with open (filename, 'a', newline = '') as f:

    writer = csv.writer(f)

    writer.writerow(fields)

    userId = train['userId'].values.tolist()

    for user in userId:

        scores = []

        items =[]

        results = []

        results.append(user)

        recommendations = model.recommend(user, user_items, N = 5)

        for item in recommendations:

            ids, score = item

            scores.append(score)

            items.append(ids)

        results.append(items)

        writer.writerow(results)
predicted = pd.read_csv('rec_train.csv')

predicted = predicted['item_list']

import ast

predicted = [ast.literal_eval(a) for a in predicted]

actual = train['itemId']

import numpy as np

actual = np.array(actual).reshape(193882,1)

import ml_metrics

score = ml_metrics.mapk(actual, predicted, 5)

print('Mean avg. precision at k for train set:','{:.8f}'.format(score))
import csv

fields = 'user_id', 'item_list'

filename = 'rec_cros.csv'

with open(filename, 'a', newline = '') as f:

    writer = csv.writer(f)

    writer.writerow(fields)

    userId = cros_val['userId'].values.tolist()

    for user in userId:

        scores = []

        items = []

        results = []

        results.append(user)

        recommendations = model.recommend(user, user_items, N = 5)

        for item in recommendations:

            ids, score = item

            scores.append(score)

            items.append(ids)

        results.append(items)

        writer.writerow(results)
predicted_c = pd.read_csv('rec_cros.csv')

predicted_c = predicted_c['item_list']

import ast 

predicted_c = [ast.literal_eval(a) for a in predicted_c]

actual_c = cros_val['itemId']

import numpy as np

actual_c = np.array(actual_c).reshape(64628,1)

import ml_metrics

score_c = ml_metrics.mapk(actual_c, predicted_c, 5)

print('Mean avg. precision at k for cros_val set:','{:.6f}'.format(score_c))
fields = 'user_id', 'item_list'

filename = 'rec_test.csv'

with open(filename, 'a', newline = '') as f:

    writer = csv.writer(f)

    writer.writerow(fields)

    userId = test['userId'].values.tolist()

    for user in userId:

        scores = []

        items = []

        results = []

        results.append(user)

        recommendations = model.recommend(user, user_items, N = 5)

        for item in recommendations:

            ids, score = item

            scores.append(score)

            items.append(ids)

        results.append(items)

        writer.writerow(results)
predicted_t = pd.read_csv('rec_test.csv')

predicted_t = predicted_t['item_list']

predicted_t = [ast.literal_eval(a) for a in predicted_t]

actual_t = test['itemId']

actual_t = np.array(actual_t).reshape(64628,1)

score_t = ml_metrics.mapk(actual_t, predicted_t, 5)

print('Mean avg. precision at k for test set:','{:.6f}'.format(score_t))
model.explain(40428, user_items, 4, N = 5)
print('List of similar items for itemId 4:')

model.similar_items(4, N = 5)
print('List of similar users for userId 40428:')

model.similar_users(40428, N = 5)