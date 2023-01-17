# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import time

from datetime import datetime

from scipy import integrate, optimize

import warnings

warnings.filterwarnings('ignore')



# ML libraries

import lightgbm as lgb

import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn import linear_model

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_log_error

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler



from tensorflow.python.ops import variables

from tensorflow.python.framework import ops

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()



from fancyimpute import KNN 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load Data



# Train is 22338 by 6. Each column is Id, Province_State, Country_Region, Date, ConfirmedCases, Fatalities

# Test is 13158 by 4 ForecastId, Province_State, Country_Region, Date

test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")



# Separate by date

train_df = train.query("Date<='2020-03-25'")

valid_df = train.query("Date>'2020-03-25'")



# Convert dates to integers, starting from 0

test["Date"] = (pd.to_datetime(test['Date']) - pd.to_datetime(min(train['Date']))).dt.days

valid_df["Date"] = (pd.to_datetime(valid_df['Date']) - pd.to_datetime(min(train['Date']))).dt.days

train_df["Date"] = (pd.to_datetime(train_df['Date']) - pd.to_datetime(min(train['Date']))).dt.days



train_data = train_df.to_numpy()

test_data = test.to_numpy()

valid_data = valid_df.to_numpy()
# Helper functions for managing the data



def get_place(row):

    place = row[2]

    if isinstance(row[1], str):

        place = row[1]

    return place



# Returns a dictionary, keyed by places, of their data

def separate_by_place(data):

    place_data = {}

    for row in data:

        place = get_place(row)

        if place in place_data:

            place_data[place].append(row)

        else:

            place_data[place] = [row]

    return place_data
# Calculate error



def rmsle(y_test, predictions):

    return np.sqrt(mean_squared_log_error(y_test, predictions))
# Linear Regression Test, this is the baseline

# Optimal performance appears to be at 5th order death, 4th order case

# Performance actually worsens if 0's are removed



def linear_reg(order, train, valid):

    # Split by place

    train_place = separate_by_place(train)

    valid_place = separate_by_place(valid)

    

    case_predictions = []

    case_actual = []

    death_predictions = []

    death_actual = []



    poly = PolynomialFeatures(order)

    

    for place in train_place.keys():

        train = np.asarray(train_place[place])

        valid = np.asarray(valid_place[place])

        

        # Remove days where there were no cases reported yet

        train_del = np.delete(train, np.where(train[:, 4] == 0), axis=0)

        if len(train_del) > 0:

            train = train_del



        days = poly.fit_transform(train[:, [3]])

        cases = train[:, [4, 5]]

        days_predict = poly.fit_transform(valid[:, [3]])

        cases_predict = valid[:, [4, 5]]



        d_reg = LinearRegression().fit(days, cases[:, [1]])

        death_predictions.extend(d_reg.predict(days_predict).flatten())

        death_actual.extend(cases_predict[:, [1]])



        c_reg = LinearRegression().fit(days, cases[:, [0]])

        case_predictions.extend(c_reg.predict(days_predict).flatten())

        case_actual.extend(cases_predict[:, [0]])



    # Remove negatives

    case_predictions = np.asarray(case_predictions)

    case_predictions = np.where(case_predictions < 0, 0, case_predictions)

    death_predictions = np.asarray(death_predictions)

    death_predictions = np.where(death_predictions < 0, 0, death_predictions)



    c_error = rmsle(case_actual, case_predictions)

    d_error = rmsle(death_actual, death_predictions)

    print("Death error:", d_error, "\tCase error:", c_error)

    

for i in range(10):    

    print("Order:", i)

    linear_reg(i, train_data, valid_data)
# Now try a neural network

# The goal of the network is to compute the factor to multiply previous day by

NUM_FEATURES = 92



tf.reset_default_graph()

graph = tf.Graph()



learning_rate = 0.001



with graph.as_default():

    X = tf.placeholder(tf.float32, (None, NUM_FEATURES))

    labels = tf.placeholder(tf.float32, (None, 2))

    with tf.name_scope("fcn1"):

        W1 = tf.get_variable('W1', shape=(NUM_FEATURES, 500), initializer=tf.keras.initializers.glorot_normal())

        b1 = tf.Variable(tf.zeros((500,)), trainable=True)

        X1 = tf.add(tf.matmul(X, W1), b1)

        X1 = tf.layers.batch_normalization(X1)

        X1 = tf.nn.leaky_relu(X1)

    with tf.name_scope("fcn2"):

        W2 = tf.get_variable('W2', shape=(500, 500), initializer=tf.keras.initializers.glorot_normal())

        b2 = tf.Variable(tf.zeros((500,)), trainable=True)

        X2 = tf.add(tf.matmul(X1, W2), b2)

        X2 = tf.layers.batch_normalization(X2)

        X2 = tf.nn.leaky_relu(X2)

    with tf.name_scope("fcn3"):

        W3 = tf.get_variable('W3', shape=(500, 100), initializer=tf.keras.initializers.glorot_normal())

        b3 = tf.Variable(tf.zeros((100,)), trainable=True)

        X3 = tf.add(tf.matmul(X2, W3), b3)

        X3 = tf.layers.batch_normalization(X3)

        X3 = tf.nn.leaky_relu(X3)

    with tf.name_scope("fcn4"):

        W4 = tf.get_variable('W4', shape=(100, 10), initializer=tf.keras.initializers.glorot_normal())

        b4 = tf.Variable(tf.zeros((10,)), trainable=True)

        X4 = tf.add(tf.matmul(X3, W4), b4)

        X4 = tf.layers.batch_normalization(X4)

        X4 = tf.nn.leaky_relu(X4)

    with tf.name_scope("fcn5"):

        W5 = tf.get_variable('W5', shape=(10, 2), initializer=tf.keras.initializers.glorot_normal())

        b5 = tf.Variable(tf.zeros((2,)), trainable=True)

        predictions = tf.add(tf.matmul(X4, W5), b5)

    loss = tf.losses.mean_squared_error(labels, predictions)



    optimizer = tf.train.AdagradOptimizer(learning_rate)

    train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
def make_nn_train_data(country_data):

    # Sort by date

    country_data = np.asarray(country_data)

    indices = country_data[:, 3].argsort()

    country_data = country_data[indices]

    

    # Remove rows without new cases

#     indices = [0]

#     for i in range(len(country_data) - 1):

#         train_y = country_data[i+1, [4, 5]] - country_data[i, [4, 5]]

#         if np.sum(train_y) == 0.0:

#             continue

#         indices.append(i)

#     country_data = country_data[indices]

    

    # Add two columns: one each for previous day total cases and deaths

    added_data = np.c_[country_data, np.zeros(len(country_data))]

    added_data = np.c_[added_data, np.zeros(len(country_data))]

    for i in range(1, len(country_data)):

        added_data[i, [-2, -1]] = country_data[i-1, [4, 5]]

    x_indices = [3] + [i for i in range(6, len(added_data[0]))]

    train_x = added_data[:, x_indices]

    

    # Y data is growth from previous day

    train_y = np.zeros((len(country_data), 2))

    for i in range(len(country_data) - 1):

        train_y[i] = country_data[i+1, [4, 5]] - country_data[i, [4, 5]]

    train_y[len(country_data) - 1] = train_y[len(country_data) - 2]

    

    return train_x, train_y
def debug_grads(sess, feed_dict):

    var_list = (variables.trainable_variables() + ops.get_collection(

        ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))

    print('variables')

    for v in var_list:

        print('  ', v.name)

    # get all gradients

    grads_and_vars = optimizer.compute_gradients(loss)

    # train_op = optimizer.apply_gradients(grads_and_vars)



    zipped_val = sess.run(grads_and_vars, feed_dict=feed_dict)



    for rsl, tensor in zip(zipped_val, grads_and_vars):

        print('-----------------------------------------')

        print('name', tensor[0].name.replace('/tuple/control_dependency_1:0', '').replace('gradients/', ''))

        print('gradient', rsl[0])

        print('value', rsl[1])
# First, add some data (first WDI obtained here https://www.kaggle.com/sambitmukherjee/covid-19-data-adding-world-development-indicators/output)

train_wdi = pd.read_csv("../input/combine-wdi-covid/train_with_WDI.csv")

test_wdi = pd.read_csv("../input/combine-wdi-covid/test_with_WDI.csv")

test_norm = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")



# Add personality info

personality_info = pd.read_csv("../input/covid19-country-data-wk3-release/Data Join - RELEASE.csv")

personality_info = personality_info.rename(columns={"TRUE POPULATION": "TRUE_POPULATION"})

personality_info.pct_in_largest_city = personality_info.pct_in_largest_city.apply(lambda x: x.replace('%', ''))

personality_info.TRUE_POPULATION = personality_info.TRUE_POPULATION.apply(lambda x: x.replace(',', ''))

train_wdi = pd.merge(train_wdi, personality_info,  how='left', on=['Province_State','Country_Region'])



# Add leader info https://www.kaggle.com/lunatics/global-politcs-and-governance-data-apr-2020

leader_info = pd.read_csv("../input/politics/politics_apr2020.csv")

train_wdi = pd.merge(train_wdi, leader_info,  how='left', on=['Country_Region'])



# Add immunization coverage https://www.kaggle.com/lsind18/who-immunization-coverage

for filename in os.listdir("../input/who-immunization-coverage"):

    immun_info = pd.read_csv("../input/who-immunization-coverage/" + filename).iloc[:,0:2]

    immun_info = immun_info.rename(columns={"Country": "Country_Region", "2018": filename})

    train_wdi = pd.merge(train_wdi, immun_info,  how='left', on=['Country_Region'])



# Replace bad data with nan

train_wdi = train_wdi.apply(lambda x: x.replace('#NULL!', np.nan))

train_wdi = train_wdi.apply(lambda x: x.replace('#DIV/0!', np.nan))

train_wdi = train_wdi.apply(lambda x: x.replace('#N/A', np.nan))

train_wdi = train_wdi.apply(lambda x: x.replace('N.A.', np.nan))



# Separate by date 

valid_wdi = train_wdi.query("Date>'2020-04-03'")

# Remove for public leaderboard

# train_wdi = train_wdi.query("Date<='2020-04-03'")



# Convert dates to integers, starting from 0

test_wdi["Date"] = (pd.to_datetime(test_wdi['Date']) - pd.to_datetime(min(train_wdi['Date']))).dt.days

valid_wdi["Date"] = (pd.to_datetime(valid_wdi['Date']) - pd.to_datetime(min(train_wdi['Date']))).dt.days

test_norm["Date"] = (pd.to_datetime(test_norm['Date']) - pd.to_datetime(min(train_wdi['Date']))).dt.days

train_wdi["Date"] = (pd.to_datetime(train_wdi['Date']) - pd.to_datetime(min(train_wdi['Date']))).dt.days



train_wdi = train_wdi.to_numpy()

test_wdi = test_wdi.to_numpy()

valid_wdi = valid_wdi.to_numpy()

test_norm = test_norm.to_numpy()



# Cast to float

indices = [i for i in range(3, len(train_wdi[0]))]

train_wdi[:, indices] = train_wdi[:, indices].astype('float64') 



# Apply k nearest neighbors to obtain data for nan

train_wdi[:, indices] = KNN(k=10).fit_transform(train_wdi[:, indices])



train_place = separate_by_place(train_wdi)

valid_place = separate_by_place(valid_wdi)



train_x = {}

train_y = {}

valid_x, valid_y = make_nn_train_data(valid_wdi)

for place in train_place.keys():

    train_x[place], train_y[place] = make_nn_train_data(train_place[place])

    

# Normalize the data

all_x = -1

all_y = -1

# Train set

for place in train_x.keys():

    try:

        all_x = np.vstack([all_x, train_x[place]])

        all_y = np.vstack([all_y, train_y[place]])

    except:

        all_x = train_x[place]

        all_y = train_y[place]
NUM_FEATURES = all_x.shape[1]
NUM_EPOCHS = 5000

DEBUG = False

learning_rate = 0.007

days_to_extend = 60

restore = False

save = False



# SKLearn scalers

x_scaler = StandardScaler()

x_scaler.fit(all_x)

y_scaler = StandardScaler()

y_scaler.fit(all_y)



with tf.Session(graph=graph) as sess:

    if restore:

        saver.restore(sess, "tmp/model.ckpt")

        NUM_EPOCHS = 0

    else:

        init.run()

    for epoch in range(NUM_EPOCHS):

        avg_loss = 0

        # Train

        for place in train_x.keys():

            batch_x = train_x[place]

            batch_y = train_y[place]

            standardized_x = x_scaler.transform(batch_x)

            standardized_y = y_scaler.transform(batch_y)

            feed_dict = {X: standardized_x, labels: standardized_y}

            if DEBUG:

                debug_grads(sess, feed_dict)

            _, loss_val, outs = sess.run([train_op, loss, predictions], feed_dict=feed_dict)

            avg_loss += loss_val

        print(epoch, "Total Loss", avg_loss)

            

        # Test on CV set

#         standardized_x = x_scaler.transform(valid_x)

#         standardized_y = y_scaler.transform(valid_y)

#         feed_dict = {X: standardized_x, labels: standardized_y}

#         loss_validation = sess.run(loss, feed_dict=feed_dict)

        

    # Make the predictions

    for place in train_x.keys():

        for day in range(days_to_extend):

            # Predict change from last day

            old_row = [train_x[place][-1]]

            standardized_x = x_scaler.transform(old_row)

            feed_dict = {X: standardized_x}

            outs = sess.run(predictions, feed_dict=feed_dict)

            preds = y_scaler.inverse_transform(outs)

            # Insert new row

            # 3 (date) is index 0, last two indices are total cases as of yesterday

            new_row = np.zeros(old_row[0].shape) + old_row[0]

            new_row[0] += 1

            # Check for 0

            if preds[0][0] >= 0:

                new_row[-2] += preds[0][0]

            if preds[0][1] >= 0:

                new_row[-1] += preds[0][1]

            train_x[place] = np.vstack([train_x[place], new_row])

            

    # Save

    if save:

        save_path = saver.save(sess, "tmp/model.ckpt")
# Make predictions

my_columns = ["ForecastId", "ConfirmedCases", "Fatalities"]

predictions = []

for row in test_norm:

    # ForecastId, cases, mortality

    place = get_place(row)

    date = row[3]

    new_row = [row[0], train_x[place][date][-2], train_x[place][date][-1]]

    predictions.append(new_row)

df = pd.DataFrame(predictions, columns=my_columns) 

df.to_csv('submission.csv', index=False)