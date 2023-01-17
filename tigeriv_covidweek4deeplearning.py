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

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn import linear_model

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_log_error

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold



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



def rmsle(y_test, predictions):

    return np.sqrt(mean_squared_log_error(y_test, predictions))



# Get dict of [cases_dom, deaths_dom, cases_int, deaths_int] for each country

def current_day_info(dataset, day):

    day_data = {}

    indices = np.where(dataset[:, 3] == day)

    total_cases = np.sum(dataset[indices, 4])

    total_deaths = np.sum(dataset[indices, 5])

    for row in dataset[indices]:

        place = get_place(row)

        day_data[place] = [row[4], row[5], total_cases, total_deaths]

    return day_data



# Add previous day total cases, deaths domestically and internationally

def make_nn_train_data(dataset):

    # Create the columns to fill

    added_data = np.c_[dataset, np.zeros(len(dataset))]

    added_data = np.c_[added_data, np.zeros(len(dataset))]

    added_data = np.c_[added_data, np.zeros(len(dataset))]

    added_data = np.c_[added_data, np.zeros(len(dataset))]

    

    # For each day in data set, and each country, grab [cases_dom, deaths_dom, cases_int, deaths_int]

    data_day_place = []

    num_days = np.amax(dataset[:, 3])

    for day in range(int(num_days)):

        data_day_place.append(current_day_info(dataset, day))

        

    # Now insert into the dataset

    for index in range(len(added_data)):

        row = dataset[index]

        place = get_place(row)

        prev_day = int(row[3] - 1)

        if prev_day >= 0:

            added_data[index, [-4, -3, -2, -1]] = data_day_place[prev_day][place]

        else:

            added_data[index, [-4, -3, -2, -1]] = [0.0, 0.0, 0.0, 0.0]

            

    x_indices = [3] + [i for i in range(6, len(added_data[0]))]

    y_indices = [4, 5]

    id_indices = [0, 1, 2]

    

    train_x = added_data[:, x_indices]

    train_y = added_data[:, y_indices]

    train_id = added_data[:, id_indices]

    

    # Change y to be delta cases and deaths. Spots -4, -3 of x are already domestic cases, deaths previously.

    # And y is how many occur by the end of the day, so take difference

    train_y = train_y - train_x[:, [-4, -3]]

    

    return train_x, train_y, train_id
# Load Data

train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

test_norm = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")



# First, add some data (first WDI obtained here https://www.kaggle.com/sambitmukherjee/covid-19-data-adding-world-development-indicators/output)

wdi_info = pd.read_csv("../input/wdi-data-covid19/wdi_data.csv")

train_wdi = pd.merge(train, wdi_info,  how='left', on=['Province_State','Country_Region'])



# Now add health system data

health_info = pd.read_csv("../input/world-bank-wdi-212-health-systems/2.12_Health_systems.csv")

train_wdi = pd.merge(train_wdi, health_info,  how='left', on=['Province_State','Country_Region'])



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

train_wdi = train_wdi.drop(['World_Bank_Name'], axis=1)



# Convert dates to integers, starting from 0

test_norm["Date"] = (pd.to_datetime(test_norm['Date']) - pd.to_datetime(min(train_wdi['Date']))).dt.days

train_wdi["Date"] = (pd.to_datetime(train_wdi['Date']) - pd.to_datetime(min(train_wdi['Date']))).dt.days

train["Date"] = (pd.to_datetime(train['Date']) - pd.to_datetime(min(train['Date']))).dt.days



train = train.to_numpy()

train_wdi = train_wdi.to_numpy()

test_norm = test_norm.to_numpy()



# Cast to float

indices = [i for i in range(3, len(train_wdi[0]))]

train_wdi[:, indices] = train_wdi[:, indices].astype('float64') 
# Apply k nearest neighbors to obtain data for nan

train_wdi[:, indices] = KNN(k=5).fit_transform(train_wdi[:, indices])
# Create training sets

train_x, train_y, train_info = make_nn_train_data(train_wdi)
# Helper function to view gradients for debugging purposes

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
# Create graph



def fcn(num_in, num_out, X, name):

    with tf.name_scope(name):

        W = tf.get_variable(name + 'W', shape=(num_in, num_out), initializer=tf.keras.initializers.glorot_normal())

        b = tf.Variable(tf.zeros((num_out,)), trainable=True)

        X = tf.add(tf.matmul(X, W), b)

        X = tf.layers.batch_normalization(X)

        X = tf.nn.leaky_relu(X)

    return X



tf.reset_default_graph()

graph = tf.Graph()



learning_rate = 0.007

NUM_FEATURES = train_x.shape[1]

DEPTH = 20



with graph.as_default():

    X = tf.placeholder(tf.float32, (None, NUM_FEATURES))

    labels = tf.placeholder(tf.float32, (None, 2))

    X_es = [X]

    for i in range(DEPTH):

        num_in = 150

        num_out = 150

        if i == 0:

            num_in = NUM_FEATURES

        if i == DEPTH - 1:

            num_out = 2

        X_es.append(fcn(num_in, num_out, X_es[-1], "fcn" + str(i)))

    predictions = X_es[-1]

    loss = tf.losses.huber_loss(labels, predictions)



    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

# Train



NUM_EPOCHS = 500

save_freq = 100

DEBUG = False

learning_rate = 0.007

restore = False

save = True

load_path = "../input/pretrained-covid19/model.ckpt"

batch_size = 16

test_size = 0.01



# SKLearn scalers

x_scaler = StandardScaler()

x_scaler.fit(train_x)

y_scaler = StandardScaler()

y_scaler.fit(train_y)

transformed_x = x_scaler.transform(train_x)

transformed_y = y_scaler.transform(train_y)



with tf.Session(graph=graph) as sess:

    if restore:

        saver.restore(sess, load_path)

        NUM_EPOCHS = 0

    else:

        init.run()

        

    indices = [i for i in range(len(train_x))]

    for epoch in range(NUM_EPOCHS):

        # Shuffle and split data set

        np.random.shuffle(indices)

        split_index = int((1 - test_size) * (len(indices)))

        train_indices = indices[:split_index]

        cv_indices = indices[split_index:]

        # Initialize some variables

        avg_loss = 0

        batch_index = 0



        # Mini batches

        while batch_index < len(train_indices) - batch_size:

            batch_indices = train_indices[batch_index: batch_index + batch_size]

            batch_x = transformed_x[batch_indices]

            batch_y = transformed_y[batch_indices]

            batch_index += batch_size



            if DEBUG:

                debug_grads(sess, feed_dict)

                

            feed_dict = {X: batch_x, labels: batch_y}

            _, loss_val, outs = sess.run([train_op, loss, predictions], feed_dict=feed_dict)

            avg_loss += loss_val

            

        cv_x = transformed_x[cv_indices]

        cv_y = transformed_y[cv_indices]

        feed_dict = {X: cv_x, labels: cv_y}

        cv_loss = sess.run([loss], feed_dict=feed_dict)[0]

        print(epoch, "Avg Train Loss", avg_loss/batch_index, "Avg CV Loss", cv_loss/len(cv_indices))

        

        # Save

        if save and (epoch % save_freq == 0): 

            save_str = "tmp/model" + str(epoch) + ".ckpt"

            save_path = saver.save(sess, save_str)

    

    # Save final weights

    save_path = saver.save(sess, "tmp/model.ckpt")
days_to_extend = 60



def row_to_nn(row, prev_day_data):

    place = get_place(row)

    new_row = np.copy(row)

    new_row = np.append(new_row, prev_day_data[place])

    x_indices = [3] + [i for i in range(6, len(new_row))]

    return new_row[x_indices]



longer_train = np.copy(train_wdi)



with tf.Session(graph=graph) as sess:

    saver.restore(sess, "tmp/model.ckpt")

    num_days = int(np.amax(longer_train[:, 3]))

    # x_indices = [3] + [i for i in range(6, len(added_data[0]))]

    for day in range(num_days, num_days + days_to_extend):

        print(day)

        prev_day_data = current_day_info(longer_train, day)

        indices = np.where(longer_train[:, 3] == day)

        for row in longer_train[indices]:

            # turn each item into nn data format

            row_x = np.asarray([row_to_nn(row, prev_day_data)])

            # Run through NN

            standardized_x = x_scaler.transform(row_x)

            feed_dict = {X: standardized_x}

            outs = sess.run(predictions, feed_dict=feed_dict)

            inverse_outs = y_scaler.inverse_transform(outs)[0]

            # Floor at 0

            if inverse_outs[0] < 0:

                inverse_outs[0] = 0.0

            if inverse_outs[1] < 0:

                inverse_outs[1] = 0.0

            # Create new row

            new_row = np.copy(row)

            new_row[3] += 1

            new_row[4] += inverse_outs[0]

            new_row[5] += inverse_outs[1]

            longer_train = np.append(longer_train, [new_row], axis=0)
# Convert date back

train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

start_date = np.datetime64(np.min(train['Date']))

new_dates = []

for index in range(len(longer_train)):

    new_date = start_date + np.timedelta64(int(longer_train[index][3]), 'D')

    new_dates.append(new_date)



conv_predictions = np.copy(longer_train)

conv_predictions[:, 3] = new_dates



# Save predictions as a file

my_columns = ["ForecastId", "Province_State", "Country_Region", "Date", "ProjectedCases", "Fatalities"]

outputs = conv_predictions[:, [0, 1, 2, 3, 4, 5]]

df = pd.DataFrame(outputs, columns=my_columns) 

df.to_csv('predictions.csv', index=False)
# Create submission file

submission = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

submission["Date"] = pd.to_datetime(submission['Date'])

submission = pd.merge(submission, df,  how='left', on=['Province_State', 'Country_Region', 'Date'])

submission = submission[['ForecastId_x', 'ProjectedCases', 'Fatalities']]

submission = submission.rename(columns={"ForecastId_x": "ForecastId", "ProjectedCases": "ConfirmedCases"})

submission.to_csv('submission.csv', index=False)