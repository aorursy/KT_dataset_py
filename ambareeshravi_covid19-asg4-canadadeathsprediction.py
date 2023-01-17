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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from keras.layers import *

from keras.models import Model, Sequential, load_model
def load_data(path = "/kaggle/input/ece657aw20asg4coronavirus/"):

    '''

    Loads the COVID data as DataFrames

    '''

    confirmed = pd.read_csv(path + "time_series_covid19_confirmed_global.csv")

    dead = pd.read_csv(path + "time_series_covid19_deaths_global.csv")

    recovered = pd.read_csv(path + "time_series_covid19_recovered_global.csv")

    return confirmed, dead, recovered
confirmed, dead, recovered = load_data()
def plot_data(df, title):

    '''

    Plots the COVID cases given a data frame

    '''

    cases_dict = dict()

    plt.clf()

    plt.figure(figsize=(25,5))

    

    # Line plot showing cases over time

    for idx,  row in df.iterrows():

        plt.plot(row[1:], label = row[0])

        cases_dict[row[0]] = sum(row[1:])

    plt.xticks(rotation='vertical')

    plt.legend()

    plt.title(title)

    plt.show()

    

    # Pie chart showing cases per region

    if len(cases_dict) > 1:

        plt.clf()

        plt.figure(figsize=(10,10))

        regions = list(cases_dict.keys())

        num_cases = np.array(list(cases_dict.values()))

        amount_cases =  num_cases/ sum(num_cases)

        plt.pie(amount_cases, labels=regions, autopct='%.1f%%')

        plt.show()
# Take only canada

canada_confirmed = confirmed.where(confirmed["Country/Region"] == "Canada").dropna()

canada_dead = dead.where(dead["Country/Region"] == "Canada").dropna()

canada_recovered = recovered.where(recovered["Country/Region"] == "Canada").dropna()
# Remove redundant columns

columns = list(canada_confirmed.columns)

for col in ["Country/Region", "Lat", "Long"]:

    columns.remove(col)
# PLot the cases - confirmed, deceased, recovered

for df, title in zip([canada_confirmed, canada_dead, canada_recovered], ["Confirmed", "Dead", "Recovered"]):

    plot_data(df[columns], title = "Region wise " + title + " cases in Canada")
data_dict = dict([(row[0], np.array(row[1:])) for idx, row in canada_dead[columns].iterrows()])
data = np.array(list(data_dict.values()))
# Normalize data

ss = StandardScaler()

data_scaled = ss.fit_transform(data.T).T
# Define model for predicting

def get_LSTM_model(num_features = 15, toLook = 4):

    # toLook - how many time steps to learn

    # num_features - number of regions to predict for

    input_layer = Input(shape=(num_features, toLook))

    lstm1 = LSTM(16)(input_layer)

    out = Dense(15)(lstm1)

    model = Model(inputs = input_layer, outputs = out)

    return model
def create_training_data(data, toLook = 4):

    '''

    Creates the training data according to how many time steps to learn

    '''

    X = list()

    Y = list()

    for idx in range(0, data.shape[-1]-(toLook + 1)):

        X.append(data[:, idx: (idx + toLook)])

        Y.append(data[:, idx + toLook + 1])

    return np.array(X), np.array(Y)
toLook = 7

data_X, data_y = create_training_data(data_scaled, toLook)

X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.25, random_state=42)
model = get_LSTM_model(toLook = toLook)

model.compile(loss='mean_squared_error', optimizer='sgd')

model.fit(X_train, y_train, epochs=50, batch_size=1, validation_split=0.1)
# # Check how different predicted is from actual

# for actual, predicted in zip(y_test, model.predict(X_test)):

#     plt.plot(actual, label = "actual")

#     plt.plot(predicted, label = "predicted")

#     plt.legend()

#     plt.show()
# show predictions

def show_predictions(total_actual):

    plt.figure(figsize=(25, 5))

    for region, cases_count in zip(list(data_dict.keys()), total_actual):

        plt.plot(cases_count, label = region)

    plt.legend()

    plt.show()


predictions = list()

to_predict = 15



X = data_scaled[:, -toLook:]

for step in range(to_predict):

    X = np.expand_dims(X, axis = 0)

    pred = np.squeeze(model.predict(X))

    predictions.append(pred.T)

    X = np.hstack([np.squeeze(X)[:, -(toLook-1):], pred.reshape(15,1)])

    

predictions = np.array(predictions).T
# Combine actual and predictions

total = np.hstack((data_scaled, predictions))
# Convert back to actual numbers from normalized form

total_actual = ss.inverse_transform(total.T).T
show_predictions(total_actual)