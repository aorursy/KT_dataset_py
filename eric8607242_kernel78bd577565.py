import os

import csv



import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt





from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.metrics import mean_squared_error

from sklearn.neural_network import MLPRegressor



%matplotlib inline



TRAIN_PATH = "/kaggle/input/PM25_train.csv"

TEST_PATH = "/kaggle/input/PM25_test.csv"



SUBMISSION_PATH = "./submission.csv"
df_train_data = pd.read_csv(TRAIN_PATH, engine="python")

df_train_data.head()
df_train_data_clean = df_train_data.drop_duplicates()

df_train_data_clean = df_train_data_clean.drop(["Time"], axis=1)

df_train_data_clean.head()
df_train_data_day = df_train_data_clean.groupby(["device_id", "Date"], level=None).mean()

df_train_data_day.head()
relation_line = df_train_data_clean.groupby(["Date"], level=None).mean()

relation_line.plot.line(y=["PM2.5", "Temperature"])

relation_line.plot.line(y=["PM2.5", "Humidity"])

plt.matshow(relation_line.corr())

plt.xticks(range(len(relation_line.columns)), relation_line.columns)

plt.yticks(range(len(relation_line.columns)), relation_line.columns)

plt.colorbar()

plt.show()
df_test_data = pd.read_csv(TEST_PATH, engine="python")

df_test_data.head()
df_test_data_clean = df_test_data.drop_duplicates()

df_test_data_clean = df_test_data_clean.drop(["Time"], axis=1)

df_test_data_clean.head()
df_test_data_day = df_test_data_clean.groupby(["device_id", "Date"], level=None).mean()

df_test_data_day.head()
with open(SUBMISSION_PATH, 'w+', newline='') as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow(['device_id', 'pred_pm25'])
num = 0

for device in df_train_data_clean["device_id"].unique():

    device_data = df_train_data_day.loc[[device]].copy()

    device_data = device_data.reset_index(drop=True)

    

    for i in range(1, 6):

        ave = device_data["PM2.5"].rolling(window=i).mean()

        device_data["ave"+str(i)] = ave

        device_data["ave"+str(i)][1:] = device_data["ave"+str(i)][:-1]

    

    if num < 2:

        for i in range(1, 6):

            ave = device_data["Temperature"].rolling(window=i).mean()

            device_data["ave_t"+str(i)] = ave

            device_data["ave_t"+str(i)][1:] = device_data["ave_t"+str(i)][:-1]

         

    device_data = device_data.drop(list(range(0, 5)))

    train_Y = device_data["PM2.5"]

    

    device_data.drop(['PM2.5'], axis=1, inplace=True)

    train_X = device_data



    reg = LinearRegression().fit(train_X, train_Y)

    score = reg.score(train_X, train_Y)



    device_test_data = df_test_data_day.loc[device].copy()

    device_test_data = device_test_data.reset_index(drop=True)

    

    for i in range(1, 6):

        device_test_data["ave"+str(i)] = train_Y[-i:].mean()

        

    if num < 2:

        for i in range(1, 6):

            device_test_data["ave_t"+str(i)] = device_data["Temperature"][-i:].mean()



    y_pred = reg.predict(device_test_data)

    y_pred = y_pred.mean()



    num += 1

    

    with open(SUBMISSION_PATH, 'a', newline='') as csvfile:

      writer = csv.writer(csvfile)

      writer.writerow([device, y_pred])

    