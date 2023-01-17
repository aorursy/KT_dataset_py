# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from datetime import datetime

import time



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/hackathon/task_2-COVID-19-death_cases_per_country_after_fifth_death-till_26_June.csv')

df.head()
df["deaths_per_million_10_days_after_fifth_death"].plot.hist()

plt.show()
df["deaths_per_million_50_days_after_fifth_death"].plot.hist()

plt.show()
df["deaths_per_million_85_days_after_fifth_death"].plot.hist()

plt.show()
# Prepare a full dataframe

#num_records = 7303

data = {}

data["date_fifth_death"] = pd.date_range("23/03/2020", "16/06/2020", freq="D")



complete = pd.DataFrame(data=data)

complete = complete.set_index("date_fifth_death")

complete = complete.merge(df, left_index=True, right_index=True, how="left")

complete = complete.bfill().ffill()
complete.head()
toInspect = ["deaths_per_million_10_days_after_fifth_death", "deaths_per_million_50_days_after_fifth_death", "deaths_per_million_85_days_after_fifth_death"]

rows, cols = 3, 2

fig, ax = plt.subplots(rows, cols, figsize=(20,rows*5))



for row in range(rows):

    sns.lineplot(data=df[[toInspect[row]]], ax=ax[row][0])

    sns.lineplot(data=complete[[toInspect[row]]], ax=ax[row][1])
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
# Data Conversion Utility



def getTimeSeriesData(A, window=7):

    X, y = list(), list()

    for i in range(len(A)):

        end_ix = i + window

        if end_ix > len(A) - 1:

            break

        seq_x, seq_y = A[i:end_ix], A[end_ix]

        X.append(seq_x)

        y.append(seq_y)

    return np.array(X), np.array(y)
from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense
window = 2

num_features = 1



X, y = getTimeSeriesData(list(df["deaths_per_million_40_days_after_fifth_death"]), window=window)

print("X:", X.shape)

print("Y:", y.shape)



# We need to add one more dimension to X, i.e Num of features in 1 sample of time step. as we are doing a univariate prediction which means number of features are 1 only

X = X.reshape((X.shape[0], X.shape[1], num_features))  # For LSTM

print("-----------")

print("X:", X.shape)

print("Y:", y.shape)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("-----------")

print("X train:", X_train.shape)

print("y train:", y_train.shape)

print("X test:", X_test.shape)

print("y test:", y_test.shape)
# Define Model

model = Sequential()

model.add(LSTM(7, activation='relu', input_shape=(window, num_features)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=5, verbose=1)
plt.plot(history.history["loss"])

# plt.plot(history.history["val_loss"])

plt.title("Model Loss")

plt.ylabel('Loss')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')

plt.show()
yPred = model.predict(X_test, verbose=0)

yPred.shape = yPred.shape[0]
plt.figure(figsize=(30,5))

sns.set(rc={"lines.linewidth": 8})

sns.lineplot(x=np.arange(y_test.shape[0]), y=y_test, color="green")

sns.set(rc={"lines.linewidth": 3})

sns.lineplot(x=np.arange(y_test.shape[0]), y=yPred, color="coral")

plt.margins(x=0, y=0.5)

plt.legend(["Original", "Predicted"])
points = 50

plt.figure(figsize=(30,5))

sns.set(rc={"lines.linewidth": 8})

sns.lineplot(x=np.arange(points), y=y_test[:points], color="green")

sns.set(rc={"lines.linewidth": 3})

sns.lineplot(x=np.arange(points), y=yPred[:points], color="coral")

plt.margins(x=0, y=0.5)

plt.legend(["Original", "Predicted"])