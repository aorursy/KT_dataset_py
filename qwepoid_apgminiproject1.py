import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import keras

import pandas as pd

from sklearn.model_selection import train_test_split
print(os.listdir("../input"))
data = pd.read_csv("../input/Data_Training.csv")

data.head()
data['match_url'].count()
data.dropna(inplace=True)

data['match_url'].count()
x = data.drop(columns=["Output"])

x = data.drop(columns=["Output", "Unnamed: 0", "match_url"])

x.head()

    
y = data.Output

y.head()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=64)

x_train.head()
data_2019 = pd.read_csv("../input/Data_Testing_2019.csv")

data_2019.head()
data_2019.drop(columns=["Unnamed: 0"], inplace=True)

data_2019.head()
x_2019 = data_2019.drop(columns= ["Match", "Team1", "Team2"])
# Converting categorical data to categorical

num_categories = 2

y_train = keras.utils.to_categorical(y_train, num_categories)

y_test= keras.utils.to_categorical(y_test, num_categories)

# Model Building

model = keras.models.Sequential()

model.add(keras.layers.Dense(50, activation="relu", input_dim = 41))

model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(80, activation="relu"))

model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(100, activation="relu"))

model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(2, activation="softmax"))

# Compiling the model - adaDelta - Adaptive learning

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# Training and evaluating

batch_size = 50

num_epoch = 1000

model_log = model.fit(x_train, y_train, batch_size = batch_size, epochs=num_epoch, verbose=1, validation_data=(x_test, y_test))



train_score = model.evaluate(x_train, y_train, verbose=0)

test_score = model.evaluate(x_test, y_test, verbose=0)

print('Train accuracy:', train_score[1])

print('Test accuracy:', test_score[1])

# Predictions for the 2019 World Cup

prediction_2019 = model.predict_classes(x_2019)

data_2019["Result"] = prediction_2019
def winner(x):

    if x.Result == 1:

        x["Winning_Team"] = x.Team1

    else:

        x["Winning_Team"] = x.Team2

    return x



data_2019_final = data_2019.apply(winner, axis= 1)

results_2019 = data_2019_final.groupby("Winning_Team").size()

results_2019 = results_2019.sort_values(ascending=False)

print(results_2019)