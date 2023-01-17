import numpy as np

import pandas as pd



from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
x = pd.read_csv("/kaggle/input/dungeons-dragons/monsters.csv", usecols=["armor_class", "hit_points", "strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"]).to_numpy()

y = pd.read_csv("/kaggle/input/dungeons-dragons/monsters.csv", usecols=["challenge_rating"]).to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=30, test_size=0.3)



x_scaler = StandardScaler()

x_scaler.fit(x_train)



x_train = x_scaler.transform(x_train)

x_test = x_scaler.transform(x_test)



y_scaler = StandardScaler()

y_scaler.fit(y_train)



y_train = y_scaler.transform(y_train)

y_test = y_scaler.transform(y_test)
model = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver="lbfgs", max_iter=1000)

model.fit(x_train, y_train)
print("Accuracy: ", model.score(x_test, y_test))