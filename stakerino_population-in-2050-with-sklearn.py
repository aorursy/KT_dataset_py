# I import all the classes I need

from sklearn.linear_model import LinearRegression

import pandas as pd # I use pandas for import csv

from sklearn.model_selection import train_test_split # I'll take a look at my model

from sklearn.metrics import mean_absolute_error # I use mean_absolute_error cause it's a regression

import matplotlib.pyplot as plt # In order to take a look at my model and real values

import numpy as np
# Import datas, use the name as index, it's datas you can find here : https://donnees.banquemondiale.org/indicateur/sp.pop.totl

# I use first column as index, to find world easier

world_population = pd.read_csv("../input/population_mondiale.csv", sep=";", index_col=0).loc["Monde"][4:-2].astype(int)

# I slice because first values aren't interesting and last one is a problem of columns formating in original file

y = list(world_population) # output values

X = world_population.index.astype(int).values.reshape(-1, 1) # input

train_X, val_X, train_y, val_y = train_test_split(X, y) # to create my model and look at his accuracy after



model = LinearRegression().fit(train_X, train_y)
error = mean_absolute_error(model.predict(val_X), val_y)
error
plt.plot(X, y, label="real values")

plt.plot(X, model.coef_*X+model.intercept_, label="linear regression")

plt.legend()
model.predict(np.array(2050).reshape(-1,1)).astype(int)