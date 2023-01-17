import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
dir = '/kaggle/input/league-of-legends-challenger-ranked-games2020/'

ranks = ['Challenger', 'GrandMaster', 'Master']



data = pd.DataFrame()



for rank in ranks:

    data_in = pd.read_csv(dir+rank+'_Ranked_Games.csv')

    data_in['Rank'] = rank

    print("Size of {}: {}".format(rank,data_in.shape))

    data = data.append(data_in, ignore_index=True)

    

print("Total size: {}".format(data.shape))
data.head()
data = data.drop('redWins', axis=1)

data = data.drop('redFirstBlood', axis=1)

data = data.drop('redFirstTower', axis=1)

data = data.drop('gameId', axis=1)
data.head()
data = data.sample(frac=1).reset_index(drop=True)
data.head()
y_data = data['blueWins']

x_data = data.drop('blueWins', axis=1)

y_data.head()
# x_data = x_data.drop('Rank', axis=1)
x_data = pd.get_dummies(x_data)

x_data.head()
x_train = x_data[:180000]

y_train = y_data[:180000]

print(x_train.shape)

print(y_train.shape)
x_test = x_data[180000:]

y_test = y_data[180000:]

print(x_test.shape)

print(y_test.shape)