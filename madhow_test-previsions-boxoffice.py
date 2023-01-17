from os import path



import pandas as pd

import numpy as np
INPUT_PATH = path.join('..', 'input')
TRAIN_PATH = path.join(INPUT_PATH, 'train.csv')
df = pd.read_csv(TRAIN_PATH)

df.head()
df.loc[df['budget'].idxmax(), ['title', 'budget']]
from sklearn.ensemble import RandomForestRegressor
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeric_features = df.select_dtypes(include=numerics)

null_indexes = numeric_features['runtime'].isnull()

x = numeric_features.loc[~null_indexes].drop('revenue', axis=1)

y = df.loc[~null_indexes, 'revenue']

x.shape, y.shape

rf = RandomForestRegressor()

rf.fit(x, y)
import matplotlib.pyplot as plt



%matplotlib inline #afficher des données graphiques quelque soit le systeme
rf = RandomForestRegressor()

rf.fit(x, y)
y_hat = rf.predict(x)

y_hat.shape
#affichage du graphe

plt.scatter(y, y_hat, c = 'r')

data = ['True values', 'predictions']

plt.xlabel('True values')

plt.ylabel('Predictions')

#plt.scatter(y, y_hat, c = 'r', label='data')
(y == y_hat).all
y.mean()
abs(y - y_hat).mean() #moyenne de plantage par film
def mean_absolute_error(y_true, y_predict):

    return abs(y_true - y_predict).mean()
mean_absolute_error(y, y_hat)
df_train = df[:2000] #dataframe train

df_test = df[2000:] #datafralme test
#test des données

df_train.shape, df_test.shape