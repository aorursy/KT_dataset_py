import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

from sklearn.neighbors import NearestNeighbors

from sklearn.cross_validation import train_test_split
df = pd.read_excel("../input/football_players.xlsx", sheetname=3)

df.head()
X = df.iloc[:,1:]

y = df.iloc[:,0]
X.info()
sorted(y)
def predict_best_replacements(args, k=4, norm=2):

    knn = NearestNeighbors(n_neighbors=k, p=norm)

    knn.fit(X)

    for player in args:

        replacements = []

        ind = y[y == player].index[0]

        neighbors = knn.kneighbors(X.iloc[ind,:].values.reshape(1, -1))[1][0]

        for n in neighbors:

            replacements.append(y[n])

        print('Player: ', player, ', Replacements: ', ", ".join(replacements[1:]))
predict_best_replacements(['Lionel Messi'],5)