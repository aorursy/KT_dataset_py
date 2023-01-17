from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_wine

from sklearn.metrics import adjusted_rand_score



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data = load_wine()
x_wine = data.data

y_wine = data.target
x_df = pd.DataFrame(x_wine, columns=data.feature_names)

y_df = pd.DataFrame(y_wine, columns=['target'])
x_df.info()
y_df.info()
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2)
model_errors = []

for i in range(1, 10):

    model = KMeans(n_clusters=i)

    model.fit(x_train)

    model_errors.append(model.inertia_)

plt.plot(range(1, 10), model_errors)
model = KMeans(n_clusters=3).fit(x_train)
# Accuracy with all dataset

adjusted_rand_score(y_wine, model.predict(x_wine))
# Accuracy with test data

adjusted_rand_score(y_test.to_numpy().reshape(-1,), model.predict(x_test))