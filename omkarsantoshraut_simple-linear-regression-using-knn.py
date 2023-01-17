import pandas as pd

import numpy as np

df = pd.read_excel('../input/studentmarks/studentmarks.xlsx')

df
temp_mse = df[['MSE']]

from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

df_values = temp_mse.values

std_df = std_scaler.fit_transform(df_values)

df_normalize = pd.DataFrame(std_df)

df_normalize
x = df_normalize[[0]]

y = df['ESE']

print(x, y)
import matplotlib.pyplot as plt

plt.scatter(x, y)

plt.xlabel('MSE marks')

plt.ylabel('ESE marks')

plt.title('Graph: MSE marks VS ESE marks')

plt.show()
from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors = 2)

neigh.fit(x, y)
neigh.predict([[12]])
import math

n = len(y)

index = []

error = []

for i in range(1, 41):

    neigh = KNeighborsRegressor(n_neighbors = i)

    neigh.fit(x, y)

    y_pre = neigh.predict(x)

    rss = sum((y_pre - y)**2)

    rse = math.sqrt(rss/(n-2))

    index.append(i)

    error.append(rse)

print(index, error)
plt.plot(index, error)

plt.xlabel('Values of K')

plt.ylabel('RSE Error')

plt.title('Graph: K VS Error')

plt.show()
neigh = KNeighborsRegressor(n_neighbors = 8)

neigh.fit(x, y)
neigh.predict([[12]])