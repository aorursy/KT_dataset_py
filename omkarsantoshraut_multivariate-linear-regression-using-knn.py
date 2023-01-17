import pandas as pd

import numpy as np



df = pd.read_csv('../input/markswithattendence/markswithattendence.csv')

df
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

temp_data = df.iloc[:,:2]

df_values = temp_data.values

std_df = std_scaler.fit_transform(temp_data)

df_normalize = pd.DataFrame(std_df)

df_normalize
x = df_normalize

y = df['ESE']
import matplotlib.pyplot as plt

plt.scatter(x[0],y)

plt.xlabel('Attendance')

plt.ylabel('ESE marks')

plt.show()



plt.scatter(x[1],y)

plt.xlabel('MSE marks')

plt.ylabel('ESE marks')

plt.show()


from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors = 3)

knn.fit(x, y)
knn.predict([[78, 12]])
import math

n = len(y)

index = []

error = []

for i in range(1, 41):

    knn = KNeighborsRegressor(n_neighbors = i)

    knn.fit(x, y)

    y_pre = knn.predict(x)

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
knn = KNeighborsRegressor(n_neighbors = 3)

knn.fit(x, y)

knn.predict(x)