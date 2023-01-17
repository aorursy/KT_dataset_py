import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
df = pd.read_csv("../input/Mall_Customers.csv")
df.head()
x = df[["Annual Income (k$)", "Spending Score (1-100)", "Age"]].values
plt.scatter(x[:, 0], x[:, 1])

plt.show()
sse = []

n = []

for i in range(1, 10):

    n.append(i)

    model = KMeans(n_clusters=i)

    model.fit_predict(x)

    sse.append(model.inertia_)
plt.plot(n, sse)

plt.show()
model = KMeans(n_clusters=5)

y_pred = model.fit_predict(x)
df["Target"] = y_pred

df.head()
cen = model.cluster_centers_

cen
for target in df.Target.unique():

    newdf = df[df.Target == target]

    x = newdf[["Annual Income (k$)", "Spending Score (1-100)"]].values

    plt.scatter(x[:, 0], x[:, 1])



for c in cen:

    plt.scatter(c[0], c[1], s = 130, marker="*", color = "black")

    

plt.show()