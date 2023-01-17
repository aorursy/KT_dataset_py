import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
data = pd.read_csv("../input/breastCancer.csv")
data.tail()
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
data.head()
M = data[data.diagnosis == "M"]

B = data[data.diagnosis == "B"]
plt.scatter(M.radius_mean, M.texture_mean, color="red", label="kotu",alpha=.29)

plt.scatter(B.radius_mean, B.texture_mean, color="green", label = "iyi", alpha=.29)

plt.xlabel("radius_mean")

plt.ylabel("texture_mean")

plt.legend()

plt.show()
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

y = data.diagnosis.values

x_data = data.drop(["diagnosis"],axis=1)
y
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.3, random_state=1)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)
prediction
print("{} - nn score: {}".format(3, knn.score(x_test,y_test)))
score_list = []

for each in range(1,15):

    knns = KNeighborsClassifier(n_neighbors = each)

    knns.fit(x_train,y_train)

    score_list.append(knns.score(x_test,y_test))



plt.plot(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()