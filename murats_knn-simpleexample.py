import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv("../input/breastCancer.csv")
df.head()
df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
df.head()
M = df[df.diagnosis =="M"]

B = df[df.diagnosis =="B"]
plt.scatter(M.radius_mean, M.area_mean, color ="red")
plt.scatter(B.radius_mean, B.area_mean, color="green")
plt.scatter(M.radius_mean, M.area_mean, color ="red", alpha=.15, label="kotu")

plt.scatter(B.radius_mean, B.area_mean, color="green", alpha=.15, label="iyi")

plt.legend()

plt.show()
plt.scatter(M.radius_mean, M.texture_mean, color = "red", label= "kotu", alpha=.3)

plt.scatter(B.radius_mean, B.texture_mean, color = "green", label = "iyi", alpha=.3)

plt.xlabel("radius mean")

plt.ylabel("texture mean")

plt.legend()

plt.show()
df.diagnosis = [1 if each =="M" else 0 for each in df.diagnosis]
y = df.diagnosis.values

x_data = df.drop(["diagnosis"], axis=1)
x = (x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .3, random_state=1)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(x_train,y_train)
prediction = model.predict(x_test)
prediction
print("{} nn score : {}".format(3, model.score(x_test,y_test)))
score_list = []

for each in range(1,50):

    model2 = KNeighborsClassifier(n_neighbors=each)

    model2.fit(x_train,y_train)

    score_list.append(model2.score(x_test, y_test))

    print("{} NN score : {}".format(each, model2.score(x_test,y_test)))
plt.plot(range(1,50), score_list)

plt.xlabel("k values")

plt.ylabel("score metrics")

plt.legend()

plt.show()