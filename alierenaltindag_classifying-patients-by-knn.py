import numpy as np # Linear algebra

import pandas as pd # Data processing.

import matplotlib.pyplot as plt # Visualize
data = pd.read_csv("../input/column_2C_weka.csv") # Import Data

print(data["class"].unique())

data.head()
abnormal = data[data["class"] == "Abnormal"]

normal = data[data["class"] == "Normal"]



plt.scatter(abnormal.pelvic_radius, abnormal.sacral_slope,color = "red",label = "abnormal")

plt.scatter(normal.pelvic_radius, normal.sacral_slope,color = "green",label = "normal")

plt.legend()

plt.xlabel("pelvic_radius")

plt.ylabel("sacral_slope")

plt.show()
data["class"] = [1 if each == "Abnormal" else 0 for each in data["class"]] # We need 1 and 0

data.head()
y = data["class"].values

x_data = data.drop(["class"],axis = 1)

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)) # Normalize

x.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1, random_state = 15) # 85% train, 15% test
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 2) # Optimal n_neighbors value is 2

knn.fit(x_train,y_train)

prediction = knn.predict(x_test) # We predicted x_test values

print("KNN score:", knn.score(x_test,y_test))
scores = []

for each in range(1,len(x_train)):

    knn2 = KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score = knn2.score(x_test,y_test) # Calculate R-Square

    scores.append(score)



plt.plot(range(1,len(x_train)), scores)

plt.xlabel("k values")

plt.ylabel("scores")

plt.show()

print("Maximum Score :", max(scores))

print("K value :", scores.index(max(scores))+1)

# Here, We write max(scores)+1 because normally counting starts from 0 in software but scores list is starting with 1

# As you can see, optimal k value is 2
import seaborn as sns

import matplotlib.pyplot as plt



y_predict = knn.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_predict) # We use confusion matrix for comparison



f, ax = plt.subplots(figsize = (5,5))

sns.heatmap(cm,annot = True, linewidths = 0.5, linecolor = "red", fmt = ".0f",ax = ax)

plt.xlabel("predicted")

plt.ylabel("real")

plt.show()