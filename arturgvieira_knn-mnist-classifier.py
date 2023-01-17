import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X = np.array(train.drop("label", axis=1)).reshape(42000, 784)
y = np.array(train["label"])
data = np.array(test).reshape(28000, 784)
# # graph plotting commands
lowRowRange = 19500
highRowRange = 22500
testRow = 11246
plt.imshow(data[testRow].reshape(28, 28), cmap='Greys')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def knn(data, target):
    (training_data, test_data, training_labels, test_labels) = train_test_split(
        data, target, test_size=0.2, random_state=100)
    
    n = 3
    classifier = KNeighborsClassifier(n_neighbors=n)
    classifier.fit(training_data, training_labels)
    score = classifier.score(test_data, test_labels)
    for k in range(1, 40):
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(training_data, training_labels)
        result = classifier.score(test_data, test_labels)
        if score < result:
            score = result
            n = k
    
    classifier = KNeighborsClassifier(n_neighbors=n)
    classifier.fit(training_data, training_labels)
    score = classifier.score(test_data, test_labels)
    return (classifier, score, n)
(knn, score, n) = knn(X[lowRowRange:highRowRange], y[lowRowRange:highRowRange])
result = knn.predict(data[testRow].reshape(1, 784))[0]
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def kmc():
    sc = StandardScaler()
    sc.fit(X[lowRowRange:highRowRange])
    dataset = sc.transform(X[lowRowRange:highRowRange])
    model = KMeans(n_clusters=3, random_state=10)
    model.fit(dataset)
    score = model.score(dataset)
    return (model, model.labels_)

(model, labels) = kmc()
percent = int(score * 100)
print("Neightbors:", n, " Accuracy:", percent, "% ", "Result: ", result)
# graph plotting commands
plt.bar(0, 0)
plt.xticks(ticks = range(10))
plt.bar(sorted(set(labels)), 0)
plt.bar(result, int(score * 100))