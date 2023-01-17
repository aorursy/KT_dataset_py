import pandas as pd
import numpy as np

data = pd.read_csv("/kaggle/input/iris/Iris.csv")
def label(label):
        if label == 'Iris-setosa': 
            return "1"
        if label == 'Iris-versicolor': 
            return "2"
        if label == 'Iris-virginica': 
            return "3"
        if label == "1": 
            return 'Iris-setosa'
        if label == "2": 
            return 'Iris-versicolor'
        if label == "3": 
            return 'Iris-virginica'
        
y = np.array(data.drop("Species", axis=1)).reshape(150, 5)
X = np.array(list(map(lambda x: label(x), data["Species"])))
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
knn, score, n = knn(y, X)

# Row to Sample
row = y[38]
target = np.array(row)
result = knn.predict(target.reshape(1, -1))[0]
import matplotlib.pyplot as plt

colors = np.array(list(map(lambda x: int(x) * 0.250, X)))
plt.scatter(data["PetalLengthCm"] / data["PetalWidthCm"], data["SepalLengthCm"] / data["SepalWidthCm"], c=colors)
plt.xlabel('Sepal Width vs. Sepal Length')
plt.ylabel('Petal Width vs. Petal Length')
plt.title("K-Nearest Neightbors Plot")
plt.draw()
percent = int(score * 100)
print("Neightbors:", n, " Accuracy:", percent, "% ", label(result))