import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import matplotlib.pyplot as plt


def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
def evaluate(train, test, k):
    results = []
    for i in range(len(test)):
        distances = []
        for j in range(len(train)):
                distances.append((distance(test[i], train[j]), train[j][2]))
        distances.sort()
        results.append(1 if sum(list(zip(*distances[0:k]))[1]) > k/2 else 0)
    return results
        
    
def get_accuracy(test, results):
    correct = 0
    for i in range(len(test)):
        if test[i][2] == results[i]:
            correct += 1
    accuracy = correct/len(test)
    return accuracy

    
train = pd.read_csv("../input/train.csv")
train = train.values.tolist()
x, y, c = list(zip(*train))

c = ['red' if value == 1 else 'blue' for value in c]

plt.scatter(x, y, c=c, s=4)
plt.show()
test = pd.read_csv("../input/test.csv")
test = test.values.tolist()

x, y, c = list(zip(*test))

c = ['red' if value == 1 else 'blue' for value in c]

plt.scatter(x, y, c=c, s=4)
plt.show()
k = 1
results = evaluate(train, test, k)    

print(get_accuracy(test, results))
c = ['red' if value == 1 else 'blue' for value in results]

plt.scatter(x, y, c=c, s=4)
plt.show()
