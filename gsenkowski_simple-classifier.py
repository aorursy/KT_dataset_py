# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import copy
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/kiwhs-comp-1-complete/"))

# Any results you write to the current directory are saved as output.

# Copied from competition page
def read_data(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() # get rid of newline at the end of each input line
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data

train = read_data("../input/int18whs-classify-simple-data/train.arff")
print('number of training samples', len(train))
def distance_squared(point1, point2):
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
def force_on_point(bodys, point):
    force_sum = 0
    for body in bodys:
        if body[0] != point[0] or body[1] != point[1]:
            force_sum += 1/distance_squared(body, point)
        else:
            force_sum += 10000
    return force_sum
def evaluate(train, test):
    results = []
    red = [[sample[0], sample[1]] for sample in train if sample[2] == 1]
    blue = [[sample[0], sample[1]] for sample in train if sample[2] == -1]

    for i in range(len(test)):
        results.append(1 if force_on_point(red, test[i]) > force_on_point(blue, test[i]) else -1)
    return results
def get_accuracy(test_samples, results):
    correct = 0
    for i in range(len(test_samples)):
        if test_samples[i][2] == results[i]:
            correct += 1

    accuracy = correct / len(test_samples)
    return accuracy
train_results = evaluate(train, train)

print(get_accuracy(train, train_results))

x, y, c = list(zip(*train))

c = ['red' if value > 0 else 'blue' for value in train_results]

plt.scatter(x, y, c=c, s=4)
plt.show()
test = pd.read_csv("../input/kiwhs-comp-1-complete/test.csv")
test = test.values.tolist()

for row in test:
    del row[0]

results = evaluate(train, test)

x, y = list(zip(*test))

c = ['red' if value > 0 else 'blue' for value in results]

plt.scatter(x, y, c=c, s=4)
plt.show()

#from the second tutorial https://www.kaggle.com/poonaml/deep-neural-network-keras-way

submissions=pd.DataFrame({"Id (String)": list(range(len(results))),
                         "Category (String)": results})
submissions.to_csv("results.csv", index=False, header=True)