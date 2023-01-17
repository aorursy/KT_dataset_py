from math import sqrt

import numpy as np

import pandas as pd 
dataset = [

    [2.7810836,2.550537003,0],

    [1.465489372,2.362125076,0],

    [3.396561688,4.400293529,0],

    [1.38807019,1.850220317,0],

    [3.06407232,3.005305973,0],

    [7.627531214,2.759262235,1],

    [5.332441248,2.088626775,1],

    [6.922596716,1.77106367,1],

    [8.675418651,-0.242068655,1],

    [7.673756466,3.508563011,1]

          ]
def Euclidean_distance(row1, row2):

    distance = 0

    for i in range(len(row1)-1):

        distance += (row1[i] - row2[i])**2

    return sqrt(distance)

    





test = [8.675418651, 2.088626775,1]

for i in dataset:

    dis = Euclidean_distance(test, i)



    print(dis)
def Get_Neighbors(train, test_row, num):

    distance = list() # []

    data = []

    for i in train:

        dist = Euclidean_distance(test_row, i)

        distance.append(dist)

        data.append(i)

    distance = np.array(distance)

    data = np.array(data)

    index_dist = distance.argsort()

    data = data[index_dist]

    neighbors = data[:num]

    

    return neighbors

   
def predict_classification(train, test_row, num):

    Neighbors = Get_Neighbors(train, test_row, num)

    Classes = []

    for i in Neighbors:

        Classes.append(i[-1])

    prediction = max(Classes, key= Classes.count)

    return prediction
predict_classification(dataset, test, 9)
def Euclidean_distance(row1, row2):

    distance = 0

    for i in range(len(row1)-1):

        distance += (row1[i] - row2[i])**2

    return sqrt(distance)



def Get_Neighbors(train, test_row, num):

    distance = list() # []

    data = []

    for i in train:

        dist = Euclidean_distance(test_row, i)

        distance.append(dist)

        data.append(i)

    distance = np.array(distance)

    data = np.array(data)

    index_dist = distance.argsort()

    data = data[index_dist]

    neighbors = data[:num]

    return neighbors



def predict_classification(train, test_row, num):

    Neighbors = Get_Neighbors(train, test_row, num)

    Classes = []

    for i in Neighbors:

        Classes.append(i[-1])

    prediction = max(Classes, key= Classes.count)

    return prediction





def Evaluate(y_true, y_pred):

    n_correct = 0

    for i in range(len(y_true)):

        if y_true[i] == y_pred[i]:

            n_correct += 1

    acc = n_correct/len(y_true)

    return acc

    



prediction = predict_classification(dataset, test, 4)

print("We expected {}, Got {}".format(test[-1], prediction))
from sklearn.model_selection import train_test_split
df = pd.read_csv("../input/iris.csv")

iris_data = df.values
trainiris, testiris = train_test_split(iris_data, test_size = 0.25)
y_pred = []

y_true = testiris[:, -1]

for i in testiris:

    prediction = predict_classification(trainiris, i, 10)

    y_pred.append(prediction)
Evaluate(y_true, y_pred)