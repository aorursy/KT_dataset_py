from math import sqrt

import numpy as np



def Euclidean_Distance(row1, row2):

    distance = 0

    for i in range(len(row1)-1):

        distance += (row1[i] - row2[i])**2

    return sqrt(distance)
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

    



test = [8.675418651, 2.088626775,1]

for i in dataset:

    dis = Euclidean_Distance(test, i)

    print(dis)

    
def Get_Neighbors(train, test_row, num):

   

    

    distance = list() # []

    data = []

    for i in train:

        dist = Euclidean_Distance(test_row, i)

        distance.append(dist)

        data.append(i)

    distance = np.array(distance)

    data = np.array(data)

    index_dist = distance.argsort()

    data = data[index_dist]

    neighbors = data[:num]

    

    return neighbors

    

    
Get_Neighbors(dataset, test, 5)


def Predict_Classification(train, test_row, num):

    Neighbors = Get_Neighbors(train, test_row, num)

    Classes = []

    for i in Neighbors:

        Classes.append(i[-1])

    prediction = max(Classes, key= Classes.count)

    return prediction
Predict_Classification(dataset,dataset[0], 4)
prediction = Predict_Classification(dataset, dataset[0], 4)

print("We expected {}, Got {}".format(dataset[0][-1], prediction))