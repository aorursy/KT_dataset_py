import pandas as pd

import numpy as np

import math

import operator

import random
data=pd.read_csv("../input/iris.csv")



#Removing the labeling coloum

testdata = data.iloc[:,0:4]



#Converting the dataset into a list

testset = []

for i in range(len(testdata)):

    testset.append(list(testdata.values[i]))
def ED(x1, x2): 

    distance = 0

    for x in range(len(x1)):

        distance += np.square(x1[x] - x2[x])

    return np.sqrt(distance)
def distance(testset,centroid):

    dist1 = []

    dist2 = []

    dist3 = []

    dist = []

    

    for i in range(len(testset)):

        dist1.append(ED(centroid[0],testset[i]))

        dist2.append(ED(centroid[1],testset[i]))

        dist3.append(ED(centroid[2],testset[i]))

    

    dist.append(dist1)

    dist.append(dist2)

    dist.append(dist3)

    return dist
def group(testset,distance):

    g1 = []

    g2 = []

    g3 = []

    g = []

    for i in range(len(testset)):

        mini = 0

        for j in range(len(distance)):

            if distance[mini][i] > distance[j][i]:

                mini = j

        if mini == 0:

            g1.append(1)

            g2.append(0)

            g3.append(0)

        elif mini == 1:

            g1.append(0)

            g2.append(1)

            g3.append(0)

        else:

            g1.append(0)

            g2.append(0)

            g3.append(1)



    g.append(g1)

    g.append(g2)

    g.append(g3)

    return g
def centroid(testset,group):

    cen1 = [0,0,0,0]

    cen2 = [0,0,0,0]

    cen3 = [0,0,0,0]

    cen = []

    

    one1 = 0

    one2 = 0

    one3 = 0



    for i in range(len(testset)):

        if group[0][i] == 1:

            one1 += 1

            for j in range(4):

                cen1[j] = cen1[j] + testset[i][j]

        if group[1][i] == 1:

            one2 += 1

            for j in range(4):

                cen2[j] = cen2[j] + testset[i][j]

        if group[2][i] == 1:

            one3 += 1

            for j in range(4):

                cen3[j] = cen3[j] + testset[i][j]



    for j in range(4):

        cen1[j] = cen1[j]/one1

        cen2[j] = cen2[j]/one2

        cen3[j] = cen3[j]/one3

    

    cen.append(cen1)

    cen.append(cen2)

    cen.append(cen3)

    return cen
def accuracy(grup):

    

    #devide the final group matrix depending on the label of the original dataset.

    p1 = grup[0][0:50]

    p2 = grup[1][50:101]

    p3 = grup[2][101:152]

    

    pred=0

    

    for i in range(len(p1)):

        if p1[i] == 1:

            pred += 1

    for i in range(len(p2)):

        if p2[i] == 1:

            pred += 1

    for i in range(len(p3)):

        if p3[i] == 1:

            pred += 1

    pred=(pred/150)*100

    

    return pred
def kmeans():

    

    #assuming 3 clusters

    k = 3

    

    #randomly choose three index

    clus_index = []

    random.seed(42)

    for i in range(k):

        clus_index.append(random.randint(0,151))

    

    #initialize three centroids depending on the indices for the first iteration

    c1 = testset[clus_index[0]]

    c2 = testset[clus_index[1]]

    c3 = testset[clus_index[2]]

    centrd = []

    centrd.append(c1)

    centrd.append(c2)

    centrd.append(c3)

    

    grp1 = []

    

    while True:

        dist = distance(testset,centrd)

        grp = group(testset,dist)

        centrd = centroid(testset,grp)

        

        #checking that the last iteration group matrix and current iteration group matrix is equal or not

        if grp == grp1:

            break

        else:

            grp1 = grp

    pred_acc = accuracy(grp)

    print(f'Accuracy: {pred_acc}%')
kmeans()