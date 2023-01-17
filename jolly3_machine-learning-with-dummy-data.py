import numpy as np # linear algebra

import math

import random

from copy import deepcopy
# Random data points

data_points = np.random.randint(0,50,(100,2)) 

dist = {}



# Class labels

labels = ["class-A", "class-B", "class-C", "class-D", "class-E", "class-F", "class-G"]

'''Assigns a random label'''

def assign_random_label() :

  return labels[random.randint(0,6)]

# Contains labels associated with data points

data_dict = dict()

for i in range(len(data_points)) :

  data_dict[tuple(data_points[i])] = assign_random_label()

  print(str(tuple(data_points[i])) +","+ str(data_dict[tuple(data_points[i])]))
'''Classifies the point with k nearest neighbour algorithm

params : point - The point to be classified, k - the number of nearest neighbourrs to be considered '''

def KNN(point, k) :

  # Calculation of distance from the point to all other data points.

  for i in data_dict.keys() :

    dist[i] = math.hypot(point[0]-i[0], point[1]-i[1])

    

  # sorting on the basis of distance

  dist_sorted = sorted(dist.items(), key=lambda kv: kv[1])

  

  #For maintaing the count of labels in nearby k data points

  dict_classes = {}

  for i in labels :

    dict_classes[i] = 0

    

  for point in dist_sorted :

    dict_classes[data_dict[point[0]]] += 1

    k = k-1

    if(k == 0) :

      # If k data points covered, then stop iterating

      break

    

  mx = 0

  final_label = ""

  for i in labels :

    if(mx < dict_classes[i]) :

      #Assigning labels based on majority

      mx = dict_classes[i]

      final_label = i

  

  print(final_label)
KNN((40,25), 20)
'''Creating a dummy dataset'''

data_points = np.random.randint(0,50,(100,2)) 



def kMeans(k) :

    rep = {}

    oldLabels = []

    newLabels = []

    for i in range(k) :

        newLabels.append(list((random.randint(1,50), random.randint(1,50))))

    while(True) :

        # Assigning means to the data points.

        for i in range(len(data_points)) :

            point = tuple(data_points[i])

            index = 0

            dist = [0 for x in range(k)]

            for j in range(k) :

                dist[j] = math.hypot(newLabels[j][0] - point[0] , newLabels[j][1] - point[1])

            mn = 9999999

            for j in range(k) :

                if(dist[j] < mn) :

                    mn = dist[j]

                    index = j

        #Update the representative of the points with the nearest mean point

            rep[point] = newLabels[index]

        oldLabels = deepcopy(newLabels)

        newLabels = [[0,0]]*k

        count = [0]*k



        #Updating the means

        for i in range(len(data_points)) :

            point = tuple(data_points[i])

            position = {}

            for j in range(k) :

                position[tuple(oldLabels[j])] = j

            label = deepcopy(newLabels[position[tuple(rep[point])]])

            label[0] += point[0]

            label[1] += point[1]

            newLabels[position[tuple(rep[point])]] = deepcopy(label)

            count[position[tuple(rep[point])]] += 1



        for i in range(k) :

            newLabels[i][0] = newLabels[i][0]//count[i]

            newLabels[i][1] = newLabels[i][1]//count[i]



        # Checking if the means after updation are different. Terminating Condition

        cnt = 0

        for i in newLabels :

            if i in oldLabels :

                cnt += 1

            if(cnt == k) :

                  break

    #separating the clusters

    cluster = [[] for x in range(k)]

    position = {}

    for j in range(k) :

        position[tuple(newLabels[j])] = j



    for i in range(len(data_points)) :

        point = tuple(data_points[i])

        index = int(position[tuple(rep[point])])

        cluster[index].append(point)



    #Printing the clusters

    for i in range(len(cluster)) :

        print("Cluster-"+str(i+1))

        print(cluster[i])

        print()
kMeans(20)