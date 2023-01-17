from IPython.display import Image

import os

Image("../input/week4images/K-Nearest Neighbor Algorithm 1.jpeg", width="800px")
Image("../input/week4images/K-Nearest Neighbor Algorithm 2.jpeg", width="800px")
from IPython.display import YouTubeVideo



YouTubeVideo('HVXime0nQeI', width=800, height=300)
Image("../input/week4images/K-Nearest Neighbor Algorithm 3.jpeg", width="1600px")
import pandas as pd

import numpy as np
"""

Generate a set of sample data

"""



def create_data():

    features = np.array(

        [[2.88, 3.05], [3.1, 2.45], [3.05, 2.8], [2.9, 2.7], [2.75, 3.4],

         [3.23, 2.9], [3.2, 3.75], [3.5, 2.9], [3.65, 3.6],[3.35, 3.3]])

    labels = ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']

    return features, labels
'''

Print the data

'''



features, labels = create_data()

print('features: \n',features)

print('labels: \n',labels)
"""

Draw the sample data

"""



from matplotlib import pyplot as plt

%matplotlib inline



plt.figure(figsize=(5, 5))

plt.xlim((2.4, 3.8))

plt.ylim((2.4, 3.8))

x_feature=list(map(lambda x:x[0],features)) # Return feature x of each data

y_feature=list(map(lambda y:y[1],features)) 

plt.scatter(x_feature[:5],y_feature[:5],c="b") # Draw the data points of A

plt.scatter(x_feature[5:],y_feature[5:],c="g") 

plt.scatter([3.18],[3.15],c="r",marker="x") # The coordinates of the testing point: [3.1，3.2]
Image("../input/week4images/K-Nearest Neighbor Algorithm 4.jpeg", width="1600px")
"""

Manhattan Distance

"""



import numpy as np



def d_man(x, y):

    d = np.sum(np.abs(x - y))

    return d



x = np.array([3.1, 3.2])

print("x:", x)

y = np.array([2.5, 2.8])

print("y:", y)

d_man = d_man(x, y)

print(d_man)
"""

Euclidean distance

"""



import numpy as np



def d_euc(x, y):

    d = np.sqrt(np.sum(np.square(x - y)))

    return d



x = np.random.random(10)  # Randomly generate an array of 10 numbers as the value of x

print("x:", x)

y = np.random.random(10)

print("y:", y)

distance_euc = d_euc(x, y)

print(distance_euc)
"""

Majority voting method

"""



import operator



def majority_voting(class_count):

    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_class_count



arr = {'A': 3, 'B': 2, "C": 6, "D": 5}

majority_voting(arr)
"""

Complete realization of KNN

"""



def knn_classify(test_data, train_data, labels, k):

    distances = np.array([])  # Create an empty array to save the distance

    for each_data in train_data:  # Calculate data similarity using Euclidean distance

        d = d_euc(test_data, each_data)

        distances = np.append(distances, d)

    sorted_distance_index = distances.argsort()  # Get the indices sorted by distance

    sorted_distance = np.sort(distances)

    r = (sorted_distance[k]+sorted_distance[k-1])/2  # Calculate the radius

    class_count = {}

    for i in range(k):  # Majority vote

        vote_label = labels[sorted_distance_index[i]]

        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    final_label = majority_voting(class_count)

    return final_label, r
test_data=np.array([3.18,3.15])

final_label,r=knn_classify(test_data, features, labels, 5)

final_label
def circle(r, a, b):  # Polar coordinates ：x=r*cosθ，y=r*sinθ.

    theta = np.arange(0, 2*np.pi, 0.01)

    x = a+r * np.cos(theta)

    y = b+r * np.sin(theta)

    return x, y



k_circle_x, k_circle_y = circle(r, 3.18, 3.15)



plt.figure(figsize=(5, 5))

plt.xlim((2.4, 3.8))

plt.ylim((2.4, 3.8))

x_feature = list(map(lambda x: x[0], features))  # Return feature x of each data

y_feature = list(map(lambda y: y[1], features))

plt.scatter(x_feature[:5],y_feature[:5],c="b") # Draw the data points of A

plt.scatter(x_feature[5:],y_feature[5:],c="g") 

plt.scatter([3.18],[3.15],c="r",marker="x") # The coordinates of the testing point: [3.1，3.2]

plt.plot(k_circle_x, k_circle_y)
from ipywidgets import interact, fixed



def change_k(test_data, features, k):

    final_label, r = knn_classify(test_data, features, labels, k)

    k_circle_x, k_circle_y = circle(r, 3.18, 3.15)

    plt.figure(figsize=(5, 5))

    plt.xlim((2.4, 3.8))

    plt.ylim((2.4, 3.8))

    x_feature = list(map(lambda x: x[0], features))  # Return feature x of each data

    y_feature = list(map(lambda y: y[1], features))

    plt.scatter(x_feature[:5],y_feature[:5],c="b") # Draw the data points of A

    plt.scatter(x_feature[5:],y_feature[5:],c="g") 

    plt.scatter([3.18],[3.15],c="r",marker="x") # The coordinates of the testing point: [3.1，3.2]

    plt.plot(k_circle_x, k_circle_y)



interact(change_k, test_data=fixed(test_data),

         features=fixed(features), k=[3, 5, 7, 9])
"""

Load dataset

"""



import pandas as pd



lilac_data = pd.read_csv("../input/week4data/syringa.csv")

lilac_data.head() # Preview first 5 rows
"""

Plot subgraphs of features

"""



fig, axes = plt.subplots(2, 3, figsize=(20, 10))  # Build a 2*3 graph space, 2 rows and 3 columns

fig.subplots_adjust(hspace=0.3, wspace=0.2)  # Define height space and width space

axes[0, 0].set_xlabel("sepal_length")  # Define x-axis label

axes[0, 0].set_ylabel("sepal_width")  # Define y-axis label

axes[0, 0].scatter(lilac_data.sepal_length[:50],

                   lilac_data.sepal_width[:50], c="b")

axes[0, 0].scatter(lilac_data.sepal_length[50:100],

                   lilac_data.sepal_width[50:100], c="g")

axes[0, 0].scatter(lilac_data.sepal_length[100:],

                   lilac_data.sepal_width[100:], c="r")

axes[0, 0].legend(["daphne", "syringa", "willow"], loc=2)  # Define sample



axes[0, 1].set_xlabel("petal_length")

axes[0, 1].set_ylabel("petal_width")

axes[0, 1].scatter(lilac_data.petal_length[:50],

                   lilac_data.petal_width[:50], c="b")

axes[0, 1].scatter(lilac_data.petal_length[50:100],

                   lilac_data.petal_width[50:100], c="g")

axes[0, 1].scatter(lilac_data.petal_length[100:],

                   lilac_data.petal_width[100:], c="r")



axes[0, 2].set_xlabel("sepal_length")

axes[0, 2].set_ylabel("petal_length")

axes[0, 2].scatter(lilac_data.sepal_length[:50],

                   lilac_data.petal_length[:50], c="b")

axes[0, 2].scatter(lilac_data.sepal_length[50:100],

                   lilac_data.petal_length[50:100], c="g")

axes[0, 2].scatter(lilac_data.sepal_length[100:],

                   lilac_data.petal_length[100:], c="r")



axes[1, 0].set_xlabel("sepal_width")

axes[1, 0].set_ylabel("petal_width")

axes[1, 0].scatter(lilac_data.sepal_width[:50],

                   lilac_data.petal_width[:50], c="b")

axes[1, 0].scatter(lilac_data.sepal_width[50:100],

                   lilac_data.petal_width[50:100], c="g")

axes[1, 0].scatter(lilac_data.sepal_width[100:],

                   lilac_data.petal_width[100:], c="r")



axes[1, 1].set_xlabel("sepal_length")

axes[1, 1].set_ylabel("petal_width")

axes[1, 1].scatter(lilac_data.sepal_length[:50],

                   lilac_data.petal_width[:50], c="b")

axes[1, 1].scatter(lilac_data.sepal_length[50:100],

                   lilac_data.petal_width[50:100], c="g")

axes[1, 1].scatter(lilac_data.sepal_length[100:],

                   lilac_data.petal_width[100:], c="r")



axes[1, 2].set_xlabel("sepal_width")

axes[1, 2].set_ylabel("petal_length")

axes[1, 2].scatter(lilac_data.sepal_width[:50],

                   lilac_data.petal_length[:50], c="b")

axes[1, 2].scatter(lilac_data.sepal_width[50:100],

                   lilac_data.petal_length[50:100], c="g")

axes[1, 2].scatter(lilac_data.sepal_width[100:],

                   lilac_data.petal_length[100:], c="r")
from sklearn.model_selection import train_test_split



#Get the entire features in the lilac dataset

feature_data = lilac_data.iloc[:, :-1]

label_data = lilac_data["labels"]  # Get the labels in the lilac dataset

x_train, x_test, y_train, y_test = train_test_split(

    feature_data, label_data, test_size=0.3, random_state=2)



x_test  # Output lilac_test and view
"""

Use scikit-learn to construct KNN model

"""



from sklearn import neighbors

import sklearn



def sklearn_classify(train_data, label_data, test_data, k_num):

    knn = neighbors.KNeighborsClassifier(n_neighbors=k_num)

    # Train

    knn.fit(train_data, label_data)

    # Predict

    predict_label = knn.predict(test_data)

    # Return

    return predict_label
"""

Predict by dataset

"""



y_predict=sklearn_classify(x_train, y_train, x_test, 3)

y_predict
"""

Accuracy calculation

"""



def get_accuracy(test_labels, pred_labels):  

    correct = np.sum(test_labels == pred_labels) # Calculate the number of correct predictions

    n = len(test_labels) # Total number of test data

    accur=correct/n

    return accur
get_accuracy(y_test, y_predict)
normal_accuracy = []  # Create an empty list of accuracy rates

k_value = range(2, 11)

for k in k_value:

    y_predict = sklearn_classify(x_train, y_train, x_test, k)

    accuracy = get_accuracy(y_test, y_predict)

    normal_accuracy.append(accuracy)



plt.xlabel("k")

plt.ylabel("accuracy")

new_ticks = np.linspace(0.6, 0.9, 10)  # Set the y-axis display

plt.yticks(new_ticks)

plt.plot(k_value, normal_accuracy, c='r')

plt.grid(True)  # Add grid
Image("../input/week4images/K-Nearest Neighbor Algorithm 5.png", width="1000px")
"""

Introduce the time function to calculate the running time of the program

"""



import time



#Without kd tree

time_start1 = time.time()

knn = neighbors.KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)

predict_label = knn.predict(x_test)

time_end1 = time.time()

print("normal_time:", time_end1-time_start1)



#With kd tree

time_start2 = time.time()

kd_knn = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')

kd_knn.fit(x_train, y_train)

predict_label = knn.predict(x_test)

time_end2 = time.time()

print("kd_tree_time:", time_end2-time_start2)