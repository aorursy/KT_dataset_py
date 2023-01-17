# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np


# -*- coding: utf-8 -*-

"""
ECE657A, S'20, Midterm, Problem 1
Skeleton solution file.
"""

"""
You are not allowed to import anything except for the mentioned in the file.
You need to implement the following method. You are
allowed to define whatever subroutines you like to
structure your code.
"""

"""
    You need to implement this method. 
    
    Input:
    input_features: It will have dimensions n x d,
    where n will be number of samples and d will be number of features for each sample.
    num_neighbors: They are the number of neighbors that will be look while
    determining the class of the input features.
    true_labels: The true label of the features with dimensions n x 1, where n
    is the number of samples.

    
    Output:
    predicted_class: It will be the list of the class labels for each sample. The 
    dimension of the list will be n x 1, where n will be number of samples
    f1_score:Weighted F1-Score of the prediction.
    confusion_matrix = confusion matrix.
"""
    




def euclidean_distance(a1,a2):
    a1,a2 = np.array(a1), np.array(a2)
    distance = 0
    for i in range(len(a1) - 1):
        distance += (a1[i] - a2[i]) ** 2
    euclidean_distanc = np.sqrt(distance)
    return euclidean_distanc


def predict(k ,train_set,test_instance):
    distances = []
    for i in range(len(train_set)):
        d = euclidean_distance(train_set[i][:-1],test_instance)
        distances.append((train_set[i],d))
    distances.sort(key=lambda x: x[1])
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    c = {}
    for i in range(len(neighbors)):
        response = neighbors[i][-1]
        if response in c:
            c[response] += 1
        else:
            c[response] = 1
    votes = sorted(c.items(), key=lambda x:x[1], reverse=True)
    return(votes[0][0])

    
def evaluate(y_true, y_pred):
    c = [[0,0,0], [0,0,0], [0,0,0]]
    weighted_f1_score = 0
    for act, pred in zip(y_true, y_pred):
        act = int(act)
        pred = int(pred)
        c[act][pred] += 1
    precision = [0,0,0]
    recall = [0,0,0]
    accuracy = (c[0][0]+c[1][1]+c[2][2]) / len(y_true)
    precision[0] = c[0][0]/(c[0][0]+c[1][0]+c[2][0])
    precision[1] = c[1][1]/(c[0][1]+c[1][1]+c[2][1])
    precision[2] = c[2][2]/(c[0][2]+c[1][2]+c[2][2])
    
    recall[0] = c[0][0]/(c[0][0]+c[0][1]+c[0][2])
    recall[1] = c[1][1]/(c[1][0]+c[1][1]+c[1][2])
    recall[2] = c[2][2]/(c[2][0]+c[2][1]+c[2][2])
    total = 0
    class_f1_score = [0,0,0]
    for i in range(3):
        class_f1_score[i] = 2 * (precision[i] * recall[i])/(precision[i]+recall[i])
        total += class_f1_score[i]
    weighted_f1_score = total/3
    confusion_matrix = c
    return accuracy, weighted_f1_score, confusion_matrix


def predictor(input_features,num_neighbors,true_labels):
    inputcsv = "../DataA.csv"
    data = np.genfromtxt(inputcsv,delimiter = ',')
    np.random.seed(1)
    np.random.shuffle(data)
    data = data[~np.isnan(data).any(axis=1)]    
    train_set = data[: int(len(data)*8/10)].tolist()
    
    for row in input_features:
        test_samples = row[:-1]
        prediction = predict(num_neighbors, train_set, test_samples)
        predicted_class.append(prediction)

    accuracy, f1_score, confusion_matrix = evaluate(true_labels,predicted_class)
    print("k is:", num_neighbors)
    print("Accuracy is:", accuracy)
    print("f1_score is:", f1_score)
    print("confusion_matrix is:", confusion_matrix)
    
    return predicted_class,f1_score,confusion_matrix

    
# inputcsv = "../input/ece657a-midterm/DataA.csv"
# data = np.genfromtxt(inputcsv,delimiter = ',')
# np.random.seed(1)
# np.random.shuffle(data)
# data = data[~np.isnan(data).any(axis=1)]    
# train_set = data[: int(len(data)*8/10)].tolist()
# test_set = data[int(len(data)*8/10):].tolist()
# true_labels = np.array(test_set)[:,4]
# predicted_class = []
# num_neighbors = 8    

# inputcsv = "../input/ece657a-midterm/DataA.csv"
# test = np.genfromtxt(inputcsv,delimiter = ',')
# np.random.seed(1)
# np.random.shuffle(test)
# test_set = data[int(len(data)*8/10):].tolist()
# test_x = np.array(test_set)[:,0:-1]
# test_y= np.array(test_set)[:,4]
# num_neighbors = 8 


# predictor(test_x,num_neighbors,test_y)



# k_evaluations = []
# for k in range(1,20,1):
#     predicted_class = []
#     for row in test_set:
#         predictors_only = row[:-1]
#         prediction = predict(k, train_set, predictors_only)
#         predicted_class.append(prediction)
#     actual = np.array(test_set)[:, -1]    
#     curr_accuracy = evaluate(actual,predicted_class)
#     k_evaluations.append((k,curr_accuracy))
# k_evaluations