# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import load_breast_cancer

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import random

from operator import itemgetter

import math



# iris = load_breast_cancer()



# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.4, random_state=42)



# clf1 = GaussianNB()

# clf2 = DecisionTreeClassifier()

# clf3 = KNeighborsClassifier()





# def generate_population():

#     pass



# def crossover():

#     pass



# def mutation():

#     raise Exception("Not Implemented Yet")



# def fintness():

#     pass



# mutation()

# clfs = [(clf1,"Naive Bayes"),(clf2,"Decision Tree"),(clf3,"KNN")]

# for _ in range(5000):

#     for clf,l in clfs:

#         clf.fit(X_train,y_train)

#         y = clf.predict(X_test)

# print(l,metrics.precision_score(y_test,y),metrics.recall_score(y_test,y))
dataset = load_breast_cancer()

data = dataset.data

target = dataset.target
dataset
dataset.feature_names
# dataset.data[:,[1,2,3]]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.4, random_state=42)

clf = DecisionTreeClassifier()
def fitness(features):

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.4, random_state=42)

    selected_feature_id = []

    for i in range(len(dataset.feature_names)):

        if features[i] == 1:

            selected_feature_id.append(i)

    pass_X = X_train[:,selected_feature_id]

    clf.fit(pass_X,y_train)

    pass_X_test = X_test[:,selected_feature_id]

    y_pred = clf.predict(pass_X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

    

    
def crossover(parent1,parent2):

    child1 = []

    child2 = []

    size = len(parent1)

    index1 = random.randrange(0,size-2)

    index2 = random.randrange(index1,size-1)

    child1 = child1 + parent2[:index1] + parent1[index1:index2] + parent2[index2:]

#     child1.insert(parent2[:index1],-1)

#     child1.insert(parent1[index1+1:index2],-1)

#     child1.insert(parent2[index2+1:],-1)

    

#     child2.insert(parent1[:index1],-1)

#     child2.insert(parent2[index1+1:index2],-1)

#     child2.insert(parent1[index2+1:],-1)

    child2 = child2 + parent1[:index1] + parent2[index1:index2] + parent1[index2:]

    return child1,child2
t = [4, 5, 6, 3, 9] 

i = [2, 3] 

t = t+i

print(t)
def selection(popu):

    popu = sorted(popu,key=itemgetter(0),reverse = True)

    popu = popu[:10]

    return popu
def mutation(parent):

    l = len(parent)

    index = random.randint(0,l-1)

    parent[index] = 1 - parent[index]

    return parent
#intializing the population

popu = []

print(type(popu))

feature_len = len(dataset.feature_names)



for i in range(100):

    x = []

    for i in range(feature_len):

        x.append(random.randint(0,1))

    t = fitness(x) , x

    popu.append(t)

# print(popu)

# len(popu[0][1])
#selction

popu = selection(popu)
print(popu[6][1])

popu_mut = mutation(popu[6][1])

print(popu_mut)
t = sorted(popu , key = itemgetter(0),reverse=True)
print(t)
popu = sorted(popu , key = itemgetter(0),reverse=True)
popu_mut
p1 = popu[0][1]

p2 = popu[1][1]
print(p1)

print(p2)
c1,c2 = crossover(p1,p2)

print(c1)

print(len(c1))

print(c2)
generations = 100

get_out_condition = popu[0][0]

for i in range(generations):

    #selection

    popu = popu[:10]

    

    #crossover

    for i in range(50):

        index1 = random.randint(0,len(popu)-1)

        index2 = random.randint(0,len(popu)-1)

        child1 = []

        child2 = []

        parent1 = popu[index1][1]

        parent2 = popu[index2][1]

        child1,child2 = crossover(parent1,parent2)

#         print(child1)

#         print(len(child1),feature_len)

#         print(child2)

        t1 = fitness(child1) , child1

        t2 = fitness(child2) , child2

        popu.append(t1)

        popu.append(t2)



    #mutation

    index = random.randint(0,len(popu)-1)

    mut = popu[index][1]

    popu.pop(index)

    mut = mutation(mut)

    t = fitness(mut),mut

    popu.append(t)

    

    popu = sorted(popu , key = itemgetter(0),reverse=True)

    

    x = abs(get_out_condition - popu[0][0])

    if x <0.001:

        break;

    else:

        get_out_condition = popu[0][0]

print(i)      

#solution

print("the solution is :",popu[0])



    