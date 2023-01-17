from collections import Counter

input = ['abcde', 'sdaklfj', 'asdjf', 'na', 'basdn', 'sdaklfj', 'asdjf', 'na', 'asdjf', 'na', 'basdn', 'sdaklfj', 'asdjf']

query = ['abcde', 'sdaklfj', 'asdjf', 'na', 'basdn']



def search(arr, s): 

    counter = 0

    for j in range(len(arr)): 

        if (s == (arr[j])): 

            counter += 1

    return counter 

  

def res(arr, q): 

    for i in range(len(q)): 

        print(search(arr, q[i]), end = " ") 

        

print(res(input, query))
bracket = "{[()]}"



from pythonds.basic import Stack



def isBalance(input):

    s = Stack()

    balanced = True

    index = 0

    while index < len(input) and balanced:

        symbol = input[index]

        if symbol == "(":

            s.push(symbol)

        else:

            if s.isEmpty():

                balanced = False

            else:

                s.pop()



        index = index + 1



    if balanced and s.isEmpty():

        return True

    else:

        return False



print(isBalance(bracket))

    
import numpy as np



matrix1 = [[1,2],[3,4]]

matrix2 = [[7,8,9],[6,5,4]]



res = np.array(matrix1)

print(res,"\n")

print("Transpose")

print(res.transpose(),"\n")

print("Flatten")

print(res.flatten())
from sklearn.naive_bayes import GaussianNB

from sklearn import datasets

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

iris = datasets.load_iris()

data = iris.data

target = iris.target
plt.figure(figsize=(15,10))

plt.plot(data)
from sklearn.model_selection import train_test_split

(x_train, x_test, y_train, y_test) = train_test_split(data,target, test_size=0.3)
classifier = GaussianNB()

classifier.fit(x_train,y_train)

predict = classifier.predict(x_test)

print(accuracy_score(y_test, predict))
print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))