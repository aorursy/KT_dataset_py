# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from matplotlib import colors



def plot_one(array):

    ax = plt.subplot(1, 1, 1)

    cmap = colors.ListedColormap(

        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    norm = colors.Normalize(vmin=0, vmax=9)

    

    input_matrix = array

    ax.imshow(input_matrix, cmap=cmap, norm=norm)

    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    

    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])

    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     

    ax.set_xticklabels([])

    ax.set_yticklabels([])
with open("/kaggle/input/abstraction-and-reasoning-challenge/evaluation/c97c0139.json", "r") as f:

    loaded_task = json.load(f)

    

tasksOne = []

resultsOne = []

trainTasks = []

testTasks = []



print("Training set:")



for task in loaded_task["train"]:

    tempTasks = ({"input" : task["input"], "output" : task["output"]})

    plot_one(task["input"])

    plt.show()

    plot_one(task["output"])

    plt.show()

    trainTasks.append(tempTasks)

    

print("testing set:")

    

for task in loaded_task["test"]:

    tempTasks = ({"input" : task["input"], "output" : task["output"]})

    plot_one(task["input"])

    plt.show()

    plot_one(task["output"])

    plt.show()

    testTasks.append(tempTasks)
def taskToDataFrame(tasks):

    data = {

            "X" : [],

            "Y" : [],

            "Color" : [],

            "ResultColor" : []

           }

    df = pd.DataFrame(data)

    tempDf = pd.DataFrame(data)



    x = 0

    y = 0

    for task in tasks:

        for row in task["input"]:

            for px in row:

                tempDf = tempDf.append({"X" : x, "Y" : y, "Color" : px} , ignore_index=True)

                x += 1

            y += 1

            x = 0



        y = 0

        for row in task["output"]:

            for px in row:

                IndexLabel = tempDf.query("X == " + str(x) + " & Y == " + str(y)).index.tolist()

                tempDf.loc[IndexLabel, "ResultColor"] = px

                x += 1

            y += 1

            x = 0

        y = 0

        df = df.append(tempDf, ignore_index=True)

        tempDf = pd.DataFrame(data)

    #pd.set_option('display.max_rows', df.shape[0]+1) #to print all the rows of the df

    return df

    

trainDf = taskToDataFrame(trainTasks)

testDf = taskToDataFrame(testTasks)



print("Training dataframe")

print(trainDf)

print("Testing dataframe")

print(testDf)
y_train = trainDf.ResultColor

X_train = trainDf.drop(["ResultColor"], axis=1, inplace =False)

y_test = testDf.ResultColor

X_test = testDf.drop(["ResultColor"], axis=1, inplace =False)
from sklearn.svm import SVC

clf = SVC(kernel = "linear")

clf.fit(X_train, y_train)

train_score = clf.score(X_train, y_train)

test_score = clf.score(X_test, y_test)

print ('train accuracy =', train_score)

print ('test accuracy =', test_score)
def toCorrectJson(predicted):

    width = 22

    height = 21

    indexs = 0

    correctJson = []

    for i in range(height):

        indexs = i * width;

        if (i == 0):

            indexs = 22

        correctJson.append(predicted[indexs - 22:indexs].astype(int))

    

    return correctJson;        
from sklearn import metrics

predicted = clf.predict(X_test)

print("Output: ")

plot_one(toCorrectJson(predicted))

plt.show()

#print (metrics.confusion_matrix(y_test, predicted))

print (metrics.classification_report(y_test, predicted))
precision_0 = 21 /(21 + 4)

recall_1 = 93 / (93 + 4)

print ('precision_0 =', precision_0)

print ('recall_1 =',recall_1)