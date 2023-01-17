# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected = True)

import plotly.graph_objs as go



print(plt.style.available) # look at available plot styles

plt.style.use('ggplot')





import os

print(os.listdir("../input"))

data = pd.read_csv("../input/column_2C_weka.csv")

data.head(10)
data.info()
# check for NaN values

data.isna().sum()



data.describe()
fig = {

    "data" : [

        {

            "x" : data["class"],

            "name" : "class",

            "marker" : {"color" : ["blue", "red"]},

            "type" : "histogram"

        }

    ],

    

    "layout" : {

        "title" : "Class Count",

        "titlefont" : {"color" : "black",

                       "size" : 20},

        "xaxis" : {"title" : "Class",

                   "color" : "black"},

        "yaxis" : {"title" : "Count",

                   "color" : "black"}

    }

}



iplot(fig)
class_count = data["class"].value_counts()

fig = {

    "data" : [

        {

            "labels" : class_count.index,

            "values" : class_count.values,

            "name" : "Class",

            "hoverinfo" : "label+value+name",

            "textinfo" : "percent",

            "marker" : {"line" : {"width" : 1.1, "color" : "black"}},

            "hole" : .3,

            "type" : "pie"

            

        },

    ],

    

    "layout" : {

        "title" : "Class Count",

        "titlefont" : {"size" : 20}

    }

}



iplot(fig)

plt.savefig('graph.png')
# for KNN we need to change class names to the integer 

# Abnormal => 1  /   Normal => 0

data["class"] = [1 if each == "Abnormal" else 0 for each in data["class"]]
# preparing the data for classification

y = data["class"]

x_data = data.drop(["class"], axis=1)
# now we need to do normalization

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .3, random_state = 42)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import r2_score

knn = KNeighborsClassifier(n_neighbors = 3)



knn.fit(x_train, y_train)

y_head = knn.predict(x_test)



print(knn.score(x_test, y_test))
train_accuracy = []

test_accuracy = []



for k, i in enumerate(np.arange(1, 25),1):

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(x_train, y_train)

    

    train_accuracy.append(knn.score(x_train, y_train))

    test_accuracy.append(knn.score(x_test, y_test))

print("With KNN (K=3) accuracy is: ",knn.score(x_test,y_test))

    



#plt.subplots(figsize=(18,10))

#plt.plot(np.arange(1, 25), train_accuracy, label = "Train")

#plt.plot(np.arange(1, 25), test_accuracy, label = "Test")

#plt.xlabel("Number of Neighbors (K)")

#plt.ylabel("Accuracy")

#plt.title("K vs Accuracy")

#plt.legend()

#plt.show()

fig = {

    "data" : [

        {

            "x" : np.arange(1, 25),

            "y" : train_accuracy,

            "name" : "Train",

            "text" : "train",

            "marker" : {"color" : "red"},

            "type" : "scatter",

            "mode" : "lines+markers"

        },

        {

            "x" : np.arange(1, 25),

            "y" : test_accuracy,

            "name" : "Test",

            "text" : "test",

            "marker" : {"color" : "green"},

            "type" : "scatter",

            "mode" : "lines+markers"

        }

    ],

    

    "layout" : {

        "title" : "K Value vs Accuracy",

        "titlefont" : {"color" : "black",

                       "size" : 20},

        "xaxis" : {"title" : "Number of Neighbors (K)",

                   "titlefont" : {"size" : 13,

                                  "color" : "blue"}},

        "yaxis" : {"title" : "Accuracy",

                   "titlefont" : {"size" : 13,

                                  "color" : "blue"}},

        "showlegend" : True

    }

}



iplot(fig)


