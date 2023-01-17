# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")
data
data["class"].value_counts()#class groups
Abnormal = data[data["class"] != "Normal"] # Spondylolisthesis and Hernia                

Normal = data[data["class"]  == "Normal"] # Normal 
data["class"]=[1 if each =="Normal"  else 0 for each in data["class"]]



data["class"].value_counts()
color_list = ['red' if i==0 else 'yellow' for i in data.loc[:,'class']]

pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],

                                       c=color_list,

                                       figsize= [22,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '#¹',

                                       edgecolor= "orange")
# scatter plot

plt.scatter(Abnormal["degree_spondylolisthesis"],Abnormal["pelvic_tilt"],color="red",label="kotu",alpha= 0.8)

plt.scatter(Normal["degree_spondylolisthesis"],Normal["pelvic_tilt"],color="yellow",label="iyi",alpha= 0.8)

plt.xlabel("radius_mean")

plt.ylabel("texture_mean")

plt.legend()

plt.show()


y=data["class"].values



x_data=data.drop(["class"],axis=1)



x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)
# knn model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5) # n_neighbors = k

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print(" {} nn score: {} ".format(5,knn.score(x_test,y_test)))

score_list = []

for each in range(1,50):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,50),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
print("max values; f(21)={0} and f(25)={1}".format(score_list[21],score_list[25]))
neig = np.arange(1, 50)

train_accuracy = []

test_accuracy = []

# Loop over different values of k

for i, k in enumerate(neig):

    # k from 1 to 25(exclude)

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit with knn

    knn.fit(x_train,y_train)

    #train accuracy

    train_accuracy.append(knn.score(x_train, y_train))

    # test accuracy

    test_accuracy.append(knn.score(x_test, y_test))



# Plot

plt.figure(figsize=[13,8])

plt.plot(neig, test_accuracy, label = 'Testing Accuracy')

plt.plot(neig, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.title('-value VS Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neig)

plt.savefig('graph.png')

plt.show()

print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))