# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#read data

data = pd.read_csv("../input/data.csv")
data.tail() #tail is opposide head
#We can drop some columns

data.drop(["id","Unnamed: 32"],axis = 1,inplace = True)
data.info()
#Split Data as M&B

M = data[data.diagnosis == "M"]

B = data[data.diagnosis == "B"]
#Visualization, Scatter Plot



plt.scatter(M.radius_mean,M.area_mean,color = "Black",label="Malignant",alpha=0.2)

plt.scatter(B.radius_mean,B.area_mean,color = "Orange",label="Benign",alpha=0.3)

plt.xlabel("Radius Mean")

plt.ylabel("Area Mean")

plt.legend()

plt.show()



#We appear that it is clear segregation.
#Visualization, Scatter Plot



plt.scatter(M.radius_mean,M.texture_mean,color = "Black",label="Malignant",alpha=0.2)

plt.scatter(B.radius_mean,B.texture_mean,color = "Lime",label="Benign",alpha=0.3)

plt.xlabel("Radius Mean")

plt.ylabel("Texture Mean")

plt.legend()

plt.show()
#change M & B 

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

#seperate data as x (features) & y (labels)

y= data.diagnosis.values

x1= data.drop(["diagnosis"],axis= 1) #we remowe diagnosis for predict
#normalization

x = (x1-np.min(x1))/(np.max(x1)-np.min(x1))
#Train-Test-Split 

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest =  train_test_split(x,y,test_size=0.3,random_state=42)
#Create-KNN-model

from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors = 20) #n_neighbors = K value

KNN.fit(xtrain,ytrain) #learning model

prediction = KNN.predict(xtest)
print("{}-NN Score: {}".format(20,KNN.score(xtest,ytest)))
#Find Optimum K value

scores = []

for each in range(1,50):

    KNNfind = KNeighborsClassifier(n_neighbors = each)

    KNNfind.fit(xtrain,ytrain)

    scores.append(KNNfind.score(xtest,ytest))

    

plt.plot(range(1,50),scores,color="black")

plt.xlabel("K Values")

plt.ylabel("Score(Accuracy)")

plt.show()
#Create-KNN-model

from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors = 3) #n_neighbors = K value

KNN.fit(xtrain,ytrain) #learning model

prediction = KNN.predict(xtest)



print("{}-NN Score: {}".format(3,KNN.score(xtest,ytest)))
#Create-KNN-model

from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors = 4) #n_neighbors = K value

KNN.fit(xtrain,ytrain) #learning model

prediction = KNN.predict(xtest)

print("{}-NN Score: {}".format(4,KNN.score(xtest,ytest)))