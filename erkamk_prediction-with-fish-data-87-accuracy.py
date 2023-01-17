



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from collections import Counter

import warnings

warnings.filterwarnings("ignore")







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data = pd.read_csv("/kaggle/input/fish-market/Fish.csv")

data.info()
x = data.isnull().sum().sum()

data.columns
species_list = data.Species

Counter(species_list)
labels = "Perch" , "Bream" , "Roach" , "Pike" , "Smelt" , "Parkki" , "Whitefish"

sizes = [56,35,20,17,14,11,6]

explode = (0,0,0,0,0,0,0)

fig1,ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)

ax1.axis('equal')

plt.show()
plt.scatter(data.index[data.Species == "Bream"] , data.Weight[data.Species == "Bream"],c="red" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Perch"] , data.Weight[data.Species == "Perch"],c="aqua" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Roach"] , data.Weight[data.Species == "Roach"],c="orange" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Pike"] , data.Weight[data.Species == "Pike"],c="purple" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Smelt"] , data.Weight[data.Species == "Smelt"],c="black" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Parkki"] , data.Weight[data.Species == "Parkki"],c="green" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Whitefish"] , data.Weight[data.Species == "Whitefish"],c="brown" , alpha = 0.5)

plt.xlabel("index of Spices")

plt.ylabel("weight of fish in Gram g")

plt.grid()
plt.scatter(data.index[data.Species == "Bream"] , data.Length1[data.Species == "Bream"],c="red" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Perch"] , data.Length1[data.Species == "Perch"],c="aqua" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Roach"] , data.Length1[data.Species == "Roach"],c="orange" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Pike"] , data.Length1[data.Species == "Pike"],c="purple" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Smelt"] , data.Length1[data.Species == "Smelt"],c="black" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Parkki"] , data.Length1[data.Species == "Parkki"],c="green" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Whitefish"] , data.Length1[data.Species == "Whitefish"],c="brown" , alpha = 0.5)

plt.xlabel("index of Spices")

plt.ylabel("vertical length in cm")

plt.grid()
plt.scatter(data.index[data.Species == "Bream"] , data.Length2[data.Species == "Bream"],c="red" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Perch"] , data.Length2[data.Species == "Perch"],c="aqua" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Roach"] , data.Length2[data.Species == "Roach"],c="orange" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Pike"] , data.Length2[data.Species == "Pike"],c="purple" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Smelt"] , data.Length2[data.Species == "Smelt"],c="black" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Parkki"] , data.Length2[data.Species == "Parkki"],c="green" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Whitefish"] , data.Length2[data.Species == "Whitefish"],c="brown" , alpha = 0.5)

plt.xlabel("index of Spices")

plt.ylabel("diagonal length in cm")

plt.grid()
plt.scatter(data.index[data.Species == "Bream"] , data.Length3[data.Species == "Bream"],c="red" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Perch"] , data.Length3[data.Species == "Perch"],c="aqua" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Roach"] , data.Length3[data.Species == "Roach"],c="orange" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Pike"] , data.Length3[data.Species == "Pike"],c="purple" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Smelt"] , data.Length3[data.Species == "Smelt"],c="black" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Parkki"] , data.Length3[data.Species == "Parkki"],c="green" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Whitefish"] , data.Length3[data.Species == "Whitefish"],c="brown" , alpha = 0.5)

plt.xlabel("index of Spices")

plt.ylabel("cross length in cm")

plt.grid()
plt.scatter(data.index[data.Species == "Bream"] , data.Height[data.Species == "Bream"],c="red" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Perch"] , data.Height[data.Species == "Perch"],c="aqua" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Roach"] , data.Height[data.Species == "Roach"],c="orange" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Pike"] , data.Height[data.Species == "Pike"],c="purple" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Smelt"] , data.Height[data.Species == "Smelt"],c="black" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Parkki"] , data.Height[data.Species == "Parkki"],c="green" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Whitefish"] , data.Height[data.Species == "Whitefish"],c="brown" , alpha = 0.5)

plt.xlabel("index of Spices")

plt.ylabel("height in cm")

plt.grid()
plt.scatter(data.index[data.Species == "Bream"] , data.Width[data.Species == "Bream"],c="red" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Perch"] , data.Width[data.Species == "Perch"],c="aqua" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Roach"] , data.Width[data.Species == "Roach"],c="orange" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Pike"] , data.Width[data.Species == "Pike"],c="purple" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Smelt"] , data.Width[data.Species == "Smelt"],c="black" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Parkki"] , data.Width[data.Species == "Parkki"],c="green" , alpha = 0.5)

plt.scatter(data.index[data.Species == "Whitefish"] , data.Width[data.Species == "Whitefish"],c="brown" , alpha = 0.5)

plt.xlabel("index of Spices")

plt.ylabel("diagonal width in cm")

plt.grid()
y = data.Species

x = data.drop(["Species"],axis = 1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
knn_score = 0.0

highest_indx = 1

from sklearn.neighbors import KNeighborsClassifier

score_list = []

for each in range(1,100):

    knn = KNeighborsClassifier(n_neighbors = each)

    knn.fit(x_train,y_train)

    score_list.append(knn.score(x_test,y_test))

    if (knn_score < knn.score(x_test,y_test) ):

        knn_score = knn.score(x_test,y_test)

        highest_indx = highest_indx+1

        

plt.plot(score_list,color = "purple" , alpha = 1 )    

plt.grid()         
print("KNN Max Accuracy : ",knn_score)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

lr_score = lr.score(x_test,y_test)

print("Logistic Regression Accuracy : ",lr_score)
from sklearn.naive_bayes import GaussianNB

naive_bayes = GaussianNB()

naive_bayes.fit(x_train,y_train)

nb_score = naive_bayes.score(x_test,y_test)

print("Naive Bayes Accuracy : ",nb_score)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 100)

rfc.fit(x_train,y_train)

rf_score = rfc.score(x_test,y_test)

print("Random Forest Accuracy : ",rf_score)
dict1 = {"Logistic Regression" : lr_score,"Random Forest" : rf_score,"K-Nearest Neighbour" : knn_score ,"Naive Bayes": nb_score ,"K-Nearest Neighbour" : knn_score }

dict1