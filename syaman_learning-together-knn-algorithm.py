import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#we will import our data
mydata=pd.read_csv("../input/column_2C_weka.csv")

#in our data we have 2 kind class which are Abnormal and Normal
mydata["class"].unique()

#now we will seperate our data in two parth, 
Abnormal=mydata[mydata["class"]=="Abnormal"]
Normal=mydata[mydata["class"]=="Normal"]

#visalize our data
plt.scatter(Abnormal.pelvic_incidence,Abnormal.sacral_slope,color="red",label="Abnormal",alpha=0.4)
plt.scatter(Normal.pelvic_incidence,Normal.sacral_slope,color="green",label="Normal",alpha=0.6)
plt.legend()#this command will show labels
plt.xlabel("pelvic_incidence")
plt.ylabel("pelvic_radius")
plt.show()

#now we will seperate our data in two part
mydata["class"]= [1 if each == "Abnormal" else 0 for each in mydata["class"]]

y=mydata["class"].values
x=mydata.drop(["class"],axis=1)

#normalization (x-max(x))/(max(x)-min(x))
x_normal = (x-np.max(x))/(np.max(x)-np.min(x))

#train test split
xtrain,xtest,ytrain,ytest=train_test_split(x_normal,y,test_size=0.3,random_state=21)

#KNN Model
knn=KNeighborsClassifier(n_neighbors=5) #k number is n_neighbors
knn.fit(xtrain,ytrain)
prediction=knn.predict(xtest)

print("knn k = {}, score= {}".format(5,knn.score(xtest,ytest)))
#find avaible value of k
score_list=[]
k_value=[]
for i in range(1,15):
    knn_available=KNeighborsClassifier(n_neighbors=i)
    knn_available.fit(xtrain,ytrain)
    
    score_list.append(knn_available.score(xtest,ytest))
    k_value.append(i)
plt.plot(range(1,15),score_list)
plt.xlabel("K values")
plt.ylabel("Score(accracy)")
plt.show()

#we will see when max accuracy with k value
print("k : {} and max Score: {}".format(i,np.max(score_list)))
#Model Complexity
neigboars=np.arange(1,20)

train_accuracy=[]
test_accuracy=[]

for i,k in enumerate(neigboars):
    knn=KNeighborsClassifier(n_neighbors=k)
    #fitting
    knn.fit(xtrain,ytrain)
    
    train_accuracy.append(knn.score(xtrain,ytrain))
    test_accuracy.append(knn.score(xtest,ytest))
    
#visualization

plt.figure(figsize=[15,10])
plt.plot(neigboars,test_accuracy,label="Test accuracy")
plt.plot(neigboars,train_accuracy,label="Train accuracy")
plt.legend()

plt.title("ACCURACY RATE")
plt.xlabel("Number of Neighboars")
plt.ylabel("Accuracy")
plt.xticks(neigboars)
plt.show()

#we can find best accuracy in which value of k
print("Best Accuracy is {} with k= {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
