# # -*- coding: utf-8 -*-

"""

Created on Sat Mar 14 15:25:44 2020



@author: Aditya Anand

"""

### IMPORTING LIBRARIES ###

import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

import math



### DATA PREPROCESSING to generate clean_train_data.csv and clean_test_data.csv ###

'''

### Importing data ### 

train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")



### Feature Selection 

train = train[["Brand", "Capacity", "Internal Memory", "RAM", "Resolution", "Screen Size", "Rating"]]

test = test[["Brand", "Capacity", "Internal Memory", "RAM", "Resolution", "Screen Size"]]



# Dropping row no 97 in train since it has lots of missing data

train = train.drop(97)



### Filling Missing data

 "Note: By observing data i found that missing data are more in row of smaller phones, i.e phones with very less or no RAM , Internal Memory, Resolution etc

 ,and train data set and missing data are not that much, so filling data by mean or median strategy is not appropriate, therefor i filled missing data manually"

# We can also use Mean strategy

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(train)

train = imputer.transform(train)

imputer = imputer.fit(test)

test = imputer.transform(test)



### Cleaning columns

for col in [Capacity", "Internal Memory", "RAM", "Resolution", "Screen Size"]:

    for i in range(len(train[col])):

        p=train[col][i].split()

        if p[1]=='GB':

            train[col][i]=p[0]

        else:

            train[col][i]=p[0]/1000

    

    for i in range(len(test[col])):

        p=test[col][i].split()

        if p[1]=='GB':

            test[col][i]=p[0]

        else:

            test[col][i]=p[0]/1000 

        

### Label encoding Brand column

labelencoder = LabelEncoder()

train["Brand"] = labelencoder.fit_transform(train["Brand"])

test["Brand"] = labelencoder.fit_transform(test["Brand"])



            

### Binarizing Rating column

y=[]

for i in range(len(train["Rating"])):

    if train["Rating"][i]>=4:

        y.append(1)

    else:

        y.append(0)

train["Rating"] = np.array(y) 



### Converting to csv        

train.to_csv('clean_train_data.csv',index=False)

test.to_csv('clean_test_data.csv',index=False)



'''





### Using clean_train_data and clean_test_data to make model, train it, and generate predictions ###



### FETCHING DATA ###

train = pd.read_csv("clean_train_data.csv")

test  = pd.read_csv("clean_test_data.csv")



### Train Test Split , Scaling , X Y Split  ###

X= train.iloc[:,0:6]

Y= train.iloc[:,6]       

scaler = StandardScaler()

X = scaler.fit_transform(X)

test  = scaler.transform(test)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 1, stratify =Y)





### Class NEURAL NETWORK ###

class ANN:

    def __init__(self):

        self.weight = np.random.randn(7,7)*np.sqrt(2/7)

      

    def Hypothesis(self,W,X):

        return 1/(1+math.exp(-np.dot(W,X)))

    

    def Predict(self,X,threshold):

        Y = []

        for x in X:

            x = np.append(x,[1])

            l1 = np.ones(7)

            for i in range(6):

                l1[i] = self.Hypothesis(self.weight[i],x)

            y_pred = self.Hypothesis(self.weight[6],l1)

            if y_pred>=threshold:

                Y.append(1)

            else:

                Y.append(0)

        return np.array(Y)

                    

    def Loss(self, y_pred, y):

        return y*np.log(y_pred) + (1 - y)*np.log(1 - y_pred)

    

    def Fit(self, X,Y,epochs,learning_rate,threshold):

        loss_matrix = []

       

        for i in range(epochs):

            loss_per_epoch = 0

            for x, y in zip(X, Y):

                layer_1 = np.ones(7)

                ### FEED FORWARD ###

                # For layer 1

                x = np.append(x,[1])

                for i in range(6):

                    layer_1[i] = self.Hypothesis(self.weight[i],x)

                # For output 

                y_pred = self.Hypothesis(self.weight[6],layer_1)

                ### LOSS ###

                loss = self.Loss(y_pred,y)

                loss_per_epoch += loss

                

                ### BACK PROPAGATION or adjusting weights ###

                

                self.weight[6][0] = self.weight[6][0] - learning_rate*(y_pred - y)*(layer_1[0])

                self.weight[6][1] = self.weight[6][1] - learning_rate*(y_pred - y)*(layer_1[1])

                self.weight[6][2] = self.weight[6][2] - learning_rate*(y_pred - y)*(layer_1[2])

                self.weight[6][3] = self.weight[6][3] - learning_rate*(y_pred - y)*(layer_1[3])

                self.weight[6][4] = self.weight[6][4] - learning_rate*(y_pred - y)*(layer_1[4])

                self.weight[6][5] = self.weight[6][5] - learning_rate*(y_pred - y)*(layer_1[5])

                self.weight[6][6] = self.weight[6][6] - learning_rate*(y_pred - y)*(layer_1[6])

                

                self.weight[5][0] = self.weight[5][0] - learning_rate*(y_pred - y)*(self.weight[6][5])*(layer_1[5]-(layer_1[5])**2)*(x[0])

                self.weight[5][1] = self.weight[5][1] - learning_rate*(y_pred - y)*(self.weight[6][5])*(layer_1[5]-(layer_1[5])**2)*(x[1])

                self.weight[5][2] = self.weight[5][2] - learning_rate*(y_pred - y)*(self.weight[6][5])*(layer_1[5]-(layer_1[5])**2)*(x[2])

                self.weight[5][3] = self.weight[5][3] - learning_rate*(y_pred - y)*(self.weight[6][5])*(layer_1[5]-(layer_1[5])**2)*(x[3])

                self.weight[5][4] = self.weight[5][4] - learning_rate*(y_pred - y)*(self.weight[6][5])*(layer_1[5]-(layer_1[5])**2)*(x[4])

                self.weight[5][5] = self.weight[5][5] - learning_rate*(y_pred - y)*(self.weight[6][5])*(layer_1[5]-(layer_1[5])**2)*(x[5])

                self.weight[5][6] = self.weight[5][6] - learning_rate*(y_pred - y)*(self.weight[6][5])*(layer_1[5]-(layer_1[5])**2)*(x[6])

                

                self.weight[4][0] = self.weight[4][0] - learning_rate*(y_pred - y)*(self.weight[6][4])*(layer_1[4]-(layer_1[4])**2)*(x[0])

                self.weight[4][1] = self.weight[4][1] - learning_rate*(y_pred - y)*(self.weight[6][4])*(layer_1[4]-(layer_1[4])**2)*(x[1])

                self.weight[4][2] = self.weight[4][2] - learning_rate*(y_pred - y)*(self.weight[6][4])*(layer_1[4]-(layer_1[4])**2)*(x[2])

                self.weight[4][3] = self.weight[4][3] - learning_rate*(y_pred - y)*(self.weight[6][4])*(layer_1[4]-(layer_1[4])**2)*(x[3])

                self.weight[4][4] = self.weight[4][4] - learning_rate*(y_pred - y)*(self.weight[6][4])*(layer_1[4]-(layer_1[4])**2)*(x[4])

                self.weight[4][5] = self.weight[4][5] - learning_rate*(y_pred - y)*(self.weight[6][4])*(layer_1[4]-(layer_1[4])**2)*(x[5])

                self.weight[4][6] = self.weight[4][6] - learning_rate*(y_pred - y)*(self.weight[6][4])*(layer_1[4]-(layer_1[4])**2)*(x[6])

                

                self.weight[3][0] = self.weight[3][0] - learning_rate*(y_pred - y)*(self.weight[6][3])*(layer_1[3]-(layer_1[3])**2)*(x[0])

                self.weight[3][1] = self.weight[3][1] - learning_rate*(y_pred - y)*(self.weight[6][3])*(layer_1[3]-(layer_1[3])**2)*(x[1])

                self.weight[3][2] = self.weight[3][2] - learning_rate*(y_pred - y)*(self.weight[6][3])*(layer_1[3]-(layer_1[3])**2)*(x[2])

                self.weight[3][3] = self.weight[3][3] - learning_rate*(y_pred - y)*(self.weight[6][3])*(layer_1[3]-(layer_1[3])**2)*(x[3])

                self.weight[3][4] = self.weight[3][4] - learning_rate*(y_pred - y)*(self.weight[6][3])*(layer_1[3]-(layer_1[3])**2)*(x[4])

                self.weight[3][5] = self.weight[3][5] - learning_rate*(y_pred - y)*(self.weight[6][3])*(layer_1[3]-(layer_1[3])**2)*(x[5])

                self.weight[3][6] = self.weight[3][6] - learning_rate*(y_pred - y)*(self.weight[6][3])*(layer_1[3]-(layer_1[3])**2)*(x[6])

                

                self.weight[2][0] = self.weight[2][0] - learning_rate*(y_pred - y)*(self.weight[6][2])*(layer_1[2]-(layer_1[2])**2)*(x[0])

                self.weight[2][1] = self.weight[2][1] - learning_rate*(y_pred - y)*(self.weight[6][2])*(layer_1[2]-(layer_1[2])**2)*(x[1])

                self.weight[2][2] = self.weight[2][2] - learning_rate*(y_pred - y)*(self.weight[6][2])*(layer_1[2]-(layer_1[2])**2)*(x[2])

                self.weight[2][3] = self.weight[2][3] - learning_rate*(y_pred - y)*(self.weight[6][2])*(layer_1[2]-(layer_1[2])**2)*(x[3])

                self.weight[2][4] = self.weight[2][4] - learning_rate*(y_pred - y)*(self.weight[6][2])*(layer_1[2]-(layer_1[2])**2)*(x[4])

                self.weight[2][5] = self.weight[2][5] - learning_rate*(y_pred - y)*(self.weight[6][2])*(layer_1[2]-(layer_1[2])**2)*(x[5])

                self.weight[2][6] = self.weight[2][6] - learning_rate*(y_pred - y)*(self.weight[6][2])*(layer_1[2]-(layer_1[2])**2)*(x[6])

                

                self.weight[1][0] = self.weight[1][0] - learning_rate*(y_pred - y)*(self.weight[6][1])*(layer_1[1]-(layer_1[1])**2)*(x[0])

                self.weight[1][1] = self.weight[1][1] - learning_rate*(y_pred - y)*(self.weight[6][1])*(layer_1[1]-(layer_1[1])**2)*(x[1])

                self.weight[1][2] = self.weight[1][2] - learning_rate*(y_pred - y)*(self.weight[6][1])*(layer_1[1]-(layer_1[1])**2)*(x[2])

                self.weight[1][3] = self.weight[1][3] - learning_rate*(y_pred - y)*(self.weight[6][1])*(layer_1[1]-(layer_1[1])**2)*(x[3])

                self.weight[1][4] = self.weight[1][4] - learning_rate*(y_pred - y)*(self.weight[6][1])*(layer_1[1]-(layer_1[1])**2)*(x[4])

                self.weight[1][5] = self.weight[1][5] - learning_rate*(y_pred - y)*(self.weight[6][1])*(layer_1[1]-(layer_1[1])**2)*(x[5])

                self.weight[1][6] = self.weight[1][6] - learning_rate*(y_pred - y)*(self.weight[6][1])*(layer_1[1]-(layer_1[1])**2)*(x[6])

                

                self.weight[0][0] = self.weight[0][0] - learning_rate*(y_pred - y)*(self.weight[6][0])*(layer_1[0]-(layer_1[0])**2)*(x[0])

                self.weight[0][1] = self.weight[0][1] - learning_rate*(y_pred - y)*(self.weight[6][0])*(layer_1[0]-(layer_1[0])**2)*(x[1])

                self.weight[0][2] = self.weight[0][2] - learning_rate*(y_pred - y)*(self.weight[6][0])*(layer_1[0]-(layer_1[0])**2)*(x[2])

                self.weight[0][3] = self.weight[0][3] - learning_rate*(y_pred - y)*(self.weight[6][0])*(layer_1[0]-(layer_1[0])**2)*(x[3])

                self.weight[0][4] = self.weight[0][4] - learning_rate*(y_pred - y)*(self.weight[6][0])*(layer_1[0]-(layer_1[0])**2)*(x[4])

                self.weight[0][5] = self.weight[0][5] - learning_rate*(y_pred - y)*(self.weight[6][0])*(layer_1[0]-(layer_1[0])**2)*(x[5])

                self.weight[0][6] = self.weight[0][6] - learning_rate*(y_pred - y)*(self.weight[6][0])*(layer_1[0]-(layer_1[0])**2)*(x[6])

                

                

            loss_matrix.append(loss_per_epoch)

        return loss_matrix

    

### Object of ANN

ann = ANN()    

 

### TUNING VARIABLES ###

threshold = 0.5

learning_rate = 0.24

epochs = 200

         

loss_matrix = ann.Fit(X_train,Y_train,epochs,learning_rate,threshold)

output = ann.Predict(test,threshold)           

train_accuracy = accuracy_score(ann.Predict(X_train,threshold),Y_train) 

test_accuracy = accuracy_score(ann.Predict(X_test,threshold),Y_test) 

print(train_accuracy) 

print(test_accuracy) 



### Plotting loss matrix

plt.plot(range(epochs), loss_matrix, 'r-')



### Converting output to desired csv format 

output = pd.DataFrame(output)

phone_id = pd.read_csv("test.csv")

phone_id = phone_id["PhoneId"]

output = pd.concat([phone_id, output], axis = 1)

output.to_csv("attempt.csv", header = ["PhoneId", "Class"], index = False)