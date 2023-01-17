from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np



data_train = pd.read_table("/kaggle/input/data-to-work-on-basic-examples/class2_tr.txt",header=None)

data_test = pd.read_table("/kaggle/input/data-to-work-on-basic-examples/class2_test.txt",header=None)

data_train.head()
class Perceptron:

    def __init__(self):

        self.weights=[]

        self.losser= []

        #activation function

    def activation(self,data):

        activation_val=self.weights[0]

        activation_val+=np.dot(self.weights[1:],data)

        return 1 if activation_val>=0 else 0

    

    def fit(self,X,y,lrate,epochs):

        #initializing weight vector

        self.weights=[0.0 for i in range(len(X.columns)+1)]

        #no.of iterations to train the neural network

        for epoch in range(epochs):

            print(str(epoch+1),"epoch //")

            counter = 0

            for index in range(len(X)):

                x=X.iloc[index]

                predicted=self.activation(x)

                

                self.losser.append(abs(y.iloc[index]-predicted))

                #print("Error of {}.rd data: {}".format(index,y.iloc[index-predicted]))

                #check for misclassification

                if(y.iloc[index]==predicted):

                    pass

                else:

                    counter += 1

                    #calculate the error value

                    error=y.iloc[index]-predicted

                    #updation of threshold

                    self.weights[0]=self.weights[0]+lrate*error

                    #updation of weights with delta rule

                    for j in range(len(x)):

                        self.weights[j+1]=self.weights[j+1]+lrate*error*x[j]

            print(" Errord Out ", counter)

        print("To see error of each data sample just uncomment 38th row of line")

    

    def predict(self,x_test):

        predicted=[]

        for i in range(len(x_test)):

            #prediction for test set using weights

            predicted.append(self.activation(x_test.iloc[i]))

        return predicted

    

    def accuracy(self,predicted,original):

        correct=0

        lent=len(predicted)

        for i in range(lent):

            if(predicted[i]==original.iloc[i]):

                correct+=1

        return (correct/lent)*100

    

    def getweights(self):

        return self.weights

    
x_train = data_train.loc[:,:1]

y_train = data_train.loc[:,2]

x_test = data_test.loc[:,:1]

y_test = data_test.loc[:,2]
model = Perceptron()

model.fit(x_train,y_train,0.2,6) # learning rate = 0.2 6 epochs
#%%

pred = model.predict(x_test)

pred_train = model.predict(x_train)

print("Accuracy on test data: ",model.accuracy(pred,y_test))

print('Accuracy on train data: ',model.accuracy(pred_train,y_train))

print("Weights: ", model.getweights())
from matplotlib import pyplot as plt

a = np.arange(0,220)

plt.scatter(a, x_train.loc[:,0],c='r',marker='x')

plt.scatter(a, x_train.loc[:,1],c='b',marker='o')

plt.xlabel('Samples')

plt.ylabel('Data Points')

plt.title('Veri Dagilimi')

plt.grid()

plt.show()