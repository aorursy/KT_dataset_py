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
class perceptron (object): 

    

    def __init__(self,no_inputs,epochs=50,lr=0.01):

        self.epochs=epochs

        self.lr = lr

        

        #we use the number of input features to create a array of 0s of that size+1

        self.weights = np.zeros(no_inputs)

       

        self.b=0

    #define a predict method for generating predictions for given input features

        

    def predict(self,inputs):

        

        

        #basic operation of a perceptron

        # input vector * weights + bias

        

        predict = np.dot(inputs,self.weights)+self.b

        

        

        #for classification :

        #print(predict)

        

        if predict >0:

            return 1 

        else:

            return 0

        

    def train(self,training_inputs,labels):

#         y1=[]

#         y2=[]

        for i in range(self.epochs):

            for inputs, label in zip(training_inputs,labels):

                

                #make prediction to reduce loss

                pred = self.predict(inputs)

                

                #pdate the weights,label-prediction = loss

                self.weights+=self.lr * (label - pred) * inputs

                

                #update the bias

                self.b+=self.lr*(label-pred)

                

#             y1.append(abs(pred-labels))

#             y2.append(i)

            

            

#         import matplotlib.pyplot as plt

#         plt.plot(y2,y1)

                
training_inputs = []

training_inputs.append(np.array([1, 1]))

training_inputs.append(np.array([1, 0]))

training_inputs.append(np.array([0, 1]))

training_inputs.append(np.array([0, 0]))



labels = np.array([1, 

                   0, 

                   0, 

                   0])





training_inputs[:]

x = perceptron(2)

x.train(training_inputs, labels)
x.predict([1,1])
x.weights


import sklearn.datasets

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



#load the breast cancer data

data = sklearn.datasets.load_breast_cancer()
df = pd.DataFrame(data.data, columns = data.feature_names)

df["TARGET"] = data.target



df.head(4)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



#label and features

X = df.drop("TARGET",axis=1)

Y = df["TARGET"]



X = scaler.fit_transform(X)

X
#re arranging the scaled data as a dataframe.



X = pd.DataFrame(X, columns=df.drop("TARGET",axis = 1).columns)



type(X.iloc[0][0])
#split data to train and test 

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, stratify = Y, random_state = 1)

#number of inputs 

number_of_inputs =X_train.shape[1]



X_train.head()
x = perceptron(number_of_inputs)

x.train(X_train.to_numpy(), Y_train.to_numpy())

preds=[]
for i in range (X_test.shape[0]):

    

    preds.append(x.predict(X_test.iloc[i].to_numpy()))
from sklearn.metrics import accuracy_score 

#checking the accuracy of the model

print(accuracy_score(preds, Y_test))