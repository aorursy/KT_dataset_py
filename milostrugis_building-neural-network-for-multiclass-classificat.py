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
import numpy as np

import pandas as pd

import random

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
Path = "/kaggle/input/iris/Iris.csv"

Path1 = "C:/Users/wilkm/Desktop/aaaa/iris1.xlsx"



iris =  pd.read_csv(Path)

iris = iris.iloc[:, 1:6]

iris.head()
# normalising features

iris_trans = iris

iris_trans.head()

data = iris_trans.iloc[:, 0:4]

values = data.values

scaler = MinMaxScaler()

print(scaler.fit(data))

MinMaxScaler(copy=True, feature_range=(0, 1))

iris_trans.iloc[:, 0:4] = scaler.transform(data)

iris_trans.Species = pd.Categorical(iris_trans.Species)

iris_trans['categ'] = iris_trans.Species.cat.codes

iris_trans.head()
iris_trans["categ"].value_counts()

# balanced dataset
#splitting data to training and test tests 

train_df, test_df = train_test_split(iris_trans, test_size = 0.25,random_state=42)
# visualising data

categ = iris_trans.iloc[:, 5]

feat = iris_trans.iloc[:, 0:2]

categ = iris_trans.iloc[:, 5]

feat = iris_trans.iloc[:, 0:2]

plt.scatter(feat.iloc[:,0], feat.iloc[:,1], c = categ)

plt.title("Iris data set")
train_x = np.array(train_df.iloc[:, 0:4])

train_x  = train_x[:, [0,1]]

test_x = np.array(test_df.iloc[:, 0:4])

test_x  = test_x[:, [0,1]]

train_y = np.array(pd.get_dummies(train_df.iloc[:, 5]))

test_y = np.array(pd.get_dummies(test_df.iloc[:, 5]))
# First attempt to bulid neural network and to use the object-oriented programming 

def sigmoid(x):

        return 1 / (1 + np.exp(-x))



def deriv_sigmoid(x):

    return sigmoid(x) * (1 - sigmoid(x))



def softmax(x):

    expx= np.exp(x)

    return expx / expx.sum(axis=1, keepdims=True)











class NeuralNetwork:

    import matplotlib.pyplot as plt

    def __init__(self, input_x, output_act, alpha, epochs):

   

        self.input_x      = input_x

        self.output_act = output_act

        

        no_of_examp = self.input_x .shape[0]

        

        self.W1   = np.random.rand(self.input_x.shape[1],4) 

        self.B1   = np.random.randn(4)            

        self.W2   = np.random.rand(4,self.output_act.shape[1]) 

        self.B2   = np.random.randn(self.output_act.shape[1])

        self.output     = np.zeros(self.output_act.shape)

        self.alpha = alpha

        self.epochs = epochs

        self.error = [] 

   

     

    

    def feedforward(self, input_x):

        self.v1 = np.dot(input_x, self.W1) + self.B1

        self.layer1 = sigmoid(self.v1)

        self.v2 = np.dot(self.layer1, self.W2) + self.B2      

        self.output = softmax(self.v2)

        return self.output

    

    def backprop(self, input_x, output_act, alpha):

        out_error = self.output - self.output_act

        der_W2 = np.dot(self.layer1.T, out_error)

        der_B2 = out_error

        

        der_W1 = np.dot(self.input_x.T, ((self.layer1 *(1-self.layer1)) * np.dot(out_error , self.W2.T)))

        der_B1 = (self.layer1 *(1-self.layer1)) * np.dot(out_error , self.W2.T)

        

        self.W1 -= alpha * der_W1

        self.B1 -= alpha * der_B1.sum(axis=0)



        self.W2 -= alpha * der_W2

        self.B2 -= alpha * der_B2.sum(axis=0) 

        #return self.W1, self.B1, self.W2, self.B2 

    def train(self, input_x, output_act, alpha, epochs):

        self.error = []

        for epoch in range(self.epochs):

            self.feedforward(input_x)

            self.backprop(input_x, output_act, alpha)

            if epoch % 200 == 0:

                loss = np.sum(-1*np.multiply(output_act, np.log(self.output)))

                #loss = loss/no_of_examp

                print('Loss function value: ', loss)

                self.error.append(loss)

        plt.plot(self.error)

        plt.title("Loss")

    def test(self, data, act_y, results = False):

        res = np.around(self.feedforward(data))

        x = 0

        for i in range(len(res)):

            if(all(act_y[i]==res[i])):

                x+= 1

        acc = round((x / len(res)), 2)

        print('Accuracy: {0:2.2f}'.format(acc))

        if results == True:

            print("Predicted values: ", "\n" ,res)

       
# training the network

new_network = NeuralNetwork(train_x, train_y, 0.01, 5000)

new_network.train(train_x, train_y, 0.01, 5000)
# print training accuracy

new_network.test(train_x, train_y)
# print test data accuracy and results

new_network.test(test_x,test_y, True)
import pandas as pd

mushrooms = pd.read_csv("../input/mushroom-classification/mushrooms.csv", header = 0)

mushrooms.head()
print( mushrooms["class"].value_counts()/mushrooms.shape[0]*100)

# looking an % values, this is a balanced data set
print(mushrooms.shape)

# there are 22 features in the set, and all features are categorical
!pip install prince
import prince
mca = prince.MCA()

mushrooms_mca_categ = mushrooms.iloc[:, 0] # extract labels

mushrooms_mca = mushrooms.iloc[:, 1:] # features

mushrooms_mca.head()
mca = mca.fit(mushrooms_mca) # same as calling ca.fs_r(1)

mca = mca.transform(mushrooms_mca) # same as calling ca.fs_r_sup(df_new) for *another* test set.

print(mca.head(20))
print(mca.head(20))
mushrooms_new = pd.concat([mca, mushrooms_mca_categ], axis=1)

mushrooms_new["class"] = pd.Categorical(mushrooms_new["class"])

mushrooms_new["class_cat"] = mushrooms_new["class"].cat.codes

mushrooms_new.head()

labels = mushrooms_new["class_cat"]# poisonus1, edible0

feat1 = mushrooms_new[0]

feat2 = mushrooms_new[1]

plt.scatter(feat1, feat2, c = labels)

plt.title("Musrooms MCA features")
mush_train, mush_test = train_test_split(mushrooms_new, test_size = 0.25, random_state = 42)

mush_train_x = np.array(mush_train.iloc[:, 0:2])

mush_train_y = np.array(pd.get_dummies(mush_train.iloc[:, 3]))

mush_test_x = np.array(mush_test.iloc[:, 0:2])

mush_test_y = np.array(pd.get_dummies(mush_test.iloc[:, 3]))
new_network2 = NeuralNetwork(mush_train_x, mush_train_y, 0.0001, 5000)
new_network2.train(mush_train_x, mush_train_y, 0.0001, 5000)
new_network2.test(mush_train_x, mush_train_y)
new_network2.test(mush_test_x, mush_test_y)