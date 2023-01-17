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
data=pd.read_csv("/kaggle/input/ufcdata/preprocessed_data.csv")

data.head()
for col in data.columns: 

    print(col) 
from sklearn import preprocessing

from sklearn.preprocessing import label_binarize



le = preprocessing.LabelEncoder()

for i in range(0,len(data.columns)):

    data.iloc[:,i] = le.fit_transform(data.iloc[:,i])

data.head()

nrows=len(data.index)

percentage=round((nrows*90)/100)

data=data.sample(frac=1, random_state=69)

trainingData=data.iloc[:percentage,:]

testData=data.iloc[percentage:,:]



print("Number of training data examples "+str(len(trainingData.index)))

print("Number of test examples "+str(len(testData.index)))
train_x=trainingData[["B_wins", "B_losses","B_draw", "R_wins", "R_losses","R_draw"]]

#train_x=trainingData[trainingData.columns.difference(['b'])]

train_y=trainingData["Winner"]



test_x=testData[["B_wins", "B_losses","B_draw", "R_wins", "R_losses","R_draw"]]

#test_x=testData[testData.columns.difference(['b'])]

test_y=testData["Winner"]
train_x.head()
train_y.head
class AbstractPerceptron :

    weights=np.array([])

    learningRate=1

    def __init__(self, learningRate):

        self.learningRate=learningRate

        

    def predict(self,x):

        xtemp=np.append(1,x)

        return self.weights.dot(xtemp)



    def getWeights(self):

        return self.weights
class Perceptron(AbstractPerceptron):

    def train(self,x,y):

        nFeatures=x.shape[1]

        nExamples=x.shape[0]

        onesColumn=np.ones([nExamples,1],dtype=int)

        xtemp=np.append(onesColumn,x,axis=1)

        np.random.seed(69)

        self.weights=np.random.rand(nFeatures+1)

        for i in range(0,nExamples):

            output=self.predict(x[i][:])

            adjustment=(self.learningRate*(y[i]-output))*xtemp[i][:]

            self.weights=(self.weights+adjustment)

    pass
class SignPerceptron(Perceptron):

    def  predict(self,x):

        predictions=super().predict(x)

        return np.sign(predictions)

    pass
class SigmoidPerceptron(Perceptron):

    def  predict(self,x):

        predictions=super().predict(x)

        sigmoid=lambda x : 1/(np.exp(x*-1)+1)

        return sigmoid(predictions)

    pass
class GradientDescentPerceptron(AbstractPerceptron):

    def train(self,x,y):

        nFeatures=x.shape[1]

        nExamples=x.shape[0]

        onesColumn=np.ones([nExamples,1],dtype=int)

        xtemp=np.append(onesColumn,x,axis=1)

        np.random.seed(69)

        self.weights=np.random.uniform(-0.5,0.5,nFeatures+1)

        deltas=np.zeros(len(self.weights))

        for i in range(0,nExamples):

            output=self.predict(x[i][:])

            for j in range(0,len(deltas)):

                deltas[j]=deltas[j]+(self.learningRate*(y[i]-output))*xtemp[i][j]

        self.weights=(self.weights+deltas)

    pass
class SigmoidGradientDescentPerceptron(AbstractPerceptron):  

    def train(self,x,y):

        sigmoid=lambda x : 1/(np.exp(x*-1)+1)

        nFeatures=x.shape[1]

        nExamples=x.shape[0]

        onesColumn=np.ones([nExamples,1],dtype=int)

        xtemp=np.append(onesColumn,x,axis=1)

        np.random.seed(69)

        self.weights=np.random.uniform(-1*10^-10,-1*10^-10,nFeatures+1)

        deltas=np.zeros(len(self.weights))

        for i in range(0,nExamples):

            output=self.predict(x[i][:])

            for j in range(0,len(deltas)):

                deltas[j]=deltas[j]+self.learningRate*(y[i]-output)*sigmoid(y[i])*(1-sigmoid(y[i]))*xtemp[i][j]

            self.weights=(self.weights+deltas)

    def predict(self,x):

        prediction=super().predict(x)

        sigmoid=lambda x : 1/(np.exp(x*-1)+1)

        return sigmoid(prediction)
from sklearn.metrics import accuracy_score

perceptron=SignPerceptron(1)

perceptron.train(train_x.to_numpy(),train_y.to_numpy())



nExamples=test_x.shape[0]

predictions=np.empty([nExamples,1])

for i in range(0,nExamples):

    predictions[i]=perceptron.predict(test_x.iloc[i][:])



print("The accuracy score of the prediction on the test data is",accuracy_score(test_y, predictions))



nExamples=train_x.shape[0]

predictions=np.empty([nExamples,1])

for i in range(0,nExamples):

    predictions[i]=perceptron.predict(train_x.iloc[i][:])

print("The accuracy score of the prediction on the train data is",accuracy_score(train_y, predictions))



print()



print("The calculated weights are ")

print(perceptron.getWeights())
perceptron=SigmoidPerceptron(1)

perceptron.train(train_x.to_numpy(),train_y.to_numpy())



nExamples=test_x.shape[0]

predictions=np.empty([nExamples,1])

for i in range(0,nExamples):

    predictions[i]=perceptron.predict(test_x.iloc[i][:])

print("The accuracy score of the prediction on the test data is",accuracy_score(test_y, predictions.round()))



nExamples=train_x.shape[0]

predictions=np.empty([nExamples,1])

for i in range(0,nExamples):

    predictions[i]=perceptron.predict(train_x.iloc[i][:])

print("The accuracy score of the prediction on the train data is",accuracy_score(train_y, predictions.round()))



print()



print("The calculated weights are ")

print(perceptron.getWeights())
perceptron=SigmoidGradientDescentPerceptron(1)

perceptron.train(train_x.to_numpy(),train_y.to_numpy())



nExamples=test_x.shape[0]

predictions=np.empty([nExamples,1])

for i in range(0,nExamples):

    predictions[i]=perceptron.predict(test_x.iloc[i][:])

print("The accuracy score of the prediction on the test data is",accuracy_score(test_y, predictions.round()))



nExamples=train_x.shape[0]

predictions=np.empty([nExamples,1])

for i in range(0,nExamples):

    predictions[i]=perceptron.predict(train_x.iloc[i][:])

print("The accuracy score of the prediction on the train data is",accuracy_score(train_y, predictions.round()))



print()



print("The calculated weights are ")

print(perceptron.getWeights())
from sklearn.linear_model import Perceptron

perceptron = Perceptron(alpha=1)



perceptron.fit(train_x, train_y)



predictions=perceptron.predict(test_x)

print("The accuracy score of the prediction on the test data is",accuracy_score(test_y, predictions.round()))



predictions=perceptron.predict(train_x)

print("The accuracy score of the prediction on the train data is",accuracy_score(train_y, predictions.round()))