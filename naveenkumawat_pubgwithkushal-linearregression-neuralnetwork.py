# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np

dato = pd.DataFrame(pd.read_csv("../input/train_V2.csv"))
genData = dato.select_dtypes(include=['int64','int32','float64','float32'])
genData = genData.dropna(axis=0)
Y = genData['winPlacePerc']
genData = genData.drop(['winPlacePerc'],axis = 1).copy()
selectedFeatures = genData.columns.values.tolist()
print(selectedFeatures)
genData = np.array(genData)
for i in range(0,len(genData.T)):
  genData.T[i] = genData.T[i]+1
genData = pd.DataFrame(genData)
X = genData

print(X.head(),Y.head())
def line(m,X):
  return np.dot(m,X.T)
def cost(m,X,Y):
  return np.mean((line(m,X)-Y)**2)
Y = np.array(Y)
Y = np.reshape(Y,[1,len(Y)])
def derivative(m,X,Y):
  same = 2*(line(m,X)-Y)
  return np.dot(same,X)

iters = 100
lr = 0.000000000000009
error = 0
accPlot = []
m = np.random.randn(1,len(X.T))/1000000
for i in range(iters):
    m = m-lr*derivative(m,X,Y)
#     error.append(cost(m,X,Y))
    if(i%10==0):
        predictions = line(m,X)
        error = cost(m,X,Y)
        acc = 100 - (error/np.mean(Y**2))*100
        accPlot.append(int(acc*10))
        print("Accuracy: ",acc,"%", 'Iter: ',i)
    #     if(i%10==0):
    #         print("Error: ",error[i],"Iter: ",i)
  
import matplotlib.pyplot as plt
plt.plot(accPlot)
predictions = line(m,X)
error = cost(m,X,Y)
acc = 100 - (error/np.mean(Y**2))*100
print("Accuracy: ",acc,"%")
testData = pd.DataFrame(pd.read_csv("../input/test_V2.csv"))
ID = testData['Id']
testData = testData[selectedFeatures]
testData = np.array(testData)
for i in range(0,len(testData.T)):
  testData.T[i] = testData.T[i]+1
testingPrediction = line(m,testData)
testingPrediction.shape
testingPrediction = np.reshape(testingPrediction,[len(testingPrediction.T)])
subDataFrame = pd.DataFrame({
    'Id':ID,
    'winPlacePerc':testingPrediction
})
subDataFrame.to_csv("PUBGSubmission.csv",encoding='utf-8',index=False)
subDataFrame
test = pd.DataFrame(pd.read_csv("PUBGSubmission.csv"))
print(Y)
import tensorflow as tf

NumNeurons = len(X.T)
DataIn = tf.placeholder(tf.float64,[None,NumNeurons])
Label = tf.placeholder(tf.float64,[None,1])
W1 = tf.Variable(np.random.randn(NumNeurons,NumNeurons))
W1Out = tf.nn.sigmoid(tf.matmul(DataIn,W1))
W2 = tf.Variable(np.random.randn(NumNeurons,NumNeurons))
W2Out = tf.nn.sigmoid(tf.matmul(W1Out,W2))
W3 = tf.Variable(np.random.randn(NumNeurons,NumNeurons))
W3Out = tf.nn.sigmoid(tf.matmul(W2Out,W3))
W4 = tf.Variable(np.random.randn(NumNeurons,NumNeurons))
W4Out = tf.nn.sigmoid(tf.matmul(W3Out,W4))
W5 = tf.Variable(np.random.randn(NumNeurons,NumNeurons))
W5Out = tf.nn.sigmoid(tf.matmul(W4Out,W5))
W6 = tf.Variable(np.random.randn(NumNeurons,NumNeurons))
W6Out = tf.nn.sigmoid(tf.matmul(W5Out,W6))
W7 = tf.Variable(np.random.randn(NumNeurons,NumNeurons))
W7Out = tf.nn.sigmoid(tf.matmul(W6Out,W7))
W8 = tf.Variable(np.random.randn(NumNeurons,NumNeurons))
W8Out = tf.nn.sigmoid(tf.matmul(W7Out,W8))
W9 = tf.Variable(np.random.randn(NumNeurons,NumNeurons))
W9Out = tf.nn.sigmoid(tf.matmul(W8Out,W9))
W10 = tf.Variable(np.random.randn(NumNeurons,NumNeurons))
W10Out = tf.nn.sigmoid(tf.matmul(W9Out,W10))
W11 = tf.Variable(np.random.randn(NumNeurons,NumNeurons))
W11Out = tf.nn.sigmoid(tf.matmul(W10Out,W11))
W12 = tf.Variable(np.random.randn(NumNeurons,NumNeurons))
W12Out = tf.nn.sigmoid(tf.matmul(W11Out,W12))
W13 = tf.Variable(np.random.randn(NumNeurons,NumNeurons))
W13Out = tf.nn.sigmoid(tf.matmul(W12Out,W13))
W14 = tf.Variable(np.random.randn(NumNeurons,1))
FinalOut = tf.nn.sigmoid(tf.matmul(W13Out,W14))
print(FinalOut.shape,Label.shape)
CostFunction = tf.reduce_mean(tf.square(FinalOut-Label))
Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(CostFunction)
Init = tf.global_variables_initializer()
X = pd.DataFrame(X)
Y = pd.DataFrame(Y)
Y = Y.T
with tf.Session() as S:
    Epochs = 900
    S.run(Init)
    error = []
    for i in range(Epochs):
        RandomIndice = np.random.randint(0,len(X),size = 20000)
        Data2Feed = {DataIn:np.array(X.iloc[RandomIndice]),
                    Label:np.array(Y.iloc[RandomIndice]).reshape(20000,1)}
        S.run(Optimizer,feed_dict = Data2Feed)
        CE = S.run([CostFunction],feed_dict=Data2Feed)
        if(i%100==0):
            print("Error: ",CE,"Iters: ",i)
        error.append(CE)
        
        TrainedThetas = S.run([W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14])
        
plt.plot(error)
plt.show()

def sigmoid(x):
    return 1/(1+np.exp(x))
HL1Out = sigmoid(np.dot(testData,TrainedThetas[0]))
HL2Out = sigmoid(np.dot(HL1Out,TrainedThetas[1]))
HL3Out = sigmoid(np.dot(HL2Out,TrainedThetas[2]))
HL4Out = sigmoid(np.dot(HL3Out,TrainedThetas[3]))
HL5Out = sigmoid(np.dot(HL4Out,TrainedThetas[4]))
HL6Out = sigmoid(np.dot(HL5Out,TrainedThetas[5]))
HL7Out = sigmoid(np.dot(HL6Out,TrainedThetas[6]))
HL8Out = sigmoid(np.dot(HL7Out,TrainedThetas[7]))
HL9Out = sigmoid(np.dot(HL8Out,TrainedThetas[8]))
HL10Out = sigmoid(np.dot(HL9Out,TrainedThetas[9]))
HL11Out = sigmoid(np.dot(HL10Out,TrainedThetas[10]))
HL12Out = sigmoid(np.dot(HL11Out,TrainedThetas[11]))
HL13Out = sigmoid(np.dot(HL12Out,TrainedThetas[12]))
FOut = sigmoid(np.dot(HL13Out,TrainedThetas[13]))

print(np.shape(FOut))
testingPrediction = FOut
testingPrediction = np.reshape(testingPrediction,[len(testingPrediction)])
subDataFrame = pd.DataFrame({
    'Id':ID,
    'winPlacePerc':testingPrediction
})

subDataFrame.to_csv("PUBGSubmissionNeuralNet.csv",encoding='utf-8',index=False)
Y.head()
import numpy as np
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, Y)
llSKPrd = reg.predict(testData)
testingPrediction = llSKPrd
testingPrediction = np.reshape(testingPrediction,[len(testingPrediction)])
subDataFrame = pd.DataFrame({
    'Id':ID,
    'winPlacePerc':testingPrediction
})

subDataFrame.to_csv("PUBGSubmissionLinRegSL.csv",encoding='utf-8',index=False)
