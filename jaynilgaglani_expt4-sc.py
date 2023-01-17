# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import matplotlib.pyplot as plt
class WinnerTakeAll:
    def __init__(self, X, W, alpha, max_epoch):
        # self.W = np.array([[0.7071,-0.7071],[0.7071,0.7071],[-1,0]])
        self.W = W
        self.X = X
        # self.e = np.array([[0,0,0]])
        self.epochs = max_epoch
        self.alpha = alpha
        self.patterns = ["P4","P3","P2","P1","P6","P5"]        #Actual Patterns
        self.center=["C1","C2","C3"]
        
    def predict(self, P):
        max_net = float('-inf')
        winner = 0
        for i in range(0,self.W.shape[0]):
            net = (self.W[i]).T.dot(P)
            if max_net<net:
                max_net = net
                winner = i
        return winner,net           

    def train(self):
        print("Initial Weights:")
        print(self.W)
        for e in range(self.epochs):
            print("============================\n")
            print("Epoch "+str(e+1))
            net=[]
            output=[]
            curr_pattern=1
            for pattern in self.X:
                w,net = self.predict(pattern)
                self.W[w]=self.W[w]+ (self.alpha) * (pattern-self.W[w])
                print("Winner neuron for pattern "+str(curr_pattern)+" is "+str(w+1))
                print("Updated weights of winner neuron: "+str(self.W[w]))
                print()
                curr_pattern=curr_pattern+1

            fig, ax = plt.subplots()
            x=np.concatenate((self.W[:,0],self.X[:,0]),axis=0)
            y=np.concatenate((self.W[:,1],self.X[:,1]),axis=0)
            ax.scatter(x,y)
            i=0
            for center in self.center:
                ax.annotate(center, (x[i], y[i]),color="red")
                i=i+1

            for pattern in self.patterns:
                ax.annotate(pattern, (x[i], y[i]))
                i=i+1
            
            plt.show()
        print("Final weights:")
        print(self.W)
            
# Problem
W1 = np.array([0.7071, -0.7071])
W2 = np.array([0.7071, 0.7071])
W3 = np.array([-1, 0]) 


P1 = np.array([-0.1961, 0.9806])
P2 = np.array([0.1961, 0.9806])
P3 = np.array([0.9806, 0.1961])
P4 = np.array([0.9806, -0.1961])
P5 = np.array([-0.5812, -0.8137])
P6 = np.array([-0.8137, -0.5812])

P = np.array([P4, P3, P2, P1, P6, P5])
W = np.array([W1, W2, W3])

alpha = 0.5
max_epoch = 10
winnerTakeAll = WinnerTakeAll(P, W, alpha, max_epoch)
winnerTakeAll.train()
class WinnerTakeAll:
    def __init__(self, max_epoch, W, P, c):
        self.max_epoch = max_epoch
        self.W = W
        self.c = c
        self.P = P

    def showOutput(self, X, W, net, del_W):
        print("Input: ", X)
        print("Net: ", net)
        print("Del_W: ", del_W)
        print("Weights: ", W)
    
    def plotGraph(self):
        plt.figure(figsize=(7,6))
        plt.plot(self.W)
        plt.ylim([-1.0,2.5])
        plt.xlim([-1.0,2.5])
        plt.grid()
        plt.legend(loc='upper right')
        plt.show()

    def train(self):
        for epoch in range(max_epoch):
            print("Epoch no.: ", epoch+1)
            for i in range(self.P.shape[0]):
                max_net = 0
                curr_Input = self.P[i]
                curr_neuron_number = 0
                del_curr_neuron = 0
#                 updatedWeights = []
                for j in range(self.W.shape[0]):
                    net = np.dot(self.W[j].T, curr_Input)
                    e = curr_Input - self.W[j].T
                    if(net > max_net):
                        max_net = net
                        curr_neuron_number = j
                        del_curr_neuron = c * e
                print("Winner neuron is: ", curr_neuron_number+1)
                self.W[curr_neuron_number] = self.W[curr_neuron_number] + del_curr_neuron
                self.showOutput(curr_Input, self.W[curr_neuron_number], max_net, del_curr_neuron)
            self.plotGraph()
            
            
                