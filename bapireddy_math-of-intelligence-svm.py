# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Input data fromat [X value, Y value, bias term]
X = np.array([
    [-2,4,-1],
    [4, 1, -1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],
])

# Labels of the above  data points
y = np.array([-1,-1,1,1,1])

#Plotting the points in 2d graph
for d, sample in enumerate(X):
    #Plotting negative samples:
    if(y[d]==-1):
        mark = '_'
    else:
        mark = '+'
    plt.scatter(sample[0], sample[1], s=120, marker = mark, linewidths=2)

plt.plot([-2,6],[6,0.5])
# Now the weigth updating rule which we
# are using for svm is w  = w -n(2*lambda*w)
def svm_sgd_plot(X,Y):
    #Lets define weight of svm as w where w contains coeffecien of each data point
    w = np.zeros(len(X[0]))
    # learning rate for updating weights of svm
    lr= 1 
    epochs = 100000
    # list to track the deviations in objective function
    errors = []
    for epoch in range(1,epochs):
        error = 0
        for i,x in enumerate(X):
            if(Y[i]*np.dot(X[i],w)) < 1:
                w = w + lr * (X[i] *Y[i]) + (-2 * (1/epoch) * w) # 1/epoch is regularising term
                error = 1
            else:
                w = w + lr *(-2 *(1/epoch)*w)
        errors.append(error)
    
    plt.plot(errors,'|')
    plt.ylim(0.5,1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.ylabel('Misclassified')
    plt.show()  
    return w
w = svm_sgd_plot(X,y)
print(w)
#From the above plot we can see that the misclassifications are decreased over number of epochs
for d, sample in enumerate(X):
    #Plotting negative samples:
    if(y[d]==-1):
        mark = '_'
    else:
        mark = '+'
    plt.scatter(sample[0], sample[1], s=120, marker = mark, linewidths=2)

plt.plot([w[2]/float(w[0]),0],[0,w[2]/float(w[1])])  