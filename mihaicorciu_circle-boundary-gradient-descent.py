import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



#data = pd.read_csv('test.csv', header=None)

#X = np.array(data[[0,1]])

#pointType = np.array(data[2])



#array data points: x1, x2

data = np.array([

    [5,8,1],

    [4,7,1],

    [4,8,1],

    [7,7,0],

    [6.5,7.2,0],

    [5,7,0],

    [5.5,7.5,0],

    [7,8,0],



])

X = data[:, [0,1]]

pointType = data[:, [2]]

pointType = pointType.flatten()



datamin = np.min(X, axis=0)

datamax = np.max(X, axis=0)

xmin, ymin = datamin

xmax, ymax = datamax



def plot_points(X, y):

    admitted = X[np.argwhere(y==1)]

    rejected = X[np.argwhere(y==0)]

    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'blue', edgecolor = 'k',zorder=2)

    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'red', edgecolor = 'k',zorder=2)

    

phi = np.linspace(0, 2*np.pi, 200)



def circle(w1=0,w2=0,b=1,color='green'):

    x = b*np.cos(phi)

    y = b*np.sin(phi)

    plt.axis("equal")

    plt.plot(x-w1,y-w2,color,zorder=1)

  
# Activation (sigmoid) function

def sigmoid(x):

    return 1 / (1 + np.exp(-x))



def score(points,weights,bias):

    squares = (points+weights)**2

    if (squares.shape == (2,)):

        s = np.sum(squares)

    else:

        s = np.sum(squares, axis=1)

    s = -1*(s-bias)

    return s



#likelihood function

#yhat:continuous prediction function which returns [0,1] prob. instead of {0,1} discrete values

def probability(score):

    return sigmoid(score)



#0 class points have 1-p probability to be correct classified

def likelihood(y,p):

    return y*(p) + (1 - y) * (1-p)



#error function

#log_loss = log_likelihood = -1 *log(likelihood)

def log_loss(likelihood):

    return -1*np.log(likelihood)



def update(x, y, weights, bias, learnrate):

    s = score(x,weights,bias)

    p = probability(s)

    

    weights += learnrate * (-2) * (y-p) * (weights+x)

    #bias += learnrate * (y-p)

    return weights, bias       
bias = 1.2

weights = [-8,-8]



s = score(X,weights,bias)

p = probability(s)

l = likelihood(pointType,p) 

err = log_loss(l)

#print(X,weights)

#print(s)

#print(p)

#print(l)

#print(err)



epochs = 12

learnrate = 0.1

errors = []

last_loss = None





circle(weights[0],weights[1],bias,'yellow')





for e in range(epochs):

    #circle(weights[0],weights[1],bias)

    

    for x, y in zip(X, pointType):

 

        s = score(x,weights,bias)

        p = probability(s)

        l = likelihood(y,p)        

        err = log_loss(l)

        #print(s,p,l,err)

        

        #print("1",weights,bias)

        weights,bias = update(x, y, weights, bias, learnrate)

        #print("2",weights,bias)

        #print(x,weights,x.shape)

        #s = score(x,weights,bias)

        

    circle(weights[0],weights[1],bias)

    

    s = score(X,weights,bias)

    p = probability(s)

    l = likelihood(pointType,p)        

    err = log_loss(l)

    loss = np.mean(err)

    errors.append(loss) 

    

    #print(loss)

    

    #if e % (epochs / 10) == 0:

    if True:

        print("\n========== Epoch", e,"==========")

        if last_loss and last_loss < loss:

            print("Train loss: ", loss, "  WARNING - Loss Increasing")

            break

        else:

            print("Train loss: ", loss) 

        last_loss = loss    

        

circle(weights[0],weights[1],bias,'black')



more = 2

plt.xlim(xmin-more,xmax+more)

plt.ylim(ymin-more,ymax+more)

plot_points(X, pointType)