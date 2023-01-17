import numpy as np
import matplotlib.pyplot as plt
import random
%matplotlib inline

X = range(1000)
#changing our list to numpy array to benifit from numpy's broadcasting
X = np.asarray(X)
w1 = 2
b  = 500
def line_function(X):
    y = w1 * X + b
    return y
y = line_function(X)

plt.plot(X,y)

C = range(1000)
C = np.asarray(C)
w1 = 9/2
b  = 32
F = line_function(C)

fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(111)
ax.set_title('change of F with respect to C')
# ax.scatter(x=data[:,0],y=data[:,1],label='Data')
plt.plot(C,F)
ax.set_xlabel('Celsius (C)')
ax.set_ylabel('Fahrenheit (F)')
plt.show()
X1 = range(1000)
X2 = range(1000)
X1 = np.asarray(X1)
X2 = np.asarray(X2)
w1 = 8
w2 = 6
Ws = [w1,w2]
b  = 500
def hyperplane(Xs):
    y=b
    for (w,x) in zip(Ws,Xs):
        y+=w*x
    return y
ax = plt.axes(projection='3d')
xv, yv = np.meshgrid(X1, X2)
Xs = [xv, yv]
y = hyperplane(Xs)
ax.plot_surface(xv, yv,y);


import pandas as pd
df = pd.read_csv('../input/random-linear-regression/train.csv')
plt.scatter(df.x,df.y,s = 4) 
import pandas as pd
df = pd.read_csv('../input/random-linear-regression/train.csv')
plt.scatter(df.x,df.y,s = 4) 
X= range(100)
Y= X
plt.plot(X,Y,c='red')
plt.scatter(df.x,df.y,s = 4) 
X= np.asarray((range(100)))
Y= -X
plt.plot(X,Y,c='red')
#loading data from the data set 
import pandas as pd
df = pd.read_csv('../input/random-linear-regression/train.csv')
df = df.dropna()
plt.scatter(df.x,df.y,s = 4) 
def getXYfromDF(df):
    X = []
    for i in list(df.x):
        if(type(i)!=list):
            i = [i]
        i.append(1)
        X.append(i)
    X = np.asarray(X)
    y = np.asarray(df.y)
    return X,y
def randomWeights(m):
    w= []
    for i in range(m):
        w.append(random.randint(1,9))
    w = np.asarray(w)
    return w


X,y = getXYfromDF(df) 


n = X.shape[0]
m = X.shape[1]

w = randomWeights(m)

#finding the best wights analytically
w = np.dot(np.dot(np.linalg.inv(np.dot((X.T),X)),(X.T)),y) 
w
def plotTheLineWithData(X,w):
    plt.scatter(df.x,df.y,s = 4) 
    #this X is to generate test samples
    X=[]
    for i in range(100):
        X.append([i,1])
    X = np.asarray(X)
    predicted_y = np.dot(X,w) 
    plt.plot(X[:,0],predicted_y,c='red')
plotTheLineWithData(X,w)
X,y = getXYfromDF(df) 
w = randomWeights(m)

def MSE(y,y_predicted):
    return ((y- y_predicted)**2).mean()

def gradient_descent(X,y,w,max_iteration=1000,lr=0.00001): 
    w_history  = []
    loss_hostory = []
    for iteration in range(max_iteration):
        predicted_y = np.dot(X,w)
        loss =  MSE(y,predicted_y)
        loss = round(loss,9)
        w_history.append(w)
        loss_hostory.append(loss)
        derivative = -(2/y.shape[0])* X.dot(loss).sum()
        w = w + lr * derivative
    return w_history,loss_hostory

w_history,loss_hostory = gradient_descent(X,y,w,lr = 0.0000001)
perfect_i = loss_hostory.index(min(loss_hostory)) 
perfect_w = w_history[perfect_i]
w= perfect_w
# loss_hostory
plotTheLineWithData(X,w)
df = pd.read_csv('../input/simple-binary-classification-data/binary_classification_simple.xls')
plt.scatter(df.X1,df.X2,c= df.Y)
df = pd.read_csv('../input/simple-binary-classification-data/binary_classification_simple.xls')
plt.scatter(df.X1,df.X2,c= df.Y)
X = range(1000)
y= X
plt.plot(X,y,c='red')

X = np.asarray(range(1000))
y = -(2/3)*X 
plt.plot(X,y)
X = np.asarray(range(1000))
y = -(2/3)*X 
plt.plot(X,y)
plt.plot(800,-400,'+',c='green')
X = np.asarray(range(1000))
y = -(2/3)*X 
plt.plot(X,y)
plt.plot(400,-500,'_',c='red')
df = pd.read_csv('../input/simple-binary-classification-data/binary_classification_simple.xls')
def getXYfromDF(df):
    X = []
    for i in df[['X1','X2']].values.tolist():
        if(type(i)!=list):
            i = [i]
        i.append(1)
        X.append(i)
    X = np.asarray(X)
    y = np.asarray(df.Y)
    return X,y
def randomWeights(m):
    w= []
    for i in range(m):
        w.append(random.randint(1,9)/100)
    w = np.asarray(w)
    return w

X,y = getXYfromDF(df)
w= randomWeights(3)
print(w)



plt.scatter(df.X1,df.X2,c=df.Y)
X12 = np.column_stack((range(1000),np.ones(1000)))
print(X12.shape)
print(w[1:2].shape)
y0 =  -np.divide(np.dot(X12,w[1:3]),w[0])
res = [sub[0] for sub in X12] 
X1 = res
plt.plot(X1,y0)
def equales(list1,list2):
    if(len(list1)!=len(list2)):
        return False
    else: 
        for i in range(len(list1)):
            if(list1[i]!=list2[i]):
                return False
    return True
def perceptron(X,y,w,learning_rate = 0.0001,max_iterations= 1000):
    for iteration in range(max_iterations):
        prev_w = w
        for i in range(w.shape[0]):
            if(np.dot(np.dot(X[i],w),y[i]) < 0 and y[i]<0):
                w=w- learning_rate * X[i]
                
            elif(np.dot(np.dot(X[i],w),y[i]) < 0 and y[i]>0):
                w=w+ learning_rate * X[i]
        if(equales(prev_w,w)):
            print('prev_w == w in ',iteration)
            break
        
        
    return w
new_w = perceptron(X,y,w,learning_rate=0.000001,max_iterations= 100000)
w= new_w
plt.scatter(df.X1,df.X2,c=df.Y)
X12 = np.column_stack((range(1000),np.ones(1000)))
print(X12.shape)
print(w[1:2].shape)
y0 =  -np.divide(np.dot(X12,w[1:3]),w[0])
res = [sub[0] for sub in X12] 
X1 = res
plt.plot(X1,y0)
def softmax(z):
    e_z = np.exp(z)
    return e_z / e_z.sum()
z = (3,12,-5,0,10) 
np.round(softmax(z),1)
#load the data 
df = pd.read_csv('../input/simple-binary-classification-data/binary_classification_simple.xls')
#this function changes the output from numerical values to one hot vectors
#example:
#[1,-1,1] becomes: 
#[[0,1],[1,0],[0,1]
#this function is not general it's just for the case if the data has -1 and 1 cases only,
#the generalized version will be coded next.
def getOneHot(y):
    newY = []
    for i in range(y.shape[0]):
        if(y[i]==-1):
            newY.append([1,0])
        else:
            newY.append([0,1])
    return np.asarray(newY)
#this function loads the data to X and y vectors
def getXYfromDF(df):
    X = []
    for i in df[['X1','X2']].values.tolist():
        if(type(i)!=list):
            i = [i]
        i.append(1)
        X.append(i)
    X = np.asarray(X)
    y = np.asarray(df.Y)
    return X,y
#this function generates random weights to initailize the weights(+ biases ofcourse)
def randomWeights(m,k):
    w= []
    for i in range(m):
        temp = []
        for j in range(k):
            temp.append(random.randint(1,9))
        w.append(temp)
    w = np.asarray(w)
    return w


X,y = getXYfromDF(df) 
y = getOneHot(y)
X.shape
y.shape
n = X.shape[0] #number of data samples
m = X.shape[1] #number of features for each sample 
k = 2 #number of classes
w = randomWeights(m,k)
w=np.asarray(w,'float64')
w.shape
#visualize the data that we are trying to fit, and plotting the current lines with the random weights
plt.scatter(df.X1,df.X2,c=df.Y)
X12 = np.column_stack((range(1000),np.ones(1000)))
y0 =  -np.divide(np.dot(X12,w[1:3]),w[0])
res = [sub[0] for sub in X12] 
X1 = np.asarray(res)
print(X1.shape) 
print(y0.shape)
#plot the first line that represents the first linear model
plt.plot(X1,y0[:,0],c='blue')
#plot the second line that represents the second linear model
plt.plot(X1,y0[:,1],c='red')
plt.show

#the softmax that we use previously was right, but numerically it wasn't stable
#this 'edited softmax' is more numerically stable 
def softmax(x):
    temp = np.exp(x - np.max(x))  # for numerical stability
    return temp / temp.sum(axis=0)

EPS = 1e-9
#same as in softmax,the first line in this function just gives numerical stability for cross entropy 
def cross_entropy(y, y_hat):
    y_hat = np.clip(y_hat, EPS, 1-EPS) # for numerical stability
    return -np.sum(y * np.log(y_hat)/n)

history = [] #loss history 
numberOfRounds =1000 # max number of times the optimization algorithm will run
learningRate = 0.1
for _ in range(numberOfRounds):
    z = np.dot(X,w)
    y_hat = []
    for i in range(n):
        y_hat.append(softmax(z[i]))
    y_hat = np.asarray(y_hat)
    history.append(cross_entropy(y,y_hat))

    for j in range(k):
        deltaTemp=0 
        #deltaTemp is the loss derivative , and we aggregate it from all n samples (in the simple form of gradient descent,
        #and it works fine in case of offline training and smalle number of samples) 
        for i in range(n):
            deltaTemp += np.dot(X.T,(y-y_hat))
        deltaTemp  = - deltaTemp/n
        deltaTemp = np.asarray(deltaTemp)
        w-=learningRate*deltaTemp
plt.plot(history)
plt.title('the change of loss with iterations')
plt.scatter(df.X1,df.X2,c=df.Y)
X12 = np.column_stack((range(1000),np.ones(1000)))
print(X12.shape)
print(w[1:2].shape)
y0 =  -np.divide(np.dot(X12,w[1:3]),w[0])
res = [sub[0] for sub in X12] 
X1 = res
plt.plot(X1,y0)
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target
X.shape
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
currNum = 0
for i in range(2):
    for j in range(3):
        img = X[currNum,0:64]
        currNum += 1
        img = np.array(img, dtype='float')
        pixels = img.reshape((8, 8))
        ax[i, j].imshow(pixels, cmap='gray')
#simple functions to add 1 to each sample's vector for the bias
def add_bias(X):
    newX = [] 
    for i in range(X.shape[0]):
        newX.append(np.append(X[i],1))
    return np.asarray(newX)
X = add_bias(X)
X.shape

#change the form of the target values from a single digit to a onehot so we can apply out algorithm
targets=[0,1,2,3,4,5,6,7,8,9]
def oneHot(y,targets):
    newY = []
    for i in range(y.shape[0]): 
        temp = [] 
        for j in targets:
            if(y[i]==targets[j]):
                temp.append(1)
            else:
                temp.append(0)
        newY.append(temp)
    return np.asarray(newY)
y = oneHot(y,targets)

n = X.shape[0] #number of data samples
m = X.shape[1] #number of features for each sample 
k = 10 #number of classes
w = randomWeights(m,k)
w=np.asarray(w,'float64')
w.shape
history = []
maxNumOfIterations = 20
for iteration in range(maxNumOfIterations): 
    print('iteration: ',iteration)
    z = np.dot(X,w)
    y_hat = []
    for i in range(n):
        y_hat.append(softmax(z[i]))
    y_hat = np.asarray(y_hat)
    history.append(cross_entropy(y,y_hat))
    for j in range(k):
        deltaTemp=0
        for i in range(n):
            deltaTemp += np.dot(X.T,(y-y_hat))
        deltaTemp  = - deltaTemp/n
        deltaTemp = np.asarray(deltaTemp)
        w-=0.1*deltaTemp
plt.plot(history)
plt.title('the change of loss with iterations')
def giveMeValueFromOneHot(y_hat):
    return np.where(y_hat == 1)[0][0]
def predictDis(x):
    img = np.array(x, dtype='float')
    pixels = x[0:64].reshape((8, 8))
    plt.imshow(pixels, cmap='gray')
    z = np.dot(x,w)
    print ("the class of this Image is: ",giveMeValueFromOneHot(softmax(z)))
x = X[0]
predictDis(x)
x = X[1]
predictDis(x)
x = X[3]
predictDis(x)








