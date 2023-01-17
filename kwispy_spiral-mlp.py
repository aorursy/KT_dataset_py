%matplotlib inline

import numpy as np

from sklearn import datasets

from sklearn import model_selection

import matplotlib.pyplot as plt 

import math

import sys

import pandas as pd




"""

Spiral generator.

Inputs:

Radius - maximum radius of the spiral from the center.

  Defines the distance of the tail end from the center.

Step - amount the current radius increases between each point.

  Larger = spiral expands faster

Resolution - distance between 2 points on the curve.

  Defines amount radius rotates between each point.

  Larger = smoother curves, more points, longer time to calculate.

Angle - starting angle the pointer starts at on the interior

Start - starting distance the radius is from the center.

"""

def spiral(radius, step, resolution=.1, angle=0.0, start=0.0):

    dist = start+0.0

    coords=[]

    while dist*math.hypot(math.cos(angle),math.sin(angle))<radius:

        cord=[]

        cord.append(dist*math.cos(angle))

        cord.append(dist*math.sin(angle))

        coords.append(cord)

        dist+=step

        angle+=resolution

    return np.array(coords)
def F_standardize(X):

    """

    standardize X, i.e. subtract mean (over data) and divide by standard-deviation (over data)

    

    Parameters

    ----------

    X: np.array of size (nbData, nbDim)

        matrix containing the observation data

    

    Returns

    -------

    X: np.array of size (nbData, nbDim)

        standardize version of X

    """

    

    X -= np.mean(X, axis=0, keepdims=True) 

    X /= (np.std(X, axis=0, keepdims=True) + 1e-16)

    return X
def F_sigmoid(x):

    """Compute the value of the sigmoid activation function"""

    return 1 / (1 + np.exp(-x))



def F_relu(x):

    """Compute the value of the Rectified Linear Unit activation function"""

    return x * (x > 0)



def F_relu2(x):

    """Compute the value of the Rectified Linear Unit activation function"""

    return x**2 * (x > 0)



def F_reluN(x,n):

    """Compute the value of the Rectified Linear Unit activation function"""

    return x**n * (x > 0)



def F_dRelu(x):

    """Compute the derivative of the Rectified Linear Unit activation function"""

    x[x<=0] = 0

    x[x>0] = 1

    return x



def F_dRelu2(x):

    """Compute the derivative of the Rectified Linear Unit activation function"""

    xcopy=x.copy()

    xcopy[xcopy<=0] = 0

    xcopy[xcopy>0] = 1

    return 2*x*xcopy



def F_dReluN(x,n):

    """Compute the derivative of the Rectified Linear Unit activation function"""

    xcopy=x.copy()

    xcopy[xcopy<=0] = 0

    xcopy[xcopy>0] = 1

    return n*x**(n-1)*xcopy



def F_dSigmoid(x):

    """Compute the derivative of the sigmoid activation function"""

    return F_sigmoid(x)*(1-F_sigmoid(x))



def F_computeCost(hat_y, y):

    """Compute the cost (sum of the losses)

    

    Parameters

    ----------

    hat_y: (1, nbData)

        predicted value by the MLP

    y: (1, nbData)

        ground-truth class to predict

    """

    m = y.shape[1]

     

    loss =  - (np.multiply(y, np.log(hat_y)) + np.multiply((1 - y), np.log(1 - hat_y)))  

    

    cost = np.sum(loss) / m

    return cost



def F_computeAccuracy(hat_y, y):

    """Compute the accuracy

    

    Parameters

    ----------

    hat_y: (1, nbData)

        predicted value by the MLP

    y: (1, nbData)

        ground-truth class to predict

    """

    

    m = y.shape[1]    

    class_y = np.copy(hat_y)

    class_y[class_y>=0.5]=1

    class_y[class_y<0.5]=0

    return np.sum(class_y==y) / m
class C_MultiLayerPerceptron:

    """

    A class used to represent a Multi-Layer Perceptron with 1 hidden layers



    ...



    Attributes

    ----------

    W1, b1, W2, b2:

        weights and biases to be learnt

    Z1, A1, Z2, A2:

        values of the internal neurons to be used for backpropagation

    dW1, db1, dW2, db2, dZ1, dZ2:

        partial derivatives of the loss w.r.t. parameters

    VdW1, Vdb1, VdW2, Vdb2:

        momentum terms

    do_bin0_multi1:

        set wether we solve a binary or a multi-class classification problem

        

    Methods

    -------

    forward_propagation

    

    backward_propagation

    

    update_parameters

    

    """



    W1, b1, W2, b2 = [], [], [], []

    Z1, A1, Z2, A2 = [], [], [], []

    dW1, db1, dW2, db2 = [], [], [], []   

    dZ1, dA1, dZ2 = [], [], []

    # --- for momentum

    VdW1, Vdb1, VdW2, Vdb2 = [], [], [], []   

    dX=[]

    n=2

    activation_function=1

    

    def __init__(self, n_in, n_h, n_out,n, activation_f=1):

        self.W1 = np.random.randn(n_h, n_in) * 0.01

        self.b1 = np.zeros(shape=(n_h, 1))

        self.W2 = np.random.randn(n_out, n_h) * 0.01

        self.b2 = np.zeros(shape=(n_out, 1))        

        # --- for momentum

        self.VdW1 = np.zeros(shape=(n_h, n_in)) 

        self.Vdb1 = np.zeros(shape=(n_h, 1))

        self.VdW2 = np.zeros(shape=(n_out, n_h))

        self.Vdb2 = np.zeros(shape=(n_out, 1))

        self.dX=np.zeros(shape=(n_in, 1))

        self.n=n

        self.activation_function=activation_f

        return



    

    def __setattr__(self, attrName, val):

        if hasattr(self, attrName):

            self.__dict__[attrName] = val

        else:

            raise Exception("self.%s note part of the fields" % attrName)



            



    def M_forwardPropagation(self, X_):

        """Forward propagation in the MLP



        Parameters

        ----------

        X: numpy array (nbDim, nbData)

            observation data



        Return

        ------

        hat_y: numpy array (1, nbData)

            predicted value by the MLP

        """

        

        # --- START CODE HERE

        

        # --- Z1 (n_h, m) = W1 (n_h, n_in) X (n_in, m) + b1 (n_h, 1)      

        #print(self.W1.shape)

        self.Z1 = np.dot(self.W1, X_) + self.b1

        if (self.activation_function==1):

            self.A1 = F_reluN(self.Z1,self.n)

        elif (self.activation_function==2):

            #print("hee")

            self.A1 = F_sigmoid(self.Z1)

        # --- Z2 (n_out, m) = W2 (n_out, n_h) A1 (n_h, m) + b2 (n_out, 1)

        self.Z2 = np.dot(self.W2, self.A1) + self.b2        

        self.A2 = F_sigmoid(self.Z2)

        

        # --- END CODE HERE

        

        hat_y = self.A2

        

        return hat_y





    def M_backwardPropagation(self, X_, y):

        """Backward propagation in the MLP



        Parameters

        ----------

        X: numpy array (nbDim, nbData)

            observation data

        y: numpy array (1, nbData)

            ground-truth class to predict

            

        """

        

        m = y.shape[1]

                

        # --- dZ2 (n_out, m) = self.A2 (n_out, m) - self.y (n_out, m)

        self.dZ2 = self.A2 - y        

        # --- dW2 (n_out, n_h) = self.dZ2 (n_out, m) self.A1.T (n_h, m).T

        self.dW2 = (1 / m) * np.dot(self.dZ2, self.A1.T)

        # --- db2 (n_out, 1) = self.dZ2 (n_out, m) 

        self.db2 = (1 / m) * np.sum(self.dZ2, axis=1, keepdims=True)

        # --- dA1 (n_h, m) = self.W2.T (n_out, n_h).T self.dZ2 (n_out, m)

        self.dA1 = np.dot(self.W2.T, self.dZ2)

        

        if (self.activation_function==1):

            self.dZ1 = np.multiply(self.dA1, F_dReluN(self.Z1,self.n))

        elif (self.activation_function==2):

            self.dZ1 = np.multiply(self.dA1, F_dSigmoid(self.Z1))

        

        # --- dW1 (n_h, n_in) = self.dZ1 (n_h, m) X.T (n_in, m).T

        self.dW1 = (1 / m) * np.dot(self.dZ1, X_.T)

        # --- db1 (n_h, 1) = self.dZ1 (n_h, m) 

        self.db1 = (1 / m) * np.sum(self.dZ1, axis=1, keepdims=True)

        

        return



    

    def M_gradientDescent(self, alpha):

        """Update the parameters of the network using gradient descent



        Parameters

        ----------

        alpha: float scalar

            amount of update at each step of the gradient descent

            

        """



        self.W1 = self.W1 - alpha * self.dW1

        self.b1 = self.b1 - alpha * self.db1 

        self.W2 = self.W2 - alpha * self.dW2 

        self.b2 = self.b2 - alpha * self.db2 

        

        return



    

    def M_momentum(self, alpha, beta):

        """Update the parameters of the network using momentum method



        Parameters

        ----------

        alpha: float scalar

            amount of update at each step of the gradient descent

        beta: float scalar

            momentum term 

        """

        

        # --- START CODE HERE

        self.VdW1 = beta * self.VdW1 + (1-beta) * self.dW1

        self.W1 = self.W1 - alpha * self.VdW1



        self.Vdb1 = beta * self.Vdb1 + (1-beta) * self.db1

        self.b1 = self.b1 - alpha * self.Vdb1

        

        self.VdW2 = beta * self.VdW2 + (1-beta) * self.dW2

        self.W2 = self.W2 - alpha * self.VdW2

        

        self.Vdb2 = beta * self.Vdb2 + (1-beta) * self.db2

        self.b2 = self.b2 - alpha * self.Vdb2

        # --- END CODE HERE

                

        return

    

    def M_fgsm(self, X, y):

        """Backward propagation in the MLP



        Parameters

        ----------

        X: numpy array (nbDim, nbData)

            observation data

        y: numpy array (1, nbData)

            ground-truth class to predict

            

        """

        

        m = y.shape[1]

        #print("shape {}".format(m))

                

        

        # --- dZ2 (n_out, m) = self.A2 (n_out, m) - self.y (n_out, m)

        dZ2 = self.A2 - y        

        # --- dW2 (n_out, n_h) = self.dZ2 (n_out, m) self.A1.T (n_h, m).T

        dW2 = (1 / m) * np.dot(dZ2, self.A1.T)

        # --- db2 (n_out, 1) = self.dZ2 (n_out, m) 

        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        # --- dA1 (n_h, m) = self.W2.T (n_out, n_h).T self.dZ2 (n_out, m)

        dA1 = np.dot(self.W2.T, dZ2)

        

        if (self.activation_function==1):

            dZ1 = np.multiply(dA1, F_dReluN(self.Z1,self.n))



        elif (self.activation_function==2):

            dZ1 = np.multiply(dA1, F_dSigmoid(self.Z1))



        # --- dW1 (n_h, n_in) = self.dZ1 (n_h, m) X.T (n_in, m).T

        dX =  np.dot(self.W1.T,dZ1 )

        

        # --- db1 (n_h, 1) = self.dZ1 (n_h, m) 

        

        return dX
def plot_contour(myMLP_,X, xtrain_to_plot, y_train_to_plot):

    h = .02  # step size in the mesh

    offset = 0.1

    x_min, x_max = X.T[:, 0].min() - offset, X.T[:, 0].max() + offset

    y_min, y_max = X.T[:, 1].min() - offset, X.T[:, 1].max() + offset

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                         np.arange(y_min, y_max, h))

    Z = myMLP_.M_forwardPropagation( np.transpose(np.c_[xx.ravel(), yy.ravel()]))

    Z2=Z.copy()

    Z2[Z>=0.5]=1

    Z2[Z<0.5]=0

    print(xtrain_to_plot.shape)

    print(y_train_to_plot.shape)

    



    plt.plot(xtrain_to_plot.T[y_train_to_plot.reshape(-1) == 1, 0], xtrain_to_plot.T[y_train_to_plot.reshape(-1) == 1, 1], 'yo')

    plt.plot(xtrain_to_plot.T[y_train_to_plot.reshape(-1) == 0, 0], xtrain_to_plot.T[y_train_to_plot.reshape(-1) == 0, 1], 'ko')

    plt.contourf(xx, yy, Z2.reshape(xx.shape), cmap=plt.cm.Paired)
def plot_contour2(myMLP,X, xtrain_to_plot, y_train_to_plot, offset=1.):

    h = .02  # step size in the mesh

    #offset = 2.

    x_min, x_max = X[:, 0].min() - offset, X[:, 0].max() + offset

    y_min, y_max = X[:, 1].min() - offset, X[:, 1].max() + offset

    #x_min, x_max=-10,10

    #y_min, y_max=-8,8

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                         np.arange(y_min, y_max, h))

    

    xxx=np.arctan2(*np.c_[xx.ravel(), yy.ravel()].T[::-1])

    yyy=np.linalg.norm(np.c_[xx.ravel(), yy.ravel()],axis=1)

    Z = myMLP.M_forwardPropagation( np.transpose(np.c_[xxx.ravel(), yyy.ravel()]))

    Z2=Z.copy()

    Z2[Z>=0.5]=1

    Z2[Z<0.5]=0

    #print(xtrain_to_plot.shape)

    #print(y_train_to_plot.shape)

    



    plt.plot(xtrain_to_plot.T[y_train_to_plot.reshape(-1) == 1, 0], xtrain_to_plot.T[y_train_to_plot.reshape(-1) == 1, 1], 'yo')

    plt.plot(xtrain_to_plot.T[y_train_to_plot.reshape(-1) == 0, 0], xtrain_to_plot.T[y_train_to_plot.reshape(-1) == 0, 1], 'ko')

    plt.contourf(xx, yy, Z2.reshape(xx.shape), cmap=plt.cm.Paired)

    

    #X2=np.arctan2(*X.T[::-1])

    #X3=np.linalg.norm(X,axis=1)
def adversarials_analyse(points,myMLP, rangeEpsilon=np.arange(0,0.7,0.07)):

    """

    Generate adversarials from points entered, given a MLP, and epsilons

    """

    

    advs=np.zeros((10,points.shape[0],2))

    accuracies=np.zeros((10,1))

    diffs=np.zeros((points.shape[0],10))

    

    yhats=myMLP.M_forwardPropagation(points.T)

    perturbations=myMLP.M_fgsm(points.T,np.ones((1,points.shape[0])))

    print(perturbations.shape)

    print(points.shape)



    y_hats_advs=[]

    for i,eps in enumerate(rangeEpsilon):



        advs[i,:,:]=(points.T-np.sign(perturbations)*eps).T

        #advs[i,:,:]=points-perturbations.reshape(points.shape[0],2)*eps

        diffs[:,i]= np.linalg.norm(points-advs[i,:,:],axis=1)

        y_hat_advs=myMLP.M_forwardPropagation(advs[i,:,:].T)

        accuracies[i]=F_computeAccuracy(y_hat_advs.reshape(1,points.shape[0]),np.zeros((1,points.shape[0])))

        y_hats_advs.append(y_hat_advs)

    diffmeans=np.mean(diffs,axis=0)



    return advs, accuracies,diffmeans,np.array(y_hats_advs)
def plot_cost(train_cost,test_cost, train_accuracy, test_accuracy):

    plt.subplot(1,2,1)

    plt.plot(train_cost, 'r')

    plt.plot(test_cost, 'g--')

    plt.xlabel('# epoch')

    plt.ylabel('loss')

    plt.grid(True)

    plt.subplot(1,2,2)

    plt.plot(train_accuracy, 'r')

    plt.plot(test_accuracy, 'g--')

    plt.xlabel('# epoch')

    plt.ylabel('accuracy')

    plt.grid(True)

    return
def train_mlp(mlp, X_tr, y_tr, X_te, y_te, num_epoch=10000, alphagrad=0.1 ):

    """

    Train mlp

    """

    

    # Instantiate the class MLP with providing 

    # the size of the various layers (input=4, hidden=10, outout=1) 



    #n_hidden = 10

    #num_epoch = 10000







    train_cost, train_accuracy, test_cost, test_accuracy = [], [], [], []



    # Run over epochs

    for i in range(0, num_epoch):



        # --- Forward

        hat_y_train = mlp.M_forwardPropagation(X_tr)



        # --- Store results on train

        train_cost.append( F_computeCost(hat_y_train, y_tr) )

        train_accuracy.append( F_computeAccuracy(hat_y_train, y_tr) )



        # --- Backward

        mlp.M_backwardPropagation(X_tr, y_tr)



        # --- Update

        mlp.M_gradientDescent(alpha=alphagrad)

        #myMLP.M_momentum(alpha=0.1, beta=0.9)



        # --- Store results on test

        hat_y_test = mlp.M_forwardPropagation(X_te)

        test_cost.append( F_computeCost(hat_y_test, y_te) )    

        test_accuracy.append( F_computeAccuracy(hat_y_test, y_te) )



        if (i % 500)==0: 

            print("epoch: {0:d} (cost: train {1:.2f} test {2:.2f}) (accuracy: train {3:.2f} test {4:.2f})".format(i, train_cost[-1], test_cost[-1], train_accuracy[-1], test_accuracy[-1]))

    

    plot_cost(train_cost,test_cost, train_accuracy, test_accuracy)

    return mlp
X0=spiral(10,0.1)



X1=spiral(10,0.1, angle=math.pi)
X=np.append(X0,X1, axis=0)
y=np.append(np.zeros((X0.shape[0])),np.ones((X1.shape[0])))
X2=np.arctan2(*X.T[::-1])

X3=np.linalg.norm(X,axis=1)
X_spiral=np.append(X,X2.reshape(X2.shape[0],1),axis=1)

X_spiral2=np.append(X_spiral,X3.reshape(X3.shape[0],1),axis=1)
X_spiral_only=X_spiral2[:,2:4]

X_spiral_only.shape
#X, y = datasets.make_circles(n_samples=1000, noise=0.05, factor=0.5)



print("X.shape: {}".format(X.shape))

print("y.shape: {}".format(y.shape))

print(set(y))



# X is (nbExamples, nbDim)

# y is (nbExamples,)



# --- Standardize data

#X = F_standardize(X)



# --- Split between training set and test set

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_spiral2, y, test_size=0.2)



# --- Convert to proper shape: (nbExamples, nbDim) -> (nbDim, nbExamples)

X_train = X_train.T

X_test = X_test.T



# --- Convert to proper shape: (nbExamples,) -> (1, nbExamples)

y_train = y_train.reshape(1, len(y_train))

y_test = y_test.reshape(1, len(y_test))



# --- Convert to oneHotEncoding: (1, nbExamples) -> (nbClass, nbExamples)

n_in = X_train.shape[0]

n_out = 1



print("X_train.shape: {}".format(X_train.shape))

print("X_test.shape: {}".format(X_test.shape))

print("y_train.shape: {}".format(y_train.shape))

print("y_test.shape: {}".format(y_test.shape))

print("y_train.shape: {}".format(y_train.shape))

print("y_test.shape: {}".format(y_test.shape))

print("n_in: {} n_out: {}".format(n_in, n_out))
myMLP_ = C_MultiLayerPerceptron(2, 400, n_out,1)

myMLP_=train_mlp(myMLP_, X_train[0:2,:], y_train ,X_test[0:2,:], y_test, num_epoch=30000, alphagrad=0.3)
plot_contour(myMLP_,X_train[0:2,:], X_train[0:2,:], y_train)
plt.scatter(X_train.T[y_train.reshape(-1) == 0, 0],X_train.T[y_train.reshape(-1) == 0, 1])

plt.scatter(X_train.T[y_train.reshape(-1) == 1, 0],X_train.T[y_train.reshape(-1) == 1, 1])
#plot_contour2(myMLP_,X, X.T, y)
myMLP = C_MultiLayerPerceptron(2, 200, n_out,1)

myMLP=train_mlp(myMLP,X_train[2:4,:], y_train ,X_test[2:4,:], y_test, num_epoch=30000, alphagrad=0.1)
plot_contour2(myMLP,X, X.T, y)
#XX=X_train.copy()

#XX=XX.reshape(X_train.shape[1],2)

#XX.shape
XX=X.copy()

print(XX.shape)

XX=XX.reshape(XX.shape[0],2)

print(XX.shape)

XX.shape
XX.shape

plot_contour(myMLP,X_train[2:4,:], X_train[2:4,:], y_train)
plot_contour2(myMLP,X, X.T, y, offset=4.)
plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

plt.plot(X_spiral2[:,2])

plt.subplot(1,2,2)

plt.plot(X_spiral2[:,3])