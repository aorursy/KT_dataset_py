%matplotlib inline

import numpy as np

from sklearn import datasets

from sklearn import model_selection

import matplotlib.pyplot as plt 

from keras.utils.np_utils import to_categorical
def F_standardize(X_):

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

    X_ = X_.astype('float32')

    X_ = X_ / 255

    #X_-=np.mean(X_,axis=0)

    X_=X_.reshape(X_.shape[0],784)

    print(X_.shape)

    #X -= np.mean(X, axis=0, keepdims=True) 

    #X /= (np.std(X, axis=0, keepdims=True) + 1e-16)

    return X_
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



def F_computeCost(hat_y_, y_):

    """Compute the cost (sum of the losses)

    

    Parameters

    ----------

    hat_y: (1, nbData)

        predicted value by the MLP

    y: (1, nbData)

        ground-truth class to predict

    """

    m = y_.shape[1]



    y_=np.transpose(y_[0])

    

    #print(m)

    #print(np.argmax(y_,axis=0))

    #print(np.argmax(hat_y_,axis=0))

    #print(hat_y_.shape)



    #print(y_.shape)

    #print(np.argmax(hat_y_,axis=0))

    #loss =  - (np.multiply(y, np.log(hat_y)) + np.multiply((1 - y), np.log(1 - hat_y)))  

    loss =  -np.sum( np.multiply(y_, np.log(hat_y_)),axis=0)

    #print(y_)



    cost = np.sum(loss) / m

    #print(cost)

    return cost



def F_computeAccuracy(hat_y_, y_):

    """Compute the accuracy

    

    Parameters

    ----------

    hat_y: (1, nbData)

        predicted value by the MLP

    y: (1, nbData)

        ground-truth class to predict

    """

    

    m = y_.shape[1]

    #print(hat_y.shape)

    #print(y.shape)



    pred_correct=np.zeros((m,1))

    pred_correct[hat_y_.reshape(-1)==y_.reshape(-1)]=1

    return np.sum(pred_correct) / m
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



            



    def M_forwardPropagation(self, X):

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

        

        

        # --- Z1 (n_h, m) = W1 (n_h, n_in) X (n_in, m) + b1 (n_h, 1)        

        self.Z1 = np.dot(self.W1, X) + self.b1

        if (self.activation_function==1):

            self.A1 = F_reluN(self.Z1,self.n)

        elif (self.activation_function==2):

            #print("hee")

            self.A1 = F_sigmoid(self.Z1)

        # --- Z2 (n_out, m) = W2 (n_out, n_h) A1 (n_h, m) + b2 (n_out, 1)

        self.Z2 = np.dot(self.W2, self.A1) + self.b2        

        self.A2 = F_sigmoid(self.Z2)

                

        hat_y = self.A2

        

        return hat_y





    def M_backwardPropagation(self, X, y):

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

        self.dW1 = (1 / m) * np.dot(self.dZ1, X.T)

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





    def M_fgsm(self, X, y):

        """Backward propagation TO X



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
def plot_contour(myMLP,X):

    h = .02  # step size in the mesh

    offset = 0.1

    x_min, x_max = X[:, 0].min() - offset, X[:, 0].max() + offset

    y_min, y_max = X[:, 1].min() - offset, X[:, 1].max() + offset

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                         np.arange(y_min, y_max, h))

    Z = myMLP.M_forwardPropagation( np.transpose(np.c_[xx.ravel(), yy.ravel()]))

    Z2=Z.copy()

    Z2[Z>=0.5]=1

    Z2[Z<0.5]=0



    plt.plot(XX[y_train.reshape(-1) == 1, 0], XX[y_train.reshape(-1) == 1, 1], 'yo')

    plt.plot(XX[y_train.reshape(-1) == 0, 0], XX[y_train.reshape(-1) == 0, 1], 'ko')

    plt.contourf(xx, yy, Z2.reshape(xx.shape), cmap=plt.cm.Paired)
def adversarials_analyse(points,y_,myMLP, rangeEpsilon=np.arange(0,0.7,0.07)):

    """

    Generate adversarials from points entered, given a MLP, and epsilons

    """

    

    advs=np.zeros((10,points.shape[0],784))

    accuracies=np.zeros((10,1))

    diffs=np.zeros((points.shape[0],10))

    

    yhats=myMLP.M_forwardPropagation(points.reshape(784,points.shape[0]))

    perturbations=myMLP.M_fgsm(points.reshape(784,points.shape[0]),np.ones((1,points.shape[0])))

    

    for i,eps in enumerate(rangeEpsilon):



        advs[i,:,:]=points-np.sign(np.transpose(perturbations))*eps

        #advs[i,:,:]=points-perturbations.reshape(points.shape[0],2)*eps

        diffs[:,i]= np.linalg.norm(points-advs[i,:,:],axis=1)

        y_hat_advs=np.argmax(myMLP.M_forwardPropagation(advs[i,:,:].T),axis=0)

        #print(y_hat_advs.shape)

        #print(y_.shape)





        accuracies[i]=F_computeAccuracy(y_hat_advs.reshape(1,points.shape[0]),y_)



    diffmeans=np.mean(diffs,axis=0)



    return advs, accuracies,diffmeans
def train_mlp(mlp,X_train_,y_train_,X_test_,y_test_,num_epoch=10000, alphagrad=0.1):

    """

    Train mlp

    """

    y_train_cat=to_categorical(y_train_,num_classes=10)

    y_test_cat=to_categorical(y_test_,num_classes=10)

    # Instantiate the class MLP with providing 

    # the size of the various layers (input=4, hidden=10, outout=1) 



    #n_hidden = 10

    #num_epoch = 10000







    train_cost, train_accuracy, test_cost, test_accuracy = [], [], [], []



    # Run over epochs

    for i in range(0, num_epoch):



        # --- Forward

        hat_y_train = mlp.M_forwardPropagation(X_train_)

        #print(hat_y_train.shape)



        # --- Store results on train

        train_cost.append( F_computeCost(hat_y_train, y_train_cat) )

        train_accuracy.append( F_computeAccuracy(np.argmax(hat_y_train,axis=0), y_train_) )



        # --- Backward

        mlp.M_backwardPropagation(X_train_, np.transpose(y_train_cat.reshape(y_train_cat.shape[1],y_train_cat.shape[2])))



        # --- Update

        mlp.M_gradientDescent(alpha=alphagrad)

        #myMLP.M_momentum(alpha=0.1, beta=0.9)



        # --- Store results on test

        hat_y_test = mlp.M_forwardPropagation(X_test_)

        test_cost.append( F_computeCost(hat_y_test, y_test_cat) )    

        test_accuracy.append( F_computeAccuracy(np.argmax(hat_y_test,axis=0), y_test_) )



        if (i % 500)==0: 

            print("epoch: {0:d} (cost: train {1:.2f} test {2:.2f}) (accuracy: train {3:.2f} test {4:.2f})".format(i, train_cost[-1], test_cost[-1], train_accuracy[-1], test_accuracy[-1]))

            

    return mlp,train_cost, train_accuracy, test_cost, test_accuracy
import tensorflow as tf

(x_traink, y_traink), (x_testk, y_testk) = tf.keras.datasets.mnist.load_data()
X,y=np.array(x_traink[:]),y_traink[:]



#print(X.shape)

#X = F_standardize(X)



#X.shape
plt.imshow(X[0].reshape(28,28))


print("X.shape: {}".format(X.shape))

print("y.shape: {}".format(y.shape))

#print(set(y))



# X is (nbExamples, nbDim)

# y is (nbExamples,)



# --- Standardize data

X = F_standardize(X)



# --- Split between training set and test set

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)



# --- Convert to proper shape: (nbExamples, nbDim) -> (nbDim, nbExamples)

X_train = X_train.T

X_test = X_test.T



# --- Convert to proper shape: (nbExamples,) -> (1, nbExamples)

y_train = y_train.reshape(1, len(y_train))

y_test = y_test.reshape(1, len(y_test))



# --- Convert to oneHotEncoding: (1, nbExamples) -> (nbClass, nbExamples)

n_in = X_train.shape[0]

n_out = 10



print("X_train.shape: {}".format(X_train.shape))

print("X_test.shape: {}".format(X_test.shape))

print("y_train.shape: {}".format(y_train.shape))

print("y_test.shape: {}".format(y_test.shape))

print("y_train.shape: {}".format(y_train.shape))

print("y_test.shape: {}".format(y_test.shape))

print("n_in: {} n_out: {}".format(n_in, n_out))
#y_train_cat=to_categorical(y_train,num_classes=10)

#y_test_cat=to_categorical(y_test,num_classes=10)
#y_train_cat.shape
y_train.shape
# Instantiate the class MLP with providing 

# the size of the various layers (input=4, hidden=10, outout=1) 



n_hidden = 10

num_epoch = 5000





myMLP = C_MultiLayerPerceptron(784, n_hidden, n_out,1)

myMLP, train_cost, train_accuracy, test_cost, test_accuracy=train_mlp(myMLP,X_train[:,:], y_train[:,:], X_test[:,:], y_test[:,:], alphagrad=0.1)
y_train.shape[1]
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
XX=X_train.copy().T

#XX=XX.reshape(800,784)

XX.shape

Xno0=XX[y_train.reshape(y_train.shape[1])!=0]

yno0=y_train.reshape(-1)[y_train.reshape(-1)!=0].reshape(1,-1,)
Xno0.shape

advs,accuracies, l2=adversarials_analyse(Xno0[:100],yno0[:,:100],myMLP, rangeEpsilon=np.arange(0,0.7,0.07))
plt.figure(figsize=(20,10))

idx_img=12

for i in range(10):

    plt.subplot(2,5,i+1)

    pred= np.argmax(myMLP.M_forwardPropagation(advs[i,idx_img,:].reshape(784,1)), axis=0)

    plt.title("predict {}".format(pred))

    plt.imshow(advs[i,idx_img,:].reshape(28,28))
plt.plot(l2,accuracies, label="model relu")

plt.title("Accuracy vs l2")

plt.legend()

plt.show()
myMLP2 = C_MultiLayerPerceptron(784, n_hidden, n_out,1,activation_f=2)

myMLP2, train_cost, train_accuracy, test_cost, test_accuracy=train_mlp(myMLP2,X_train[:,:], y_train[:,:], X_test[:,:], y_test[:,:], alphagrad=0.1)
advs2,accuracies2, l22=adversarials_analyse(Xno0[:100],yno0[:,:100],myMLP2, rangeEpsilon=np.arange(0,0.7,0.07))
plt.plot(l2,accuracies, label="model relu")

plt.plot(l22,accuracies2, label="model sigmoid")

plt.title("Accuracy vs l2")

plt.legend()

plt.show()