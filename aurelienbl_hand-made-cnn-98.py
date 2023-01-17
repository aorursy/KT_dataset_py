# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as nm # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

from scipy import optimize as opti

## Defining some functions.



## This is the sigmoid activation function

def sigmoid(z):

    return 1./(1+ nm.exp(-z));



## This is the gradient of the sigmoid function. Used in the backprop

def sigmoidGradient(z):

     return sigmoid(z)*(1-sigmoid(z));



    

### these 2 function are used to go back and forth between all the paramters stacked and flatten

### and arrays of proper shape and size

### weights and bias are the parameters of the final dense layer (basixally a MLP woth no hidden layer)

### and Wf and bf are the parameters of the CONV layer

### all_units is a tuple that contains the number  of units of the hidden layers of the MLP. Here is empty

def Wb2theta(weights, bias, Wf,bf, all_units):

    #for each layer w then b 

    theta=weights[0].flatten()

    theta=nm.append(theta, bias[0].flatten())

    for i in range(1,len(all_units)-1):

        theta=nm.append(theta,weights[i].flatten())

        theta=nm.append(theta,bias[i].flatten())



    theta=nm.append(theta, Wf.flatten())

    theta=nm.append(theta, bf.flatten())

    return theta





def theta2Wb(theta, all_units, ):

    

    w=[]

    b=[]

    wf=[]

    bf=[]



    id=0

    for i in range(len(all_units)-1):

        size1=all_units[i+1]

        size2=all_units[i]

        w.append(nm.array(theta[id:id+size1*size2]).reshape((size1, size2)))

        id=id+size1*size2

        b.append(nm.array(theta[id:id+size1]).reshape((size1, 1)))

        id=id+size1



    size1=filterDim**2 * numFilters

    wf=nm.array(theta[id:id+size1].reshape(filterDim,filterDim, numFilters))

    id=id+size1

    size1=numFilters

    bf=nm.array(theta[id:id+size1].reshape(size1))



    return w,b, wf,bf
## More functions. These two are the convolution and mean pooling part



def convolve_images(filterDim, numFilters, images, W, b):

    N_images=images.shape[0]

    imageDim =images.shape[1]



    convDim = imageDim -filterDim +1

    convolvedFeatures = nm.zeros((convDim, convDim, numFilters, N_images))

    for i in range(N_images):

        for j in range(numFilters):

            convolvedFeatures[:,:, j,i]= signal.convolve2d(images[i, :,:],W[:,:,j], 'valid') + b[j]

        

    return convolvedFeatures





def cnnPool(poolDim , convolvedFeatures):

    numImages = convolvedFeatures.shape[3]

    numFilters = convolvedFeatures.shape[2];

    convolvedDim = convolvedFeatures.shape[0];



    pooledFeatures = nm.zeros((convolvedDim / poolDim,

                               (convolvedDim / poolDim), numFilters, numImages));

    

    for i in range(convolvedDim / poolDim):

        for j in range(convolvedDim / poolDim):

            pooledFeatures[i,j, :,:]= nm.mean(convolvedFeatures[i* poolDim:(i+1)*poolDim   ,j*(poolDim):(j+1)*poolDim ,  :,:], axis=(0,1))

            

    return pooledFeatures



    
#### this is the main part of the code; the function that compute the cost function

### and its gradient



def costFunction(theta, X, y, all_units, pred=False):



    m=X.shape[0]

    y=y.reshape(y.size, 1)

    ### reshaping the input paramters in a more convenient form

    weights, bias, Wf, bf= theta2Wb(theta,all_units)

    

    ## 1 convolution + sigmoid + mean pooling



    convolvedFeatures = convolve_images(filterDim, numFilters, X, Wf, bf);

    pooledFeatures= cnnPool(poolDim , sigmoid(convolvedFeatures))

    

    ## agreggating everything in one big vector 

    denseFeatures=pooledFeatures.reshape(pooledDim*pooledDim*numFilters, m)

 

    ### adding the ouput to a sofmax layer. This part is a code for a N hidden layer MLP. 

    ### Here N=0, but for simplicity I left the (here useless) loops.

    a0=denseFeatures

    

    zl=[]

    al=[]

    

    al.append(a0)

    zl.append(0)

        

    for i in range( Nh):

        zl.append(nm.dot(weights[i], al[i]) + bias[i])

        al.append(f(zl[i+1]))

    

    zl.append(nm.dot(weights[Nh], al[Nh]) + bias[Nh])

    

    ## In case one just want to compute prediction and no cost are gradient

    if pred:

        result=nm.exp(zl[-1])/nm.sum(nm.exp(zl[-1]), axis=0)

        pred=nm.argmax(result, axis=0)

        return  pred

    

   

    ## defing the big target array

    Y=nm.zeros((all_units[-1], m))

    for k in range(all_units[-1]):

        id=nm.where(y==k)[0]

        Y[k,id]=1

        

    ## finding the mac of the softmax argument to remove it to prevent overflow

    mm=nm.max(zl[-1], axis=0)

    

    ## cost function

    J=-(Y*((zl[-1]-mm) - nm.log(nm.sum(nm.exp(zl[-1]-mm), axis=0)))).sum()

    

    # J=-(Y*nm.log(nm.exp(zl[-1])/nm.sum(nm.exp(zl[-1]), axis=0))).sum()

    

    ### back propagation



    delta_L= -( Y - nm.exp(zl[-1]-mm)/nm.sum(nm.exp(zl[-1]-mm), axis=0)   )



    delta_Lm1 = nm.dot(weights[0].T, delta_L)

    delta_pool=delta_Lm1.reshape(pooledDim,pooledDim,numFilters,m)

    

    delta_pool2=nm.zeros((convDim, convDim, numFilters, m))



    for i in range(m):

        for j in range(numFilters):

            delta_pool2[:,:,j,i]=nm.kron(delta_pool[:,:,j,i], nm.ones((poolDim,poolDim)))/ (poolDim**2)*sigmoidGradient(convolvedFeatures[:,:,j,i])



    dJdw2=nm.zeros((filterDim, filterDim, numFilters, m))



    for i in range(m):

        for j in range(numFilters):

            dJdw2[:,:,j,i]=signal.convolve2d(delta_pool2[:,:,j,i],nm.rot90(X[i, :,:],2), 'valid')





    dJdw=nm.sum(dJdw2, axis=3)

    dJdb=delta_pool2.sum(axis=(0,1,3))

    

    dJdw0=[nm.dot(delta_L,denseFeatures.T),]

    dJdb0=[delta_L.sum(axis=1).reshape(numClasses, 1),]



    grad=Wb2theta(dJdw0,dJdb0,dJdw, dJdb, all_units )



    return J, grad

A=pd.read_csv('../input/train.csv').values

B=pd.read_csv('../input/test.csv').values



## defining the target and the feature matrix.

X=(A[:, 1:]-128)/128.

B=(B[:,:]-128)/128.

y=nm.int32(A[:,0])





imageDim = 28;       #  % image dimension

filterDim = 7;        #  % filter dimension

numFilters = 2     #   % number of feature maps



poolDim = 2; 

convDim = imageDim -filterDim +1 

pooledDim=convDim/poolDim  

numClasses=10



### Parameters for the dense MLP layer.

Nh=0

Nl=Nh+2

all_units=[]

all_units.append(nm.int32(pooledDim*pooledDim*numFilters))

Nunits=() 



for  l in range(Nh):

    all_units.append(nm.int32(Nunits[l]))



all_units.append(numClasses)



## defining the initial weights at random.  The choice for the MLP weights comes from Glorot&Bengio, 2010

Wf = nm.random.randn(filterDim,filterDim, numFilters);

bf = nm.random.rand(numFilters)*0;



weights=[]

bias=[]

for l  in range(Nl-1):

    s=nm.sqrt(6)/ nm.sqrt(all_units[l+1] +all_units[l])

    weights.append(nm.random.rand(all_units[l+1] ,all_units[l]   ) *2*s -s)

    bias.append(nm.random.rand(all_units[l+1],1)*0)



theta_init=Wb2theta(weights, bias, Wf,bf, all_units)





Wf = nm.random.randn(filterDim,filterDim, numFilters);

bf = nm.random.rand(numFilters)*0;



weights=[]

bias=[]

for l  in range(Nl-1):

    s=nm.sqrt(6)/ nm.sqrt(all_units[l+1] +all_units[l])

    weights.append(nm.random.rand(all_units[l+1] ,all_units[l]   ) *2*s -s)

    bias.append(nm.random.rand(all_units[l+1],1)*0)







theta_init=Wb2theta(weights, bias, Wf,bf, all_units)



images=X.copy()

m=images.shape[0]

theta=theta_init.copy()



### Home-made implemenantation of a SGD. 

v=0

gamma=0.5

alpha0=1e-2

alpha=alpha0

minibatch=30

idd=0

for i in range(1):

    r=nm.random.permutation(m)

    for j in range(m%minibatch):

        tt=r[j*minibatch:(j+1)*minibatch]

        J, grad=costFunction(theta, images[tt,:,:], y[tt], all_units)

    

        v=gamma*v +alpha * grad

        theta=theta - v 

        print('i=', i,'j=', j, 'J=',J, 'alpha=', alpha)

    alpha =alpha/2.



    

### one the training is done compute the training error:

#p=costFunction(theta, images, y, all_units, pred=True)



#print(nm.mean(p==y))
