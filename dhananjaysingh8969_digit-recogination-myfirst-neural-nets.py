#Hey guys i just started deep learning this my first neural nets implementation to digit recogination:) 
#on and average the accuracy on validation set of my network is  91%
import numpy as np
import pandas as pd 
import random
import matplotlib.pyplot as plt
class NeuralNet:
    def __init__(self,sizes):
        self.weights=[np.random.randn(r,c)/np.sqrt(c) for (r,c)  in zip(sizes[1:],sizes[:-1])]
        self.biases=[np.random.randn(r,1) for r in sizes[1:]]
        self.noOfLayers=len(sizes)
    def feedForward(self,x):
        activation=x
        for (w,b) in zip(self.weights,self.biases):
            activation=self.sigmoid(np.dot(w,activation)+b)
        return activation
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def sigmoidPrime(self,z):
        sg=self.sigmoid(z)
        return sg*(1-sg)
    def StochasticGradientDescent(self,train_data,validation_data,learningRate,lmda,sizeOfMiniBatch,ephocs):
        lengthOfTrainingSet=len(train_data)
        costOnEachEphocsForTrainData=[]
        costOnEachEphocsForValidationData=[]
        lv=len(validation_data)
        for i in range(ephocs):
            random.shuffle(train_data)
            miniBatchs=[train_data[k:k+sizeOfMiniBatch] for k in range(0,lengthOfTrainingSet,sizeOfMiniBatch)]
            for miniBatch in miniBatchs:
                self.updateParameters(miniBatch,learningRate,lmda,lengthOfTrainingSet,sizeOfMiniBatch)
            costOnEachEphocsForTrainData.append(self.totalCost(train_data,lmda))
            costOnEachEphocsForValidationData.append(self.totalCost(validation_data,lmda))
            print("ephocs:{0} :{1} / {2}".format(i,self.accuracy(validation_data),lv))
        return costOnEachEphocsForTrainData,costOnEachEphocsForValidationData    
    def updateParameters(self,miniBatch,learningRate,lmda,lengthOfTrainingSet,sizeOfMiniBatch):
        Del_w=[np.zeros(w.shape) for w in self.weights]
        Del_b=[np.zeros(b.shape) for b in self.biases]
        #just copied the zeros list for the shake of optimisation
        dW=Del_w
        db=Del_b
        for (x,y) in miniBatch:
            (del_w,del_b)=self.backProp(x,y,dW,db)
            Del_w=[D_w+d_w for D_w,d_w in zip(Del_w,del_w)]
            Del_b=[D_b+d_b for D_b,d_b in zip(Del_b,del_b)]
            self.weights=[w*(1-(learningRate*lmda)/lengthOfTrainingSet) -(learningRate/sizeOfMiniBatch)*D_w for w,D_w in zip(self.weights,Del_w)]
            self.biases=[b -(learningRate/sizeOfMiniBatch)*D_b for b,D_b in zip(self.biases,Del_b)]
    def backProp(self,x,y,dW,db):
        activation=x
        activations=[activation]
        zs=[]
        for w,b in zip(self.weights,self.biases):
            z=np.dot(w,activation)+b
            activation=self.sigmoid(z)
            zs.append(z)
            activations.append(activation)
        delta=self.crossEntropyCost(activation,y)
        dW[-1]=np.dot(delta,activations[-2].transpose())
        db[-1]=delta
        for i in range(2,self.noOfLayers):
            delta=np.dot(self.weights[-i+1].transpose(),delta)*self.sigmoidPrime(zs[-i])
            dW[-i]=np.dot(delta,activations[-i-1].transpose())
            db=delta
        return (dW,db) 
    def totalCost(self,data,lmda):
        cost=0
        ln=len(data)
        for (x,y) in data:
            a=self.feedForward(x)
            cost+=self.crossEntropyFn(a,y)/ln
            #cost+=0.5*(lmda/ln)*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost    
    def crossEntropyFn(self,a,y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    def crossEntropyCost(self,a,y):
        return (a-y)
    def accuracy(self,data):
        results=[(np.argmax(self.feedForward(a)),np.argmax(y)) for (a,y) in data]
        return sum(int(x==y) for (x,y) in results )
    pass
def vectorizedLabel(y):
        Y=np.zeros((10,1))
        Y[y]=1
        return Y
if __name__== '__main__':
    train_data=pd.read_csv('../input/train.csv')
    train_data=np.array(train_data)
    NeuralNetwork=NeuralNet([784,30,10])
    train_data=[(x.reshape(784,1),vectorizedLabel(y)) for x,y in zip(train_data[:,1:],train_data[:,0:1])]
    random.shuffle(train_data)
    validation_data=train_data[:100]
    train_data=train_data[100:]
    """pixels = train_data[50][0].reshape(28,28)
    plt.imshow(pixels, cmap='gray')
    plt.show() """
    costOnEachEphocsForTrainData,costOnEachEphocsForValidationData=NeuralNetwork.StochasticGradientDescent(train_data,validation_data,0.0001/3,3,10,40)
    Xaxis=[x for x in range(len(costOnEachEphocsForTrainData))]
    
    plt.plot(Xaxis,costOnEachEphocsForTrainData,color='red')
    plt.plot(Xaxis,costOnEachEphocsForValidationData,color='blue')
    plt.xlabel('ephcos')
    plt.ylabel('costs')
    plt.title('costOnEachEphocs')
    print("Done!")
