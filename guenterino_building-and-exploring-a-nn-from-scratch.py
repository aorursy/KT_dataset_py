import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) #
import matplotlib.pyplot as pt
import random as rd
from math import e
from datetime import datetime
from scipy.ndimage import rotate, shift
from skimage.transform import resize
my_data = pd.read_csv('../input/train.csv')
data=np.array(my_data)
X=data[0:42000,1:785 ] #data (the pictures)
X=X/X.max()  #Turn all values to values between 0 and 1 (Greyscale, 0 is white, 1 black - it's more intuitive)
Y=data[0:42000, 0:1] #the coresponding labels
temp=np.array([np.arange(0,10) for a in range(42000)])
Y=(temp==Y).astype(int) #create a boolean array for every example
Xt=X[0:28000] #splitting data
Yt=Y[0:28000]
Xv=X[28000:42000]
Yv=Y[28000:42000]
mastertime=datetime.now()
def reshape(picture,sidelength):
    '''
    1darray,sidelength-->2darray
    takes in a vector 
    and reshapes it into a quadratic array of size sidelength*sidelength
    '''
    picture=np.array(picture)
    if sidelength**2!=len(picture):
        raise ValueError("sidelength not correct")
    picture.shape=(sidelength,sidelength)
    return picture

for p in range (0,8): #showing 8 example images
    n=rd.randint(0, 42000) #random examples from the 42000 images in the training set
    ex1=reshape(X[n],28)
    pt.subplot(2,4,p+1)
    pt.imshow(ex1,'Greys_r') 
    pt.axis('off')
pt.show()

#In this cell I will define functions needed for the network below
def sigmoid(x): 
    '''
    float-->float between 1 and 0
    takes some number, applies the sigmoid function to it
    '''
    return 1/(1+e**(-x))  

def sigmoid_prime(x): #needed for backpropagation
    '''
    float-->float 
    '''
    return sigmoid(x)*(1-sigmoid(x))

class Neuralnetwork(object):
    def __init__(self, layers):
        '''
        Layers is expected to be a list containing the different layer sizes.
        When initialized the NN wil set up a these layers and initialize coresponding weight matrices randomly
        least number of layers is 2 (input and output)
        '''
        self.num=len(layers) #number of layers
        if self.num<2: raise ValueError("the fuck you tryna do with that shit, gotta have two layers at least")
        self.sizes=layers
        self.layers=[]
        self.wsums=[]
        for l in self.sizes:
            self.layers.append(np.array([1]+[0 for a in range(l)])) #initializing the different layers, including one bias unit each
            self.wsums.append(np.array ([0 for a in range(l)]))#initializing a matrix to store the weighted sums when propagating forward
            #The bias unit has to be restored after each matrixmultiplication, except for the output layer.
        self.weights=[0.5*np.random.randn(self.sizes[i+1],self.sizes[i]+1) for i in range(self.num-1)] #initializing weights, including the bias unit
    
    def feedforward(self,arr):
        '''
        needs an array (or list) of values as input for the first layer, saves the values inbetween
        returns an output for the network and safes the weighted sums and the activations for that example.
        '''
        self.layers[0]=np.array(arr)[:] #setting the input layer equal to the input
        for l in range(self.num-1): #iterating over all layers
            self.layers[l]=  np.insert(self.layers[l],0,1)#inserting the bias unit to the current layer
            self.wsums[l+1]=np.dot(self.weights[l],self.layers[l])#calculating the weighted sums for the next
            self.layers[l+1]=sigmoid(self.wsums[l+1]) #applying the sigmoid function to the next layers to get their activations
        return self.layers[-1] #return the output of the last layer (output of the network)
    
    def cost_function(self, output, expected_output,lambd=0):
        '''
        arrays,regularization parameter-->float
        takes in the output and the expected output and returns the cost for this particular example, lambd regularizes regularization
        '''
        self.cecost= -(expected_output*np.log(output)+(1-expected_output)*np.log(1-output)).sum()   #cross entropy cost function
        self.reg= (lambd/2)*sum([np.sum(a[:,1:]**2) for a in self.weights])#added regularization (simply the sum over all the weights, except biases, multiplied by some constant)
        self.cost=float(self.cecost+self.reg)
        return self.cost
    
    def train(self, X, Y, iterations, batch_size=0,eta=1, lambd=0,Validation=None,plot_learning=None):
        '''
        trains the network using a Trainingset X with labels Y
        you should enter the regularization paramater lamd and the learning rate eta
        furthermore you need to enter the batch_size and the number of iterations
        optimally batch size divides the number of training examples - the rest gets cut
        If Validation (print statement every iteration) please enter a tuple in the form of (Xval, Yval).
        If plot_learning include (Xval,Yval). Learning will be plottet every iteration. If you use both functions please use the same validation set. 
        '''
        self.X=X
        self.Y=Y
        if batch_size==0: 
            batch_size=len(Y) 
        if plot_learning != None:
            self.numex=0
            self.Xval=plot_learning[0]
            self.Yval=plot_learning[1]
            self.accuracytrain=[]
            self.accuracyval=[]
            self.examplesseen=[]
        self.runs=len(Y)//batch_size
        for h in range (iterations):
            for a in range(self.runs):
                self.grad=[g*0 for g in self.weights]
                #print((self.weights[1].sum(),self.grad[1].sum())) #I added these for debugging in the beginning - I had my sigma_prime function wrong and the gradients exploded everytime
                for g in range(a*batch_size,(a+1)*batch_size): #iterating over all training examples of the current batch
                    self.x=self.X[g]
                    self.y=self.Y[g]
                    self.errors=[None for b in range(len(self.layers)-1)]
                    self.feedforward(self.x) #feedforward and compute the new values for the layers and the weighted sums
                    
                    self.errors[-1]=self.layers[-1]-self.y #error for output layer
                    self.grad[-1]=self.grad[-1]+np.outer(self.errors[-1],self.layers[-2]) #updating the gradient for the last theta matrix
                
                    for n in range(2,len(self.layers)): #calculating errors and accumulating gradients for all layers/theta matrizes
                        self.errors[-n]=self.weights[-n+1].T.dot(self.errors[-n+1])[1:]*sigmoid_prime(self.wsums[-n]) 
                        self.grad[-n]=self.grad[-n]+np.outer(self.errors[-n],self.layers[-n-1])
                        
                for i in range(len(self.grad)):
                    self.grad[i]=(1/batch_size)*self.grad[i] #averaging over number of trainingexamples
                    self.grad[i][:,1:]=self.grad[i][:,1:]+(lambd/batch_size)*self.weights[i][:,1:] #adding regularization
                    self.weights[i]=self.weights[i]-eta*self.grad[i] ##updating the weights!!! "gradient descent step" WUHU

            if Validation!=None: #prints a validation statement after every iteration (kind of useless, but nice to see progress)
                self.validate(Validation[0],Validation[1], False)
                print('Iteration number',h+1,': \n',self.correct," with argmax, and ",self.correct2,' with round, out of ',self.n, 'correct.',self.accuracy)
            if plot_learning != None: #if learning needs to be plotted
                if Validation ==None:
                    self.accuracyval.append(self.validate(self.Xval,self.Yval,True)[0])
                else:
                    self.accuracyval.append(self.accuracy[0])
                self.numex+=1
                self.accuracytrain.append(self.validate(self.X,self.Y,True)[0])
                self.examplesseen.append(self.numex)
        if plot_learning != None:
            pt.plot(self.examplesseen,self.accuracytrain,'r--',self.examplesseen,self.accuracyval,'b--')
            pt.show()
            
    def validate(self,Xval,Yval,retour=True,cost=False,lamd=0):
            '''
            expects a set of examples, prints accuracys and returns training accuracy over argmax and over round if retour ==True
            if cost=True returns cost instead of accuracy
            '''
            self.Xv=Xval
            self.Yv=Yval
            self.n=len(self.Xv)
            self.correct=0
            self.correct2=0
            if cost: 
                self.costacc=0
            for t in range(self.n): #for every example
                self.feedforward(self.Xv[t]) #feedforward and thus set layers[-1] to whatever the network thinks it is
                if self.layers[-1].argmax()==self.Yv[t].argmax():
                    self.correct+=1
                if np.array_equal(self.layers[-1].round(),self.Yv[t]): 
                    self.correct2+=1
                if cost:
                    self.costacc+=self.cost_function(self.layers[-1], self.Yv[t],lamd) #I have no Idea why, but otherwise list index is out of range
            self.accuracy=(self.correct/self.n,self.correct2/self.n)
            if retour:
                return self.accuracy
            if cost:
                self.costacc=self.costacc/self.n
                return self.costacc
begin=datetime.now()
Alphaversion=Neuralnetwork((784,30,10))
Alphaversion.train(Xt,Yt,10,10,0.3,0,(Xv,Yv),(Xv,Yv))
print("Time needed for execution: ",datetime.now()-begin)
for p in range (0,24):
    n=rd.randint(0, 14000)
    label=Alphaversion.feedforward(Xv[n]).argmax()
    labelreal=Yv[n].argmax()
    ex1=reshape(Xv[n],28)
    pt.subplot(3,8,p+1)
    pt.imshow(ex1,'Greys_r',)   
    pt.title(str((label,labelreal)))
    pt.axis("off")
pt.show()

arcs=[[18,18,18,18,18],[30,30],[30,30,30]]
for g in arcs:
    g.append(10)
    g.insert(0,784)

for arc in arcs:
    Betaversion=Neuralnetwork(arc)
    Betaversion.train(Xt,Yt,10,10,0.3,0.00003,None,(Xv,Yv))
    print('performance for Network: ', arc)
    print(Betaversion.validate(Xv,Yv,True),'\n----------')
arcs=[[18,18,18,18,18],[30],[30,30],[30,30,30]]
for g in arcs:
    g.append(10)
    g.insert(0,784)
sizes=[1000,2500,5000,10000,15000,17500,20000,22500,25000,28000]
for arc in arcs:
    performancetrain=[]
    performanceval=[]
    Betaversion=Neuralnetwork(arc)
    a=0
    for b in sizes:
        Betaversion.train(Xt[a:b],Yt[a:b],10,10,0.3,0.00003,)
        performancetrain.append(Betaversion.validate(Xt[a:b],Yt[a:b],False,True))
        performanceval.append(Betaversion.validate(Xv,Yv,False,True))
        a=b
        #print(arc,'with',a,'examples','done','\n----------') #I used to do this to see how far I am, but it's not neccesary anymore
    t="Plots for architecture "+str(arc)
    pt.rcParams["figure.figsize"] = (11,3)
    pt.subplot(1,2,1)
    pt.plot(sizes,performancetrain,'r--',sizes,performanceval,'b--')
    pt.ylabel('Cost')
    pt.xlabel('Number of training examples')
    pt.title(t)
    pt.subplot(1,2,2)
    pt.plot(sizes[3:],performancetrain[3:],'r--',sizes[3:],performanceval[3:],'b--')
    pt.show()
def shiftrot(arr,sidelength,x=0,y=0,degree=0):
    '''
    numpy array---> numpy array
    shifts a picture randomly along the x and y axis by maximal -/+ x/y pixels (input are pixel values as flattened np array) 
    and rotates it by a randomly chosen value between +/- degree.
    needs the sidelength of the picture
    ###This function is incredibly ineffective, just in case someone knows a better way to do this tell me
    '''
    x=rd.randint(-x,x)
    y=rd.randint(-y,y)
    degree=rd.randint(-degree,degree)
    temp=rotate(reshape(arr,sidelength),degree,order=0,reshape=False) #reshape, then rotate, resize back to 28*28 and flatten again
    temp=shift(temp,(x,y),order=0).flatten()
    return temp

#This function creates some noise as the numbers tend to sometimes go over the sides and reappear on the other one. 
#I will see if this causes any problems
###What's mentioned above was solved by using the scipy.ndimage function for shift
###This part is used for visualizing the shift function
ueue=[]
i=rd.randint(4,28000)  #get four random examples in a row
for a in Xt[i-4:i]: 
    pt.rcParams["figure.figsize"] = (10,2) #setting figure size
    pt.subplot(1,5,1)
    pt.title('Original Image')
    pt.axis('off')
    pt.imshow(reshape(a,28),"Greys_r")
    for b in range(4):      #Testing statement for the shifting function - I'm gonna work this up so it looks better and then show it
        pt.subplot(1,5,b+2)
        pt.axis('off')
        pt.imshow(reshape(shiftrot(a,28,3,3,51),28),'Greys_r') #applying random shifts to the image
    pt.show()
class Forest(object):
    def __init__(self,n,layers=(784,30,30,10)):
        '''
        Layers is expected to be a list containing the different layer sizes for the networks that are to be used.
        When initialized the Forest wil set up n networks with this layer structure.
        '''
        self.layers=layers
        self.n=n
        self.Networks=[Neuralnetwork(self.layers) for a in range(n)] #set up n networks
        

    
   
    def train(self, X,Y,size,iterations=10,batchsize=10,eta=0.35,lambd=0.0013,showprogress=False):
        '''
        trains n classifiers that each see only a random part of the data given
        how much of the data has to be specified via the size variable.
        '''
        self.size=size
        self.X=X
        self.Y=Y
        self.it=iterations
        self.batch=batchsize
        self.eta=eta
        self.lambd=lambd
        self.a=0
        for N in self.Networks: #iterate over the n networks
            self.index=np.random.choice(len(self.X), self.size) #choose random samples out of the networks
            self.Xtemp=self.X[self.index]
            self.Ytemp=self.Y[self.index]
            N.train(self.Xtemp,self.Ytemp,self.it,self.batch,self.eta,self.lambd) #train network on random samples
            self.a+=1
            if showprogress: print('Trained Network number', self.a)
        print('Training ',self.n, 'Networks finished')
        
    def output(self,x,show=False):
        '''
        With show returns all the single outputs. 
        Without show just returns the arrray that contains the summed up information.'''
        self.x=x
        self.rlst=[]
        self.summ=np.zeros(self.layers[-1])
        for N in self.Networks:
            self.ans=N.feedforward(self.x)
            self.rlst.append((self.ans.argmax(),self.ans))
            self.summ+=self.ans
        self.rlst.append(('Total Answer is: ',self.summ.argmax(),self.summ))
        if show:
            return self.rlst   
        else: 
            return self.rlst[-1][-1]
        
    def validate(self, Xv,Yv,subval=False):
        '''
        validate the network on it's training accuracy
        if subval validate each single network and print it all
        '''
        self.Xv=Xv
        self.Yv=Yv
        self.right=0
        self.total=len(Yv)
        for i in range(self.total):
            if self.output(Xv[i]).argmax()==Yv[i].argmax():
                self.right+=1
        if subval:
            for j in range(self.n):
                print('Accuracy of Network ',j,': ', self.Networks[j].validate(self.Xv,self.Yv))
        return(self.right/self.total)
        print('Total examples: ',self.total,'\nCorrect examples: ', self.right,'\nAccuracy: ',self.right/self.total)
         

        
Hambi=Forest(n=5)
Hambi.train(Xt,Yt,size=10000)
Hambi.validate(Xv,Yv,True)
#Here I'm gonna add a function to actually split the training set

class Forestplus(Forest):
    '''These class is supposed to add the posibility of splitting the data and training
        each network only on one part of the data'''
    
    def train(self, X,Y,size,split,iterations=10,batchsize=10,eta=0.35,lambd=0.0013,showprogress=False):
        '''
        trains n classifiers that each see only a random part of the data given
        how much of the data has to be specified via the size variable.
        '''
        self.split=split
        self.size=size
        self.splitsize=int(len(Y)//self.split)
        self.X=X
        self.Y=Y
        self.it=iterations
        self.batch=batchsize
        self.eta=eta
        self.lambd=lambd
        self.a=0
        for N in self.Networks: #iterate over the n networks
            self.index=np.random.choice(np.random.choice(len(self.Y),self.splitsize,replace=False), self.size) 
            #choose random samples out of the randomly splitted data
            self.Xtemp=self.X[self.index]
            self.Ytemp=self.Y[self.index]
            N.train(self.Xtemp,self.Ytemp,self.it,self.batch,self.eta,self.lambd) #train network on random samples
            self.a+=1
            if showprogress: print('Trained Network number', self.a)
        print('Training ',self.n, 'Networks finished')
        